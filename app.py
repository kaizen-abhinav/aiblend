"""
FastAPI Surveillance Web Application for Real-Time Shoplifting Detection.
Provides MJPEG video streaming, WebSocket telemetry, source switching,
incident logging, and REST management endpoints.
"""

import os
import cv2
import time
import json
import uuid
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from detector import ShopliftingDetector
from test_video_generator import generate_shoplifting_demo_video

app = FastAPI(title="SENTINEL-CV | AI Shoplifting Detection")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
app.mount("/snapshots", StaticFiles(directory=str(config.SNAPSHOT_DIR)), name="snapshots")
ANNOTATED_DIR = config.BASE_DIR / "data" / "annotated_cctv"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/annotated_videos", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated_videos")
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))

# Global State & Video Manager
class VideoStreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_source_type = "demo"  # "demo", "webcam", "file"
        self.current_file_path = None
        self.cap = None
        self.running = True
        self.detector = ShopliftingDetector()
        
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_telemetry = {
            "fps": 0.0,
            "active_targets": 0,
            "system_status": "NORMAL",
            "max_threat": 0.0,
            "persons": [],
            "source": "demo"
        }
        self.active_alerts = []
        self.all_incidents = []
        self.connected_websockets: List[WebSocket] = []
        self.main_loop: asyncio.AbstractEventLoop = None

        # CCTV Shoplifting Dataset configuration
        self.dataset_dir = config.BASE_DIR / "CCTV_Shoplifting_Dataset" / "videos"
        self.dataset_videos = sorted([f.name for f in self.dataset_dir.glob("*.mp4")]) if self.dataset_dir.exists() else []
        self.current_dataset_video = None
        self.current_dataset_idx = 0
        self.auto_cycle = False
        
        # Ensure demo video exists
        self.demo_video_path = config.DATA_DIR / "demo_shoplifting.mp4"
        if not self.demo_video_path.exists():
            print("[StreamManager] Creating demo surveillance video...")
            generate_shoplifting_demo_video(str(self.demo_video_path))
            
        self._open_source("demo")
        
        # Start background processing thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def _open_source(self, source_type: str, file_path: str = None):
        """Open or change video capture source."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                
            self.current_source_type = source_type
            
            if source_type == "webcam":
                # Try opening index 0
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("[StreamManager] Warning: Webcam 0 failed to open. Falling back to demo video.")
                    self.current_source_type = "demo"
                    self.cap = cv2.VideoCapture(str(self.demo_video_path))
            elif source_type == "file" and file_path and os.path.exists(file_path):
                self.current_file_path = file_path
                self.cap = cv2.VideoCapture(file_path)
            else:
                self.current_source_type = "demo"
                self.cap = cv2.VideoCapture(str(self.demo_video_path))
                
            # Reset tracker states so previous tracks do not contaminate the new stream
            self.detector.reset()
            self.latest_telemetry["persons"] = []
            print(f"[StreamManager] Switched source to: {self.current_source_type}")

    def switch_source(self, source_type: str, file_path: str = None):
        self._open_source(source_type, file_path)

    def switch_to_dataset_video(self, video_name: str):
        """Switch surveillance feed to a specific video in CCTV_Shoplifting_Dataset/videos."""
        if video_name in self.dataset_videos:
            self.current_dataset_idx = self.dataset_videos.index(video_name)
        self.current_dataset_video = video_name
        target = self.dataset_dir / video_name
        self._open_source("file", str(target))
        self.current_source_type = "dataset"

    def _processing_loop(self):
        """Continuous video frame acquisition, inference, and alert handling."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # Cycle or loop videos seamlessly
                if self.auto_cycle and self.dataset_videos and self.current_source_type in ("file", "dataset"):
                    self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.dataset_videos)
                    next_vid = self.dataset_videos[self.current_dataset_idx]
                    self.current_dataset_video = next_vid
                    self._open_source("file", str(self.dataset_dir / next_vid))
                    self.current_source_type = "dataset"
                    self._broadcast_event({
                        "type": "DATASET_VIDEO_CHANGED",
                        "video": next_vid
                    })
                    continue
                elif self.current_source_type in ("demo", "file", "dataset"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                else:
                    time.sleep(0.05)
                    continue
                    
            # Resize large frames for speed & consistent inference scale
            h, w = frame.shape[:2]
            if w > 854:
                scale = 854.0 / w
                frame = cv2.resize(frame, (854, int(h * scale)))
                
            now = time.time()
            annotated_frame, new_alerts, telemetry = self.detector.process_frame(frame, now)
            telemetry["source"] = self.current_source_type
            if self.current_dataset_video and self.current_source_type == "dataset":
                telemetry["dataset_video"] = self.current_dataset_video
            
            # Handle new alert events (save snapshots & record incident)
            for alert in new_alerts:
                snapshot_id = f"inc_{int(now)}_{uuid.uuid4().hex[:6]}"
                snapshot_filename = f"{snapshot_id}.jpg"
                snapshot_path = config.SNAPSHOT_DIR / snapshot_filename
                
                # Save high-res evidence snapshot
                cv2.imwrite(str(snapshot_path), annotated_frame)
                
                incident_record = {
                    "id": snapshot_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert["timestamp"])),
                    "person_id": alert["person_id"],
                    "threat_score": alert["threat_score"],
                    "threat_level": alert["threat_level"],
                    "pocket_conf": alert["pocket_conf"],
                    "chest_conf": alert["chest_conf"],
                    "bag_conf": alert.get("bag_conf", 0.0),
                    "sequence_conf": alert["sequence_conf"],
                    "crouch_conf": alert["crouch_conf"],
                    "reasons": alert["reasons"],
                    "snapshot_url": f"/snapshots/{snapshot_filename}"
                }
                self.all_incidents.insert(0, incident_record)
                if len(self.all_incidents) > 100:
                    self.all_incidents.pop()
                    
                # Broadcast alert to WebSockets asynchronously
                self._broadcast_event({
                    "type": "NEW_ALERT",
                    "incident": incident_record
                })

            with self.lock:
                self.latest_raw_frame = frame
                self.latest_annotated_frame = annotated_frame
                self.latest_telemetry = telemetry

            # Modulate frame rate if running on pre-recorded demo video to keep real-time ~25 FPS
            if self.current_source_type in ("demo", "file"):
                time.sleep(0.028)
            else:
                time.sleep(0.005)

    def _broadcast_event(self, data: dict):
        """Broadcast payload to connected websockets safely across threads."""
        if not self.connected_websockets:
            return
            
        async def _send():
            msg = json.dumps(data)
            for ws in list(self.connected_websockets):
                try:
                    await ws.send_text(msg)
                except Exception:
                    if ws in self.connected_websockets:
                        self.connected_websockets.remove(ws)
                        
        if self.main_loop and self.main_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(_send(), self.main_loop)
            except Exception:
                pass

    def get_jpeg_frame(self) -> bytes:
        """Encode current annotated frame into JPEG for MJPEG stream."""
        with self.lock:
            frame = self.latest_annotated_frame
            if frame is None:
                # Return placeholder frame if not ready
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "INITIALIZING CV PIPELINE...", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes() if ret else b""


# Singleton Video Stream Manager
stream_mgr = VideoStreamManager()


@app.on_event("startup")
async def startup_event():
    """Capture main asyncio event loop for thread-safe WebSocket broadcasts."""
    stream_mgr.main_loop = asyncio.get_running_loop()


# -------------------------------------------------------------
# Streaming Routes
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


def mjpeg_generator():
    """Generator for streaming multipart JPEG frames."""
    while True:
        frame_bytes = stream_mgr.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.035)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------------------------------------------
# WebSocket Telemetry
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_mgr.connected_websockets.append(websocket)
    last_sent_incident_id = stream_mgr.all_incidents[0]["id"] if stream_mgr.all_incidents else None
    try:
        # Initial greeting with recent incidents
        await websocket.send_text(json.dumps({
            "type": "INIT",
            "incidents": stream_mgr.all_incidents[:15],
            "config": {
                "show_skeleton": stream_mgr.detector.show_skeleton,
                "show_zones": stream_mgr.detector.show_zones,
                "show_trajectory": stream_mgr.detector.show_trajectory,
                "show_hud": stream_mgr.detector.show_hud,
                "detection_conf": stream_mgr.detector.detection_conf,
                "source": stream_mgr.current_source_type
            }
        }))
        
        while True:
            # Check for newly added incidents (double redundancy for instant delivery)
            if stream_mgr.all_incidents:
                new_incidents = []
                for inc in stream_mgr.all_incidents:
                    if inc["id"] == last_sent_incident_id:
                        break
                    new_incidents.append(inc)
                if new_incidents:
                    last_sent_incident_id = new_incidents[0]["id"]
                    for inc in reversed(new_incidents):
                        await websocket.send_text(json.dumps({
                            "type": "NEW_ALERT",
                            "incident": inc
                        }))

            # Send periodic telemetry updates
            data = {
                "type": "TELEMETRY",
                "telemetry": stream_mgr.latest_telemetry
            }
            await websocket.send_text(json.dumps(data))
            
            # Non-blocking receive with timeout
            try:
                incoming_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.15)
                msg = json.loads(incoming_text)
                action = msg.get("action")
                
                if action == "TOGGLE_FEATURE":
                    feature = msg.get("feature")
                    val = msg.get("value")
                    if feature == "skeleton":
                        stream_mgr.detector.show_skeleton = val
                    elif feature == "zones":
                        stream_mgr.detector.show_zones = val
                    elif feature == "trajectory":
                        stream_mgr.detector.show_trajectory = val
                    elif feature == "hud":
                        stream_mgr.detector.show_hud = val
                        
                elif action == "SET_CONFIDENCE":
                    conf_val = float(msg.get("value", 0.35))
                    stream_mgr.detector.detection_conf = conf_val
                    
                elif action == "SWITCH_SOURCE":
                    src = msg.get("source", "demo")
                    stream_mgr.switch_source(src)
                    
            except asyncio.TimeoutError:
                pass
                
    except WebSocketDisconnect:
        if websocket in stream_mgr.connected_websockets:
            stream_mgr.connected_websockets.remove(websocket)
    except Exception as e:
        if websocket in stream_mgr.connected_websockets:
            stream_mgr.connected_websockets.remove(websocket)


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.post("/api/switch_source")
async def switch_source_endpoint(source: str = Form(...)):
    """Switch active input between 'demo', 'webcam', and 'dataset'."""
    if source not in ("demo", "webcam", "dataset"):
        raise HTTPException(status_code=400, detail="Invalid source")
    if source == "dataset":
        first_vid = stream_mgr.dataset_videos[0] if stream_mgr.dataset_videos else None
        if first_vid:
            stream_mgr.switch_to_dataset_video(first_vid)
    else:
        stream_mgr.switch_source(source)
    return {"status": "success", "active_source": source}


@app.get("/api/dataset_videos")
async def get_dataset_videos():
    """List all available CCTV Shoplifting Dataset videos with status."""
    dataset_dir = config.BASE_DIR / "CCTV_Shoplifting_Dataset" / "videos"
    if not dataset_dir.exists():
        return {"videos": [], "active_video": None, "auto_cycle": False}
        
    videos = sorted([f.name for f in dataset_dir.glob("*.mp4")])
    annotated_dir = config.BASE_DIR / "data" / "annotated_cctv"
    
    video_list = []
    for f in videos:
        is_shop = "shoplifting" in f.lower() and "not_" not in f.lower()
        annotated_file = f"{Path(f).stem}_sentinel_hud.mp4"
        has_annotated = (annotated_dir / annotated_file).exists()
        video_list.append({
            "filename": f,
            "label": "Shoplifting Incident" if is_shop else "Normal Customer Behavior",
            "is_shoplifting": is_shop,
            "annotated_url": f"/annotated_videos/{annotated_file}" if has_annotated else None,
            "is_active": (stream_mgr.current_dataset_video == f and stream_mgr.current_source_type in ("file", "dataset"))
        })
    return {
        "videos": video_list,
        "active_video": stream_mgr.current_dataset_video,
        "auto_cycle": stream_mgr.auto_cycle
    }


@app.post("/api/select_dataset_video")
async def select_dataset_video(video: str = Form(...)):
    """Switch surveillance feed to a specific CCTV Shoplifting Dataset video."""
    dataset_dir = config.BASE_DIR / "CCTV_Shoplifting_Dataset" / "videos"
    target_path = dataset_dir / video
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video}' not found in CCTV dataset.")
        
    stream_mgr.switch_to_dataset_video(video)
    return {"status": "success", "active_video": video, "message": f"Now streaming CCTV video: {video}"}


@app.post("/api/toggle_auto_cycle")
async def toggle_auto_cycle_endpoint(enable: bool = Form(...)):
    """Toggle auto-cycling sequentially through all CCTV dataset videos."""
    stream_mgr.auto_cycle = enable
    if enable and stream_mgr.current_source_type not in ("file", "dataset") and stream_mgr.dataset_videos:
        stream_mgr.switch_to_dataset_video(stream_mgr.dataset_videos[0])
    return {"status": "success", "auto_cycle": stream_mgr.auto_cycle}


@app.get("/api/cctv_evaluation")
async def get_cctv_evaluation():
    """Retrieve full batch evaluation metrics across the CCTV Shoplifting Dataset."""
    json_path = config.BASE_DIR / "data" / "cctv_evaluation_report.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Evaluation report not yet generated"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload custom video for shoplifting analysis with format and size validation."""
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(sorted(allowed_exts))}"
        )

    safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{file.filename}"
    save_path = config.UPLOAD_DIR / safe_name
    
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > 200 * 1024 * 1024:  # 200MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 200MB.")

    with open(save_path, "wb") as f:
        f.write(content)
        
    # Verify OpenCV can read this video file
    test_cap = cv2.VideoCapture(str(save_path))
    if not test_cap.isOpened():
        test_cap.release()
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Invalid video file: OpenCV was unable to decode it.")
    ret, _ = test_cap.read()
    test_cap.release()
    if not ret:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail="Video contains no readable frames.")

    # Switch to uploaded video immediately
    stream_mgr.switch_source("file", str(save_path))
    return {"status": "success", "filename": safe_name, "message": "Video loaded and playing"}


@app.get("/api/incidents")
async def get_incidents():
    """Retrieve full incident history with snapshot URLs."""
    return {"incidents": stream_mgr.all_incidents}


@app.get("/api/export")
async def export_incidents(format: str = "json"):
    """Export incident history as JSON or CSV."""
    if format == "csv":
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Incident ID", "Timestamp", "Person ID", "Threat Level", "Risk Score %", "Pocket %", "Jacket %", "Sequence %", "Crouch %", "Reasons"])
        for inc in stream_mgr.all_incidents:
            writer.writerow([
                inc["id"],
                inc["timestamp"],
                inc["person_id"],
                inc["threat_level"],
                inc["threat_score"],
                inc["pocket_conf"],
                inc["chest_conf"],
                inc["sequence_conf"],
                inc["crouch_conf"],
                "; ".join(inc["reasons"])
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=shoplifting_incidents_report.csv"}
        )
    else:
        return JSONResponse(content=stream_mgr.all_incidents)


@app.get("/api/status")
async def get_status():
    """System health check and diagnostic telemetry."""
    import torch
    return {
        "status": "ONLINE",
        "gpu_available": torch.cuda.is_available(),
        "device": stream_mgr.detector.device,
        "active_source": stream_mgr.current_source_type,
        "fps": stream_mgr.latest_telemetry["fps"],
        "total_incidents": len(stream_mgr.all_incidents)
    }
