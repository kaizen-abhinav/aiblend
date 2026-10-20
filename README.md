# SENTINEL-CV | AI Shoplifting Detection & Behavioral Surveillance

Real-time intelligent shoplifting detection and surveillance dashboard powered by FastAPI, OpenCV, and YOLOv8 Pose estimation.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-Pose-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**SENTINEL-CV** analyzes surveillance video streams (CCTV dataset footage, synthetic retail demo, live webcams, or uploaded footage) in real time to detect potential shoplifting behaviors. It tracks individuals using ByteTrack, reconstructs human pose skeletons, and computes threat scores based on biomechanical kinematics:

- **Shelf Reaching $\rightarrow$ Pocket Concealment Sequence**: Detects reaching up or horizontally to shelves followed by rapid hand-to-pocket dwell.
- **Pocket / Waistband Dwell**: Tracks prolonged wrist positioning within hip / pocket concealment zones.
- **Jacket / Inner Torso Concealment**: Monitors hands hovering inside jacket or cross-body upper chest areas.
- **Bag / Backpack Stashing**: Detects two hands manipulating merchandise together in lap or low-slung bags.
- **Crouching / Low-Level Loitering**: Detects prolonged ducking or kneeling near lower shelves.

---

## Features

- **Live Tactical Surveillance HUD**: Real-time bounding boxes, skeleton bone overlays, concealment zone markers, and wrist trajectory trails.
- **Integrated CCTV Shoplifting Dataset**: Includes 8 benchmark surveillance videos (`shoplifting1-4.mp4` and `not_shoplifting1-4.mp4`) with 1-click live switching and auto-cycle playback.
- **Multi-Source Streaming**:
  - CCTV Shoplifting Dataset videos with continuous Auto-Cycle mode
  - Built-in synthetic retail CCTV demo generator (`test_video_generator.py`)
  - Live webcam feed (`/dev/video0` or USB cameras)
  - Custom video file upload (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **Interactive Web Command Center**: Dark tactical UI with live threat telemetry, active target statistics, incident logs with snapshots, and alert sirens.
- **WebSocket Telemetry**: Low-latency bidirectional communication for real-time telemetry updates.
- **Automated Evidence Snapshots & Export**: Automated high-res annotated frame capture upon incident trigger, downloadable in JSON or CSV.
- **Comprehensive Benchmark Suite**: Automated evaluation reporting with confusion matrix, recall, precision, and F1-score.

---

## Project Structure

```
├── app.py                      # FastAPI application, streaming endpoints, and WebSocket manager
├── config.py                   # Path configurations, model names, and heuristic thresholds
├── detector.py                 # Inference pipeline & OpenCV tactical HUD rendering engine
├── tracker.py                  # Multi-person tracking and behavioral kinematic threat scoring
├── evaluate_cctv_dataset.py    # Batch evaluation & HUD video generator for CCTV dataset
├── batch_test.py               # Batch evaluation suite across surveillance datasets
├── test_pipeline.py            # End-to-end integration and API verification test suite
├── test_video_generator.py     # Synthetic retail store surveillance video generator
├── CCTV_Shoplifting_Dataset/   # Benchmark dataset (videos, labels, and VLM metadata)
│   ├── videos/                 # MP4 test surveillance clips (shoplifting & normal browsing)
│   ├── labels/                 # Keypoint and bounding box labels
│   └── VLM-labels/             # Ground-truth chronological sequence metadata
├── data/
│   ├── cctv_evaluation_report.csv   # Evaluation CSV summary across dataset
│   └── cctv_evaluation_report.json  # Detailed evaluation JSON metrics
├── templates/
│   └── index.html              # Tactical surveillance command center UI
├── static/
│   ├── css/
│   │   └── dashboard.css       # Dark glassmorphism styles & responsive HUD grid
│   └── js/
│       └── dashboard.js        # Client-side streaming, WebSocket handling, and controls
├── requirements.txt            # Python package dependencies
└── .gitignore                  # Ignored files, weights, and temporary artifacts
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kaizen-abhinav/aiblend.git
cd aiblend
```

### 2. Create and Activate a Virtual Environment

```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch & Ultralytics**: If you have an NVIDIA GPU with CUDA, install the CUDA-enabled PyTorch build for hardware acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Running the System

### Starting the Live Surveillance Dashboard

Launch the server with Uvicorn:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Using the Surveillance Dashboard

1. **Stream CCTV Dataset Videos**:
   - In the top navigation bar, click the **CCTV Dataset** dropdown pill.
   - Select any video (`shoplifting1` through `not_shoplifting4`). The live stream immediately switches and tracks subjects with the tactical HUD.
2. **Auto-Cycle Surveillance**:
   - Check the **Auto-Cycle** checkbox in the dropdown. The system will continuously loop through each dataset video sequentially.
3. **Live Webcam**:
   - Click the **Live Webcam** pill to switch to an attached webcam or camera feed.
4. **Upload Custom Surveillance Footage**:
   - Click **Upload Video** to analyze custom `.mp4`, `.avi`, or `.mov` files.
5. **Inspect Incidents**:
   - When a threat score exceeds 45%, the system triggers an audible security alert and saves a high-resolution evidence snapshot to `snapshots/`.
   - Click **View Evidence** on any incident row or click thumbnails in the gallery strip to inspect full biometric telemetry.
6. **Dataset Report**:
   - Click the **Dataset Report** button in the incident log header to view the benchmark confusion matrix and performance statistics.

### Stopping the Server

Press `Ctrl + C` in your terminal, or on Linux run:
```bash
pkill -f "uvicorn"
```

---

## CCTV Shoplifting Dataset Evaluation

Run the automated evaluation pipeline across all 8 surveillance benchmark videos:

```bash
python evaluate_cctv_dataset.py
```

This runs inference, generates fully annotated HUD output videos in `data/annotated_cctv/`, captures incident snapshots in `snapshots/`, and outputs summary reports:
- [`data/cctv_evaluation_report.csv`](data/cctv_evaluation_report.csv)
- [`data/cctv_evaluation_report.json`](data/cctv_evaluation_report.json)

### Benchmark Results Summary

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Detection Recall (Sensitivity)** | **100.0%** | **4 / 4 shoplifting incidents detected** (0 false negatives / 0 missed thefts) |
| **Peak Shoplifting Threat Score** | **95.0%** | Reached high-alert threshold on all theft sequences |
| **Average Processing Speed** | **~9.5 FPS** | Full pose tracking + HUD rendering on CPU |
| **Dataset F1-Score** | **66.7%** | Precision affected by counter-height hand dwell in normal videos |

---

## Verification & Testing

Run the automated end-to-end verification test suite:

```bash
python test_pipeline.py
```

This validates:
1. Configuration directories (`data/`, `uploads/`, `snapshots/`)
2. Synthetic retail CCTV demo generation
3. Kinematic tracker state machine and dwell time scoring
4. YOLOv8-Pose detection and tactical HUD rendering
5. FastAPI REST API streaming and management endpoints

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web surveillance command center UI |
| `GET` | `/video_feed` | Multipart MJPEG real-time video stream with tactical HUD |
| `WS` | `/ws` | Bidirectional WebSocket for live biometrics telemetry and alerts |
| `GET` | `/api/dataset_videos` | List available CCTV Shoplifting Dataset videos with metadata |
| `POST` | `/api/select_dataset_video` | Switch active stream to a specific dataset video (`video` parameter) |
| `POST` | `/api/toggle_auto_cycle` | Enable/disable automatic sequential cycling through dataset videos |
| `POST` | `/api/switch_source` | Switch between `demo`, `webcam`, and `dataset` sources |
| `POST` | `/api/upload` | Upload custom video file for immediate analysis |
| `GET` | `/api/incidents` | Retrieve full incident log history with snapshot links |
| `GET` | `/api/cctv_evaluation` | Retrieve benchmark metrics and confusion matrix |
| `GET` | `/api/export?format=csv` | Download incident history report as CSV |
| `GET` | `/api/status` | System health check and diagnostic telemetry |

---

## License

This project is licensed under the MIT License.
