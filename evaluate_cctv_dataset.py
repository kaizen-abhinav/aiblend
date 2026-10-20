"""
Comprehensive CCTV Shoplifting Dataset Evaluation & HUD Video Renderer.
Processes all 8 surveillance videos in CCTV_Shoplifting_Dataset/videos:
- Generates fully annotated HUD surveillance videos
- Captures incident snapshots
- Computes Confusion Matrix, Accuracy, Recall, Precision, F1-Score
- Exports detailed CSV and JSON reports
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import config
from detector import ShopliftingDetector

DATASET_VIDEOS_DIR = ROOT_DIR / "CCTV_Shoplifting_Dataset" / "videos"
OUTPUT_VIDEOS_DIR = ROOT_DIR / "data" / "annotated_cctv"
SNAPSHOTS_DIR = config.SNAPSHOT_DIR

OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def process_and_evaluate_video(detector: ShopliftingDetector, video_path: Path, save_video: bool = True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "file": video_path.name,
            "error": "Failed to open video",
            "max_threat": 0.0,
            "alert_count": 0,
            "reasons": []
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize target to match webapp standard width
    target_w = 854 if width > 854 else width
    target_h = int(height * (target_w / width))

    writer = None
    out_video_path = None
    if save_video:
        out_video_filename = f"{video_path.stem}_sentinel_hud.mp4"
        out_video_path = OUTPUT_VIDEOS_DIR / out_video_filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (target_w, target_h))

    detector.reset()
    max_threat = 0.0
    all_reasons = []
    alert_events = []
    frame_idx = 0
    t_start = time.time()

    # Ground truth determination from filename
    is_ground_truth_shoplifting = ("shoplifting" in video_path.stem.lower() and 
                                  "not_" not in video_path.stem.lower())

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        sim_time = frame_idx / fps

        if frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h))

        annotated_frame, alerts, telemetry = detector.process_frame(frame, sim_time)

        if telemetry["max_threat"] > max_threat:
            max_threat = telemetry["max_threat"]

        if alerts:
            for a in alerts:
                all_reasons.extend(a.get("reasons", []))
                # Save snapshot if high threat
                if a["threat_score"] >= (config.THREAT_SUSPICIOUS * 100):
                    snap_filename = f"cctv_{video_path.stem}_f{frame_idx:03d}_{int(a['threat_score'])}pct.jpg"
                    snap_path = SNAPSHOTS_DIR / snap_filename
                    cv2.imwrite(str(snap_path), annotated_frame)
                    a["snapshot_file"] = snap_filename
                    alert_events.append(a)

        if writer is not None:
            writer.write(annotated_frame)

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = max(0.001, time.time() - t_start)
    is_predicted_shoplifting = max_threat >= (config.THREAT_SUSPICIOUS * 100)

    return {
        "file": video_path.name,
        "ground_truth": "shoplifting" if is_ground_truth_shoplifting else "normal",
        "predicted_label": "shoplifting" if is_predicted_shoplifting else "normal",
        "predicted_alert": is_predicted_shoplifting,
        "frames_total": total_frames,
        "duration_sec": round(total_frames / fps, 1) if fps > 0 else 0,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(frame_idx / elapsed, 1),
        "max_threat": round(max_threat, 1),
        "alerts_count": len(alert_events),
        "key_reasons": list(set(all_reasons)),
        "annotated_video": str(out_video_path.name) if out_video_path else None
    }


def run_full_dataset_evaluation():
    if not DATASET_VIDEOS_DIR.exists():
        print(f"[Error] Dataset directory '{DATASET_VIDEOS_DIR}' not found.")
        return

    video_files = sorted(list(DATASET_VIDEOS_DIR.glob("*.mp4")))
    if not video_files:
        print(f"[Error] No MP4 videos found in '{DATASET_VIDEOS_DIR}'.")
        return

    print("=" * 80)
    print("      SENTINEL-CV | CCTV SHOPLIFTING DATASET BATCH EVALUATION PIPELINE")
    print(f"      Source Directory: {DATASET_VIDEOS_DIR}")
    print(f"      Total Videos to Process: {len(video_files)}")
    print(f"      Alert Threat Threshold: {int(config.THREAT_SUSPICIOUS * 100)}%")
    print("=" * 80)

    detector = ShopliftingDetector()
    results = []

    tp, fp, tn, fn = 0, 0, 0, 0

    for i, vpath in enumerate(video_files, 1):
        gt_type = "SHOPLIFTING" if ("shoplifting" in vpath.stem and "not_" not in vpath.stem) else "NORMAL"
        print(f"\n[{i}/{len(video_files)}] Processing {vpath.name} (GT: {gt_type})...")
        res = process_and_evaluate_video(detector, vpath, save_video=True)
        results.append(res)

        gt = res["ground_truth"]
        pred = res["predicted_label"]

        if gt == "shoplifting" and pred == "shoplifting":
            tp += 1
            verdict = "✓ TRUE POSITIVE (ALERT TRIGGERED)"
        elif gt == "shoplifting" and pred == "normal":
            fn += 1
            verdict = "✗ FALSE NEGATIVE (MISSED SHOPLIFTING)"
        elif gt == "normal" and pred == "normal":
            tn += 1
            verdict = "✓ TRUE NEGATIVE (CORRECTLY SAFE)"
        else:
            fp += 1
            verdict = "✗ FALSE POSITIVE (FALSE ALARM)"

        reasons = "; ".join(res["key_reasons"]) if res["key_reasons"] else "None"
        print(f"    -> Max Threat: {res['max_threat']}% | Verdict: {verdict}")
        print(f"    -> Key Flags: {reasons}")
        print(f"    -> Rendered HUD Video: {res['annotated_video']} ({res['duration_sec']}s @ {res['fps']} FPS)")

    # Metrics calculation
    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(0.0001, (precision + recall))

    # Save CSV Report
    csv_path = ROOT_DIR / "data" / "cctv_evaluation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Ground Truth", "Predicted Label", "Max Threat %", "Alert Count", "Duration (s)", "Inference FPS", "Annotated Video", "Flags"])
        for r in results:
            writer.writerow([
                r["file"],
                r["ground_truth"],
                r["predicted_label"],
                r["max_threat"],
                r["alerts_count"],
                r["duration_sec"],
                r["fps"],
                r["annotated_video"],
                "; ".join(r["key_reasons"])
            ])

    # Save JSON Report
    json_path = ROOT_DIR / "data" / "cctv_evaluation_report.json"
    summary_data = {
        "dataset": "CCTV_Shoplifting_Dataset",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(results),
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "metrics": {
            "accuracy": round(accuracy * 100, 1),
            "recall": round(recall * 100, 1),
            "precision": round(precision * 100, 1),
            "f1_score": round(f1 * 100, 1)
        },
        "video_results": results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("                    DATASET EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total Videos Processed:     {len(results)}")
    print(f"  True Positives  (TP):       {tp}  (Shoplifting correctly detected)")
    print(f"  True Negatives  (TN):       {tn}  (Normal behavior cleared)")
    print(f"  False Positives (FP):       {fp}  (Normal behavior flagged)")
    print(f"  False Negatives (FN):       {fn}  (Missed shoplifting)")
    print("-" * 80)
    print(f"  Overall Accuracy:           {accuracy * 100:.1f}%")
    print(f"  Detection Recall:           {recall * 100:.1f}%")
    print(f"  Precision:                  {precision * 100:.1f}%")
    print(f"  F1-Score:                   {f1 * 100:.1f}%")
    print("-" * 80)
    print(f"  CSV Report Saved:           {csv_path}")
    print(f"  JSON Report Saved:          {json_path}")
    print(f"  Rendered HUD Videos Saved:  {OUTPUT_VIDEOS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_dataset_evaluation()
