# GPR Anomaly Detection & Annotation Pipeline

A complete end-to-end pipeline for **Ground Penetrating Radar (GPR)** data:
- Converts raw ASCII volumes into high-resolution B-scans (real & migrated modes).
- Runs object detection (YOLO / RCNN) on B-scans.
- Matches detections between migrated & real scans.
- Maps bounding boxes back to GPS coordinates.
- Exports Examiner-compatible annotations for visualizing anomalies.

---

## 📌 Features
- Parse Examiner ASCII volume files with latitude & longitude.
- Generate B-scans in both real & migrated modes.
- YOLOv8 inference with bounding box matching.
- Convert pixel detections → GPS → `.annot` format for Examiner.
- Organized output by survey slices.
