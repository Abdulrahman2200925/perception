# Perception & Autonomy Stack — Autonomous Ackermann Robot

Perception and detection pipeline for an autonomous Ackermann-steering robot,
developed as an engineering graduation capstone. This repository holds the
**perception layer** code: tooling to record, convert, label, and train a
LiDAR-based 3D object detector, plus supporting analysis scripts.

> **Note on scope:** this repo contains **code only**. Datasets (LiDAR bags,
> point clouds, labels) and model weights are large and live outside version
> control. Internal engineering notes are kept private.

## Project overview

The robot runs a four-layer autonomous stack:

1. **Perception** — a Livox Mid-360S LiDAR feeds a PointPillars 3D object
   detector (TensorRT), producing 3D bounding boxes.
2. **Reasoning** — a language-model agent reasons over the world state at low
   frequency and emits goal poses.
3. **Navigation** — Nav2 handles path planning and obstacle avoidance.
4. **Low-level control** — a microcontroller closes velocity/steering loops and
   enforces safety reflexes over CAN.

This repository covers the **perception layer** (and its data/training pipeline).

## Research direction

The detector is fine-tuned to close the **domain gap** between models trained on
automotive datasets (KITTI / Velodyne HDL-64, roof-mounted, forward-facing) and
a low-mounted, 360°, non-repetitive-scan solid-state LiDAR (Livox Mid-360S).
Self-collected and hand-labeled data plus fine-tuning is the core contribution.

## Hardware

- **Platform:** ROSMASTER R2L (Ackermann steering)
- **Compute:** NVIDIA Jetson Orin NX
- **LiDAR:** Livox Mid-360S (360° FOV, ~30 cm mount height)
- **Camera:** Logitech Brio 100 (for future LiDAR–camera fusion)
- **Low-level MCU:** TI TM4C123 (custom PCB)

## Pipeline

```
LiDAR bag ──▶ frame extraction ──▶ 3D labeling ──▶ label conversion
             (.bin / .pcd)         (annotator)     (→ training format)
                                                          │
                                                          ▼
deploy (TensorRT/ROS2) ◀── ONNX export ◀── fine-tune (PointPillars)
```

## Repository layout

```
mine/
├── bag_to_bin/     LiDAR bag → KITTI-style .bin point clouds (normalized intensity)
├── bag_to_pcd/     LiDAR bag → .pcd clouds for the 3D annotator
├── label_convert/  annotation labels → training-format labels (LiDAR frame)
└── analysis/       point-cloud / bag / label inspection utilities

configs/            reference training configuration
```

Datasets and trained weights are stored separately (cloud storage) and are not
tracked here.

## Tooling notes

- Point clouds are read by **field name** from the message metadata (no
  hardcoded byte offsets), so the tools adapt to different LiDAR field layouts.
- Intensity is normalized consistently between training data and the inference
  runtime.
- The training-side detector is built on
  [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and deployed via NVIDIA's
  [CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars).

## Status

Active capstone development. The runtime detection stack runs end-to-end on the
Jetson; current work focuses on improving detection quality on the target sensor
through fine-tuning on self-collected data.

## License

All rights reserved by the author pending a license decision.
