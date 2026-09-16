# Real-Time Hand Centroid Tracking with Custom YOLOv8

A lightweight computer vision pipeline designed for real-time hand detection and centroid spatial extraction using a custom-trained **YOLOv8** model and OpenCV.

This module provides 2D spatial coordinates of detected targets in real time, serving as the foundational detection layer for autonomous gimbal pointing, optical tracking, and human-machine interaction systems.

---

## Overview

The application processes live camera streams, feeds frames into an optimized YOLOv8 neural network inference loop, and computes the exact pixel centroid `(cx, cy)` of detected targets directly from the bounding box tensors (`xywh`). Dynamic hardware acceleration automatically targets NVIDIA CUDA GPUs when available, falling back seamlessly to CPU execution.

### Key Capabilities

* **Custom Model Inference:** Powered by an ultra-fast custom-trained YOLOv8 weights file (`hand.pt`).
* **Instant Centroid Extraction:** Computes geometric center coordinates per detected instance with minimal mathematical overhead.
* **Dynamic Hardware Dispatch:** Automatically selects `CUDA` or `CPU` execution based on runtime hardware detection.
* **Low-Latency Stream:** Designed with lean visualization routines to maximize real-time inference FPS.

---

## Pipeline Workflow

```
[ Camera Stream (640x480) ]
             │
             ▼
[ Frame Ingestion & Mirroring ]
             │
             ▼
[ YOLOv8 Inference Engine (hand.pt) ]
             │
             ▼
[ Tensor Extraction (xywh & conf) ]
             │
             ▼
[ Centroid Computation (cx, cy) ]
             │
             ▼
[ Real-Time Display & Vector Output ]

```

---

## Getting Started

### Prerequisites

Clone the repository and install the dependencies:

```
pip install -r requirements.txt

```

### Weights File

Ensure your trained weights file named `hand.pt` is placed in the root directory alongside the script.

### Running the Tracker

Execute the real-time tracking interface:

```
python hand_tracker.py

```

* **Q:** Terminate video capture, destroy windows, and release camera resources.

---

## Use Cases & Integration

* **Gimbal & Servos:** Forward coordinate errors `(cx - center_x, cy - center_y)` to flight controllers or microcontrollers for closed-loop visual tracking.
* **HMI Interfaces:** Gesture-free cursor control and optical spatial reference logging.
