# Deep Dive: Systems-Level Real-Time Telemetry & Hardware Benchmarking – Webcam AI-Upscaling Pipelines

## 1. Motivation & Problem Statement
Modern computer vision applications, real-time video streaming, and edge AI workloads frequently hit severe hardware bottlenecks when scaling beyond standard definitions. Consumer-grade webcams are frequently limited by native resolution constraints, heavy compression artifacts (such as MJPEG encapsulation over USB 2.0 limitations), and extreme CPU overhead during raw frame acquisition, colour-space conversion, and pipeline orchestration.

This project explores the performance, latency characteristics, and hardware footprint of scaling a live video streaming pipeline from a naive CPU-bound baseline to a hardware-accelerated deep learning architecture using an Efficient Sub-Pixel Convolutional Network (ESPCN) with residual connections. Operating on a constrained consumer system—an **NVIDIA RTX 3050 (6 GB VRAM)** paired with an **Intel Core i3-12100F**—this benchmark investigates how hardware constraints, USB I/O bounds, memory transfers, and neural network inference impact real-time frame rates, thermal stability, and frame-pacing consistency.

---

## 2. Test Environment & Hardware Setup
- **CPU:** Intel Core i3-12100F (4 cores / 8 threads)
- **System RAM:** 16 GB DDR4
- **GPU:** NVIDIA GeForce RTX 3050 — 6 GB VRAM
- **Camera:** 1080p Webcam (running via USB 2.0 / MJPEG fallback at 1280x720 @ 30 FPS)
- **Software Stack:** Ubuntu 22.04 LTS, Python 3.11.15, PyTorch (CUDA backend), OpenCV, Pandas, Matplotlib, Pynvml

---

## 3. Deep-Dive Insight: The USB I/O Bottleneck & The "Capture Wall"
During extended 5-minute stress tests comparing the CPU-baseline against the GPU-accelerated pipeline, a fascinating systems-level phenomenon emerged: **Both pipelines locked in at an almost identical average framerate of ~30.68 to 30.70 FPS.**

### Why It Happens: The Capture Wall
In a real-time computer vision pipeline, throughput can never exceed the acquisition rate of the input device. Because the webcam was constrained by USB 2.0 architecture (`480M`), the UVC driver forced an MJPEG compression stream that capped raw frame delivery at 30 FPS. 
- **The Implication:** Rohe GPU-Rechenpower läuft ins Leere, solange das I/O-Subsystem (Kamera/USB-Bus) als harter Flaschenhals vor der Pipeline liegt. Weder CPU noch GPU konnten ihr volles Potenzial entfalten, weil sie auf den Takt der Kamera warten mussten (`cap.read()` blockierend).

---

## 4. Comprehensive 5-Minute Benchmark Results
The following empirical data was captured over continuous 300-second execution runs using precise CUDA events and NVML live hardware telemetry:

| Metrik | CPU-Baseline (OpenCV Resize) | GPU-Accelerated (ESPCN + PyTorch) |
| :--- | :--- | :--- |
| **Durchschnittliche FPS** | **30.70 FPS** | **30.68 FPS** |
| **FPS Jitter (StdDev)** | 4.99 FPS | 4.50 FPS |
| **1% Lows FPS** | 24.79 FPS | 24.19 FPS |
| **Ø Inferenz / E2E Latenz** | 0.20 ms | 13.31 ms |
| **Ø Leistungsaufnahme (GPU)** | **6.76 W** (IDLE State) | **28.83 W** (Active Load) |
| **Max. GPU Temperatur** | 30.0 °C | 49.0 °C |

<img width="3600" height="2400" alt="webcam_benchmark_comparison" src="https://github.com/user-attachments/assets/fca624d6-3488-4999-a596-ebedaaf0722a" />


---

## 5. Real-World Implications & Systems-Level Learnings

1. **The Power Consumption Trade-off (FPS per Watt):**
   While the GPU-accelerated pipeline delivered visually superior upscaled frames, it did so at a significant energy cost—increasing active GPU power draw from ~6.7W (IDLE) up to **28.83W** and raising core temperatures to 49°C. In constrained edge-AI environments, burning ~29W for a pipeline capped by a 30 FPS USB camera highlights the necessity of matching model complexity directly to hardware utility.
   
2. **Latency vs. Compute Complexity:**
   The CPU baseline achieved an almost negligible latency (~0.20 ms) because it relied on trivial bilinear interpolation (`cv2.resize`). Shifting the workload to the GPU introduced a measurable inference and transfer overhead (**13.31 ms**). However, because 13.31 ms sits safely below the strict 33.3 ms temporal boundary required for 30 FPS, the pipeline maintained real-time viability without dropping frames.

3. **The Danger of Misleading Benchmarks:**
   Looking purely at average FPS would suggest that "CPU and GPU perform identically." Only by incorporating telemetry (NVML power draw, temperature scaling, and latency breakdowns) does the true architectural difference become transparent.

---

## 6. Future Work & Advanced Optimization Roadmap
To bypass current I/O limitations and extract maximum enterprise-grade performance, future iterations will focus on:
- **TensorRT / ONNX Runtime Export:** Compiling the PyTorch model to TensorRT to strip away Python interpreter overhead and push inference latency deep into single-digit milliseconds.
- **Zero-Copy Memory Interop:** Eliminating CPU-side NumPy/OpenCV staging buffers entirely via direct CUDA-OpenGL/Vulkan interop.
- **High-Speed Synthetic Streams:** Removing the physical webcam constraint to benchmark absolute peak throughput and saturation limits of the RTX 3050.

---
