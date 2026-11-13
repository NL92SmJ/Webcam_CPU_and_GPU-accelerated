# Deep Dive: Systems-Level Real-Time Telemetry & Hardware Benchmarking – Webcam AI-Upscaling Pipelines

## 1. Motivation & Problem Statement
Modern computer vision applications, real-time video streaming, and edge AI workloads frequently hit severe hardware bottlenecks when scaling beyond standard definitions. Consumer-grade webcams are frequently limited by native resolution constraints, heavy compression artifacts (such as MJPEG encapsulation over USB 2.0 limitations), and extreme CPU overhead during raw frame acquisition, colour-space conversion, and pipeline orchestration.

This project explores the performance, latency characteristics, and hardware footprint of scaling a live video streaming pipeline from a naive CPU-bound baseline to a hardware-accelerated deep learning architecture using an Efficient Sub-Pixel Convolutional Network (ESPCN) with residual connections, as well as an industrial zero-copy preprocessing pipeline. Operating on a constrained consumer system—an **NVIDIA RTX 3050 (6 GB VRAM)** paired with an **Intel Core i3-12100F**—this benchmark investigates how hardware constraints, USB I/O bounds, memory transfers, and neural network inference impact real-time frame rates, thermal stability, and frame-pacing consistency.

---

## 2. Test Environment & Hardware Setup
- **CPU:** Intel Core i3-12100F (4 cores / 8 threads)
- **System RAM:** 16 GB DDR4
- **GPU:** NVIDIA GeForce RTX 3050 — 6 GB VRAM
- **Camera:** 1080p Webcam (running via USB 2.0 / MJPEG fallback at 1280x720 @ 30 FPS)
- **Software Stack:** Ubuntu 22.04 LTS, Python 3.11.15, PyTorch (CUDA backend), OpenCV, Pandas, Matplotlib, Pynvml

---

## 3. Deep-Dive Insight: The USB I/O Bottleneck & The "Capture Wall"
During extended stress tests comparing the CPU-baseline against GPU-accelerated pipelines, a fascinating systems-level phenomenon emerged: **Both pipelines locked in at an almost identical average framerate of ~30.68 to 30.72 FPS.**

### Why It Happens: The Capture Wall
In a real-time computer vision pipeline, throughput can never exceed the acquisition rate of the input device. Because the webcam was constrained by USB 2.0 architecture (`480M`), the UVC driver forced an MJPEG compression stream that capped raw frame delivery at 30 FPS. 
- **The Implication:** Rohe GPU-Rechenpower läuft ins Leere, solange das I/O-Subsystem (Kamera/USB-Bus) als harter Flaschenhals vor der Pipeline liegt. Weder CPU noch GPU konnten ihr volles Potenzial entfalten, weil sie auf den Takt der Kamera warten mussten (`cap.read()` blockierend).

---

## 4. Comprehensive Benchmark Results

### A. AI-Upscaling & 5-Minute Stress Test
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

### B. Industrial Preprocessing & Zero-Copy Comparison (CPU vs. GPU Zero-Copy)
To evaluate the impact of eliminating CPU-side staging buffers and PCIe transport overhead, an automated 10-minute stress test compared standard CPU-baseline preprocessing against a GPU-accelerated zero-copy pipeline:

| Metric | CPU-Baseline (OpenCV + PCIe H2D) | GPU-Accelerated (Zero-Copy Pipeline) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Average Framerate** | 30.72 FPS | 30.68 FPS | Bound by USB 2.0 capture wall |
| **FPS Jitter (StdDev)** | 4.88 FPS | 4.94 FPS | Comparable temporal stability |
| **1% Lows (Stutter Metric)** | 23.82 FPS | 24.70 FPS | +3.7% improved tail-latency stability |
| **Ø Preprocessing Time** | 0.35 ms | 0.31 ms | Optimized tensor execution |
| **Ø PCIe H2D Transfer Time** | **2.65 ms** | **0.00 ms** | **100% elimination of bus overhead** |
| **Ø End-to-End Latency** | **3.16 ms** | **1.02 ms** | **~67.7% latency reduction** |
| **Ø Power Consumption (GPU)**| 18.96 W | 19.31 W | Highly stable thermal envelope |
| **Max GPU Temperature** | 39.0 °C | 40.0 °C | Nominal thermal delta (+1.0 °C) |

---

## 5. Real-World Implications & Systems-Level Learnings

1. **The Power Consumption Trade-off (FPS per Watt):**
   While deep learning upscaling delivered visually superior frames, it increased active GPU power draw up to **28.83W** (raising core temperatures to 49°C). In constrained edge-AI environments, matching model complexity directly to hardware utility is critical.
