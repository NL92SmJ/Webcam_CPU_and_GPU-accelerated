import gc
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch

# --- KONFIGURATION ---
DURATION_SECONDS = 300  # 5 Minuten pro Modus (300 Sekunden)
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720  # Auflösung (stabil für USB-2.0/MJPEG)
TARGET_FPS = 30

print(
    "[INFO] Initialisiere NVML für Energie- und Thermik-Telemetrie auf RTX"
    " 3050..."
)
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


# --- MODELL-DEFINITION (ESPCN / Residual) ---
class ESPCNUpscaler(torch.nn.Module):

  def __init__(self, scale_factor=2):
    super().__init__()
    self.conv = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 3 * (scale_factor**2), 3, padding=1),
        torch.nn.PixelShuffle(scale_factor),
    )

  def forward(self, x):
    base = torch.nn.functional.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False
    )
    return self.conv(x) + base


def run_benchmark_mode(mode_name):
  print(
      "\n"
      + "=" * 60
      + f"\n[START] Starte 5-Minuten-Benchmark im Modus: [{mode_name.upper()}]\n"
      + "=" * 60
  )

  device = torch.device("cuda" if mode_name == "gpu" else "cpu")
  model = ESPCNUpscaler().eval().to(device)

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
  cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

  if not cap.isOpened():
    print("[ERROR] Konnte Webcam nicht öffnen!")
    return None

  latencies = []
  h2d_times = []
  power_draws = []
  temperatures = []
  frametimes = []

  gc.disable()  # Verhindere GC-Spikes

  start_evt = torch.cuda.Event(enable_timing=True)
  h2d_end_evt = torch.cuda.Event(enable_timing=True)
  inf_end_evt = torch.cuda.Event(enable_timing=True)

  test_start_time = time.perf_counter()
  prev_time = test_start_time

  while (time.perf_counter() - test_start_time) < DURATION_SECONDS:
    ret, frame_bgr = cap.read()
    if not ret:
      print("[WARN] Kein Frame von der Webcam erhalten.")
      break

    loop_start = time.perf_counter()
    start_evt.record() if mode_name == "gpu" else None

    if mode_name == "gpu":
      tensor_cpu = (
          torch.from_numpy(frame_bgr)
          .permute(2, 0, 1)
          .float()
          .div(255.0)
          .unsqueeze(0)
      )
      tensor_gpu = tensor_cpu.to(device, non_blocking=True)
      h2d_end_evt.record()

      with torch.inference_mode():
        _ = model(tensor_gpu)
      inf_end_evt.record()
      torch.cuda.synchronize()

      h2d_ms = start_evt.elapsed_time(h2d_end_evt)
      inf_ms = h2d_end_evt.elapsed_time(inf_end_evt)
      e2e_ms = start_evt.elapsed_time(inf_end_evt)
    else:
      h2d_ms = 0.0
      t_inf = time.perf_counter()
      _ = cv2.resize(
          frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR
      )
      inf_ms = (time.perf_counter() - t_inf) * 1000.0
      e2e_ms = (time.perf_counter() - loop_start) * 1000.0

    h2d_times.append(h2d_ms)
    latencies.append(e2e_ms)

    # NVML Telemetrie
    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
    temp_c = pynvml.nvmlDeviceGetTemperature(
        nvml_handle, pynvml.NVML_TEMPERATURE_GPU
    )
    power_draws.append(power_mw / 1000.0)
    temperatures.append(temp_c)

    # Frametime
    curr_time = time.perf_counter()
    frametimes.append((curr_time - prev_time) * 1000.0)
    prev_time = curr_time

    # Vorschaufenster (Drücke 'q' zum vorzeitigen Abbrechen)
    cv2.imshow(
        f"Benchmark - Modus: {mode_name.upper()} (Laeuft 5 Min...)",
        cv2.resize(frame_bgr, (640, 360)),
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):
      print("[INFO] Benchmark vorzeitig vom Benutzer abgebrochen.")
      break

  gc.enable()
  cap.release()
  cv2.destroyAllWindows()

  # Statistik berechnen
  ft_arr = np.array(frametimes)
  fps_arr = 1000.0 / ft_arr

  results = {
      "fps_mean": np.mean(fps_arr),
      "fps_std": np.std(fps_arr),
      "p1_low": np.percentile(fps_arr, 1),
      "p01_low": np.percentile(fps_arr, 0.1),
      "latency_mean": np.mean(latencies),
      "h2d_mean": np.mean(h2d_times) if mode_name == "gpu" else 0.0,
      "power_mean": np.mean(power_draws),
      "temp_max": np.max(temperatures),
      "fps_arr": fps_arr,
      "power_arr": power_draws,
  }
  return results


# --- HAUPTLAUF: BEIDE MODI Nacheinander ---
print("[INFO] Starte automatisierten 10-Minuten-Gesamtbenchmark (CPU -> GPU)...")

results_cpu = run_benchmark_mode("cpu")
time.sleep(3)  # Kurze Kühl- und Stabilisierungspause zwischen den Läufen
results_gpu = run_benchmark_mode("gpu")

pynvml.nvmlShutdown()

# --- GEGENÜBERSTELLUNG & AUSWERTUNG ---
print("\n" + "=" * 65)
print("📊 FINALER SYSTEM-VERGLEICH (5 Min. CPU vs. 5 Min. GPU)")
print("=" * 65)
print(
    f"{'Metrik':<25} | {'CPU-Baseline':<18} | {'GPU-Beschleunigt':<18}"
)
print("-" * 65)
print(
    f"{'Durchschnittliche FPS':<25} |"
    f" {results_cpu['fps_mean']:<18.2f} |"
    f" {results_gpu['fps_mean']:<18.2f}"
)
print(
    f"{'FPS Jitter (StdDev)':<25} |"
    f" {results_cpu['fps_std']:<18.2f} |"
    f" {results_gpu['fps_std']:<18.2f}"
)
print(
    f"{'1% Lows FPS':<25} |"
    f" {results_cpu['p1_low']:<18.2f} |"
    f" {results_gpu['p1_low']:<18.2f}"
)
print(
    f"{'Ø Latenz (ms)':<25} |"
    f" {results_cpu['latency_mean']:<18.2f} |"
    f" {results_gpu['latency_mean']:<18.2f}"
)
print(
    f"{'Ø Leistungsaufnahme (W)':<25} |"
    f" {results_cpu['power_mean']:<18.2f} |"
    f" {results_gpu['power_mean']:<18.2f}"
)
print(
    f"{'Max. GPU Temperatur (°C)':<25} |"
    f" {results_cpu['temp_max']:<18.1f} |"
    f" {results_gpu['temp_max']:<18.1f}"
)
print("=" * 65 + "\n")

# --- VISUALISIERUNG & EXPORT ---
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

axes[0].plot(
    results_cpu["fps_arr"],
    color="#d62728",
    alpha=0.6,
    lw=1,
    label="CPU FPS (5 Min)",
)
axes[0].plot(
    results_gpu["fps_arr"],
    color="#1f77b4",
    alpha=0.8,
    lw=1,
    label="GPU FPS (5 Min)",
)
axes[0].set_ylabel("Frames per Second (FPS)")
axes[0].set_title(
    "Systems-Level Telemetry Comparison: CPU Baseline vs. GPU Acceleration"
)
axes[0].legend(loc="upper right")

axes[1].plot(
    results_cpu["power_arr"],
    color="#ff7f0e",
    alpha=0.7,
    lw=1,
    label="CPU-Run GPU Power (W)",
)
axes[1].plot(
    results_gpu["power_arr"],
    color="#2ca02c",
    alpha=0.7,
    lw=1,
    label="GPU-Run GPU Power (W)",
)
axes[1].set_ylabel("Power Draw (W)")
axes[1].set_xlabel("Frame Sample Index")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("webcam_benchmark_comparison.png", dpi=300)
print(
    "[INFO] Vergleichsdiagramm erfolgreich als"
    " 'webcam_benchmark_comparison.png' gespeichert."
)