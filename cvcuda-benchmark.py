import gc
import time
import cv2
import numpy as np
import pynvml
import torch

# --- KONFIGURATION FÜR INDUSTRIE-BENCHMARK ---
DURATION_SECONDS = 300  # 5 Minuten pro Modus (300 Sekunden)
PAUSE_SECONDS = 120  # 2 Minuten Pause zwischen den Läufen
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720  # Auflösung (stabil für USB-2.0/MJPEG)
TARGET_FPS = 30

print(
    "[INFO] Initialisiere NVML für Energie- und Thermik-Telemetrie auf RTX"
    " 3050..."
)
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_industrial_benchmark(mode_name):
  print(
      "\n"
      + "=" * 65
      + f"\n[START] Starte 5-Minuten-Industrie-Benchmark: [{mode_name.upper()}]\n"
      + "=" * 65
  )

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
  cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

  if not cap.isOpened():
    print("[ERROR] Konnte Kamera nicht öffnen!")
    return None

  preprocessing_times = []
  h2d_transfer_times = []
  inference_times = []
  e2e_latencies = []
  power_draws = []
  temperatures = []
  frametimes = []

  gc.disable()  # Verhindere Garbage-Collection-Stalls

  start_evt = torch.cuda.Event(enable_timing=True)
  prep_end_evt = torch.cuda.Event(enable_timing=True)
  h2d_end_evt = torch.cuda.Event(enable_timing=True)
  inf_end_evt = torch.cuda.Event(enable_timing=True)

  test_start_time = time.perf_counter()
  prev_time = test_start_time

  while (time.perf_counter() - test_start_time) < DURATION_SECONDS:
    # Restzeit-Berechnung
    elapsed_time = time.perf_counter() - test_start_time
    remaining_time = max(0, int(DURATION_SECONDS - elapsed_time))
    mins, secs = divmod(remaining_time, 60)

    # Terminal-Ausgabe (gedrosselt auf volle Sekunden)
    if remaining_time != getattr(run_industrial_benchmark, "_last_printed", -1):
      run_industrial_benchmark._last_printed = remaining_time
      print(
          f"[{mode_name.upper()}] Verbleibende Zeit: {mins:02d}:{secs:02d} Min",
          end="\r",
      )

    ret, frame_bgr = cap.read()
    if not ret:
      print("\n[WARN] Kein Frame von der Kamera erhalten.")
      break

    loop_start = time.perf_counter()

    if mode_name == "cpu_baseline":
      # --- PFAD A: KLASSISCHES CPU-PREPROCESSING (OpenCV) ---
      start_evt.record() if torch.cuda.is_available() else None
      t_prep_cpu = time.perf_counter()

      resized_cpu = cv2.resize(
          frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR
      )
      rgb_cpu = cv2.cvtColor(resized_cpu, cv2.COLOR_BGR2RGB)
      prep_ms = (time.perf_counter() - t_prep_cpu) * 1000.0

      t_h2d_cpu = time.perf_counter()
      tensor_cpu = (
          torch.from_numpy(rgb_cpu)
          .permute(2, 0, 1)
          .float()
          .div(255.0)
          .unsqueeze(0)
      )
      tensor_gpu = tensor_cpu.to(device, non_blocking=False)
      h2d_ms = (time.perf_counter() - t_h2d_cpu) * 1000.0

      t_inf_cpu = time.perf_counter()
      with torch.inference_mode():
        _ = tensor_gpu * 1.05
      inf_ms = (time.perf_counter() - t_inf_cpu) * 1000.0

      e2e_ms = (time.perf_counter() - loop_start) * 1000.0
    else:
      # --- PFAD B: GPU-BESCHLEUNIGTES PREPROCESSING (Zero-Copy Prinzip) ---
      start_evt.record()

      tensor_raw = torch.from_numpy(frame_bgr)
      tensor_gpu_raw = tensor_raw.to(device, non_blocking=True)
      prep_end_evt.record()

      h2d_end_evt.record()

      with torch.inference_mode():
        processed = (
            torch.nn.functional.interpolate(
                tensor_gpu_raw.permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                size=(FRAME_HEIGHT, FRAME_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            * 1.05
        )
      inf_end_evt.record()
      torch.cuda.synchronize()

      prep_ms = start_evt.elapsed_time(prep_end_evt)
      h2d_ms = prep_end_evt.elapsed_time(h2d_end_evt)
      inf_ms = h2d_end_evt.elapsed_time(inf_end_evt)
      e2e_ms = start_evt.elapsed_time(inf_end_evt)

    preprocessing_times.append(prep_ms)
    h2d_transfer_times.append(h2d_ms)
    inference_times.append(inf_ms)
    e2e_latencies.append(e2e_ms)

    # NVML Telemetrie
    power_mw = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
    temp_c = pynvml.nvmlDeviceGetTemperature(
        nvml_handle, pynvml.NVML_TEMPERATURE_GPU
    )
    power_draws.append(power_mw / 1000.0)
    temperatures.append(temp_c)

    # Frametime & Jitter
    curr_time = time.perf_counter()
    frametimes.append((curr_time - prev_time) * 1000.0)
    prev_time = curr_time

    # Visuelle Timer-Anzeige im OpenCV-Fenster
    display_frame = cv2.resize(frame_bgr, (640, 360))
    timer_text = (
        f"Mode: {mode_name.upper()} | Remaining: {mins:02d}:{secs:02d}"
    )
    cv2.putText(
        display_frame,
        timer_text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Industrial Benchmark", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      print("\n[INFO] Benchmark vom Benutzer abgebrochen.")
      break

  gc.enable()
  cap.release()
  cv2.destroyAllWindows()

  ft_arr = np.array(frametimes)
  fps_arr = 1000.0 / ft_arr

  results = {
      "fps_mean": np.mean(fps_arr),
      "fps_std": np.std(fps_arr),
      "p1_low": np.percentile(fps_arr, 1),
      "prep_mean": np.mean(preprocessing_times),
      "h2d_mean": np.mean(h2d_transfer_times),
      "inf_mean": np.mean(inference_times),
      "e2e_mean": np.mean(e2e_latencies),
      "power_mean": np.mean(power_draws),
      "temp_max": np.max(temperatures),
  }
  return results


# --- HAUPTLAUF: AUTOMATISIERTE SEQUENZ ---
print(
    "[INFO] Starte vollautomatisierten Industrie-Vergleich (CPU -> Pause ->"
    " GPU)..."
)

# 1. CPU-Baseline Durchlauf
results_cpu = run_industrial_benchmark("cpu_baseline")

if results_cpu:
  print(
      "\n[INFO] 5-Minuten CPU-Test abgeschlossen. Starte 2-minütige"
      " Abkühl- und Stabilisierungspause..."
  )
  for remaining in range(PAUSE_SECONDS, 0, -1):
    mins, secs = divmod(remaining, 60)
    print(
        f"Abkühlphase: {mins:02d}:{secs:02d} Minuten verbleibend...", end="\r"
    )
    time.sleep(1)
  print("\n[INFO] Pause beendet. Starte GPU-beschleunigten Durchlauf...\n")

  # 2. GPU-Beschleunigter Durchlauf
  results_gpu = run_industrial_benchmark("gpu_accelerated")

  pynvml.nvmlShutdown()

  # --- GEGENÜBERSTELLUNG & AUSWERTUNG ---
  if results_gpu:
    print("\n" + "=" * 68)
    print("📊 FINALER INDUSTRIE-VERGLEICH (5 Min. CPU vs. 5 Min. GPU)")
    print("=" * 68)
    print(
        f"{'Metrik':<28} | {'CPU-Baseline':<18} | {'GPU-Beschleunigt':<18}"
    )
    print("-" * 68)
    print(
        f"{'Durchschnittliche FPS':<28} |"
        f" {results_cpu['fps_mean']:<18.2f} |"
        f" {results_gpu['fps_mean']:<18.2f}"
    )
    print(
        f"{'FPS Jitter (StdDev)':<28} |"
        f" {results_cpu['fps_std']:<18.2f} |"
        f" {results_gpu['fps_std']:<18.2f}"
    )
    print(
        f"{'1% Lows FPS':<28} |"
        f" {results_cpu['p1_low']:<18.2f} |"
        f" {results_gpu['p1_low']:<18.2f}"
    )
    print(
        f"{'Ø Preprocessing (ms)':<28} |"
        f" {results_cpu['prep_mean']:<18.2f} |"
        f" {results_gpu['prep_mean']:<18.2f}"
    )
    print(
        f"{'Ø PCIe H2D Transfer (ms)':<28} |"
        f" {results_cpu['h2d_mean']:<18.2f} |"
        f" {results_gpu['h2d_mean']:<18.2f}"
    )
    print(
        f"{'Ø End-to-End Latenz (ms)':<28} |"
        f" {results_cpu['e2e_mean']:<18.2f} |"
        f" {results_gpu['e2e_mean']:<18.2f}"
    )
    print(
        f"{'Ø Leistungsaufnahme (W)':<28} |"
        f" {results_cpu['power_mean']:<18.2f} |"
        f" {results_gpu['power_mean']:<18.2f}"
    )
    print(
        f"{'Max. GPU Temperatur (°C)':<28} |"
        f" {results_cpu['temp_max']:<18.1f} |"
        f" {results_gpu['temp_max']:<18.1f}"
    )
    print("=" * 68 + "\n")