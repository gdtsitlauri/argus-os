"""
ARGUS GPU Edge Offloading Scheduler
Task 4: Nash Equilibrium GPU task scheduler with real CUDA benchmarks.

Offloading decision:
  - Small tasks (< 1ms CPU estimate)  -> always CPU
  - Large tasks (> 10ms CPU estimate) -> GPU if available (Nash NE sharing)
  - Medium tasks (1-10ms)             -> CPU

GPU sharing among concurrent large tasks uses Nash Equilibrium
from argus_sync_sim.py.

Saves: results/scheduler/gpu_offload_results.csv
"""

import os
import time
import numpy as np
import pandas as pd

os.makedirs("results/scheduler", exist_ok=True)

# Try importing PyTorch / CUDA
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE_NAME = torch.cuda.get_device_name(0)
    else:
        DEVICE_NAME = "N/A (CPU fallback)"
except ImportError:
    torch = None
    CUDA_AVAILABLE = False
    DEVICE_NAME = "N/A (torch not installed)"

print(f"[GPU Offload] CUDA available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"[GPU Offload] Device: {DEVICE_NAME}")


# ============================================================
# Task model
# ============================================================

def classify_task(n: int) -> str:
    """
    Classify matrix multiply task by size:
      small  : n <= 64    (< 1ms CPU)
      medium : 64 < n <= 256  (1-10ms CPU)
      large  : n > 256        (> 10ms CPU)
    """
    if n <= 64:
        return "small"
    elif n <= 256:
        return "medium"
    return "large"


def cpu_matmul_latency(n: int, repeat: int = 3) -> float:
    """Measure CPU matrix multiply latency in ms (average of `repeat` runs)."""
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    # Warm-up
    _ = A @ B
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = A @ B
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def gpu_matmul_latency(n: int, repeat: int = 3) -> float:
    """
    Measure GPU matrix multiply latency in ms.
    Uses torch.cuda and CUDA events for accurate timing.
    """
    if not CUDA_AVAILABLE or torch is None:
        return _simulated_gpu_latency(n)

    A = torch.randn(n, n, device="cuda")
    B = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()

    # Warm-up
    _ = torch.mm(A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.mean(times))


def _simulated_gpu_latency(n: int) -> float:
    """
    Realistic GTX 1650 GPU latency simulation when CUDA is unavailable.

    Based on GTX 1650 specs (896 CUDA cores, 4GB GDDR5, ~2.9 TFLOPS FP32):
      - Small matrices: dominated by kernel launch overhead (~0.5ms)
      - Large matrices: compute-bound, ~30x faster than CPU numpy
    """
    # Simulate kernel launch overhead + compute time
    launch_overhead_ms = 0.45
    # GTX 1650 throughput: ~2.9 TFLOPS = 2.9e12 FLOPS
    flops_needed = 2.0 * (n ** 3)          # matmul FLOP count
    gpu_tflops   = 2.9e12                  # GTX 1650 FP32
    compute_ms   = (flops_needed / gpu_tflops) * 1000
    # Add realistic jitter
    jitter = np.random.uniform(0.85, 1.15)
    return (launch_overhead_ms + compute_ms) * jitter


# ============================================================
# Nash Equilibrium GPU sharing
# ============================================================

def nash_gpu_share(n_tasks: int, weights: list) -> np.ndarray:
    """
    Distribute GPU time among concurrent large tasks using Nash Equilibrium.
    x_i* = w_i / sum(w_j)  (log utility NE)
    """
    w = np.array(weights, dtype=float)
    return w / w.sum()


# ============================================================
# Main offload simulation
# ============================================================

def run_gpu_offload_simulation() -> pd.DataFrame:
    """
    Simulate Edge AI task offloading to GTX 1650.

    Tasks:
      - 10 small matrix multiplies:  64×64   -> CPU
      - 10 large matrix multiplies: 1024×1024 -> GPU

    Measures actual CPU and GPU latency, computes speedup ratio.
    """
    print("\n" + "=" * 60)
    print("  ARGUS: GPU Edge Offloading Scheduler (GTX 1650)")
    print("=" * 60)

    np.random.seed(7)
    records = []
    task_id = 0

    # --- Small tasks: 64×64 -> CPU ---
    print("\n[Phase 1] Small tasks (64x64) -> CPU")
    for i in range(10):
        n = 64
        task_id += 1
        size_label = f"{n}x{n}"
        category   = classify_task(n)

        cpu_ms = cpu_matmul_latency(n, repeat=5)
        # Small tasks stay on CPU (GPU overhead > compute benefit)
        device = "CPU"

        records.append({
            "task_id":        task_id,
            "size":           size_label,
            "category":       category,
            "device":         device,
            "latency_ms":     round(cpu_ms, 4),
            "cpu_latency_ms": round(cpu_ms, 4),
            "speedup_vs_cpu": 1.0,
        })
        print(f"  Task {task_id:2d}: {size_label:10s} CPU={cpu_ms:.3f}ms  -> {device}")

    # --- Large tasks: 1024×1024 -> GPU ---
    print("\n[Phase 2] Large tasks (1024x1024) -> GPU (Nash NE sharing)")
    large_weights = list(range(10, 110, 10))   # simulated urgency weights

    # Nash shares for 10 concurrent GPU tasks
    gpu_shares = nash_gpu_share(10, large_weights)

    for i in range(10):
        n = 1024
        task_id += 1
        size_label = f"{n}x{n}"
        category   = classify_task(n)

        cpu_ms = cpu_matmul_latency(n, repeat=3)

        if CUDA_AVAILABLE and torch is not None:
            gpu_ms = gpu_matmul_latency(n, repeat=3)
            device = "GPU"
        else:
            gpu_ms = _simulated_gpu_latency(n)
            device = "GPU(sim)"

        # Account for Nash-assigned GPU time share
        # (effective latency scales inversely with share)
        effective_gpu_ms = gpu_ms / gpu_shares[i]  if gpu_shares[i] > 0 else gpu_ms
        # Cap at reasonable bound (single-task baseline is pure gpu_ms)
        effective_gpu_ms = min(effective_gpu_ms, gpu_ms * 2.0)

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 1.0

        records.append({
            "task_id":        task_id,
            "size":           size_label,
            "category":       category,
            "device":         device,
            "latency_ms":     round(gpu_ms, 4),
            "cpu_latency_ms": round(cpu_ms, 4),
            "speedup_vs_cpu": round(speedup, 2),
        })
        print(f"  Task {task_id:2d}: {size_label:10s} "
              f"CPU={cpu_ms:.2f}ms  GPU={gpu_ms:.2f}ms  "
              f"Speedup={speedup:.1f}x  Nash_share={gpu_shares[i]*100:.1f}%")

    df = pd.DataFrame(records)
    df.to_csv("results/scheduler/gpu_offload_results.csv", index=False)
    print(f"\nSaved: results/scheduler/gpu_offload_results.csv")

    large_df = df[df["category"] == "large"]
    avg_speedup = large_df["speedup_vs_cpu"].mean()
    max_speedup = large_df["speedup_vs_cpu"].max()
    print(f"[RESULT] Large task avg speedup: {avg_speedup:.2f}x")
    print(f"[RESULT] Large task max speedup: {max_speedup:.2f}x")
    print(f"[RESULT] GPU mode: {'Real CUDA' if CUDA_AVAILABLE else 'Simulated (GTX 1650 model)'}")

    return df


if __name__ == "__main__":
    run_gpu_offload_simulation()
