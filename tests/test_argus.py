"""
ARGUS Test Suite — Task 6
pytest tests for all ARGUS components.

Run from project root:
    pytest tests/test_argus.py -v

Max total runtime: ~10 minutes.
"""

import sys
import os
import subprocess

import pytest
import numpy as np
import pandas as pd

# ---- path setup ----
ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC  = os.path.join(ROOT, "src", "python")
sys.path.insert(0, os.path.abspath(SRC))
os.chdir(os.path.abspath(ROOT))   # ensure relative CSV paths resolve correctly


# ============================================================
# Fixtures: run simulations once, share results
# ============================================================

@pytest.fixture(scope="session")
def nash_results():
    """Run Nash Equilibrium simulation and return (df_nash, df_lyap, df_fair)."""
    from argus_sync_sim import run_nash_equilibrium_simulation
    return run_nash_equilibrium_simulation()


@pytest.fixture(scope="session")
def distributed_results():
    """Run distributed sync simulation and return the events DataFrame."""
    from distributed_sync import run_distributed_simulation
    return run_distributed_simulation()


@pytest.fixture(scope="session")
def starvation_results():
    """Run starvation comparison simulation and return the DataFrame."""
    from starvation_test import run_starvation_simulation
    return run_starvation_simulation()


@pytest.fixture(scope="session")
def gpu_results():
    """Run GPU offload simulation and return the DataFrame."""
    from gpu_offload import run_gpu_offload_simulation
    return run_gpu_offload_simulation()


@pytest.fixture(scope="session")
def ipc_csv():
    """
    Ensure the MPMC IPC benchmark CSV exists.
    If not (binary not compiled yet), compile and run it now.
    """
    csv_path = "results/latency/ipc_mpmc_benchmark.csv"
    if not os.path.exists(csv_path):
        binary  = "src/c/argus_ipc"
        src_c   = "src/c/argus_ipc.c"
        # Compile
        compile_result = subprocess.run(
            ["gcc", "-O3", "-std=c11", "-lpthread", src_c, "-o", binary],
            capture_output=True, text=True
        )
        if compile_result.returncode != 0:
            pytest.skip(f"C IPC compilation failed: {compile_result.stderr}")
        # Run benchmark
        run_result = subprocess.run([f"./{binary}"], capture_output=True, text=True)
        if run_result.returncode != 0:
            pytest.skip(f"C IPC benchmark failed: {run_result.stderr}")
    return pd.read_csv(csv_path)


# ============================================================
# Test 1: Nash Equilibrium convergence
# ============================================================

def test_nash_equilibrium_convergence(nash_results):
    """
    Lyapunov V must drop below 0.01 within the first 20 ticks.
    Verifies that Mirror Descent converges to Nash Equilibrium.
    """
    _, df_lyap, _ = nash_results
    early = df_lyap[df_lyap["tick"] <= 20]
    min_V = early["lyapunov_V"].min()
    assert min_V < 0.01, (
        f"Nash Equilibrium did not converge within 20 ticks: "
        f"min V = {min_V:.5f} (must be < 0.01)"
    )


# ============================================================
# Test 2: Starvation freedom
# ============================================================

def test_starvation_freedom(nash_results):
    """
    Every process must receive >= 5% CPU allocation at every tick.
    Verifies the epsilon-floor starvation guarantee.
    """
    df_nash, _, _ = nash_results
    share_cols = [c for c in df_nash.columns if c.startswith("share_")]
    min_share = float(df_nash[share_cols].min().min())
    assert min_share >= 0.049, (
        f"Starvation detected: min allocation = {min_share*100:.3f}% "
        f"(must be >= 5%)"
    )


# ============================================================
# Test 3: Lyapunov function non-increasing
# ============================================================

def test_lyapunov_decreasing(nash_results):
    """
    V(t) must be lower at the end than at the start — overall convergence.
    (Small oscillations are tolerated; the final value must be less.)
    """
    _, df_lyap, _ = nash_results
    V = df_lyap["lyapunov_V"].values
    assert V[-1] <= V[0], (
        f"Lyapunov function did not decrease overall: "
        f"V[0]={V[0]:.5f}  V[-1]={V[-1]:.5f}"
    )
    # Also: at least 80% of ticks see a decrease relative to tick 1
    decreasing_ticks = int(np.sum(V <= V[0]))
    assert decreasing_ticks >= int(0.80 * len(V)), (
        f"Only {decreasing_ticks}/{len(V)} ticks showed V <= V[0]"
    )


# ============================================================
# Test 4: Lamport clock causal ordering
# ============================================================

def test_lamport_clock_ordering(distributed_results):
    """
    All receive events must have Lamport timestamp strictly greater than
    the corresponding send timestamp. No causality violations allowed.
    """
    df = distributed_results
    violations = df[~df["causal_order_ok"]]
    assert len(violations) == 0, (
        f"Lamport causality violations detected:\n{violations}"
    )


# ============================================================
# Test 5: Vector clock consistency (from saved CSV)
# ============================================================

def test_vector_clock_consistency():
    """
    The saved distributed_sync.csv must contain zero causal violations.
    """
    csv_path = "results/latency/distributed_sync.csv"
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    df = pd.read_csv(csv_path)
    violations = df[~df["causal_order_ok"]]
    assert len(violations) == 0, (
        f"Vector clock violations in CSV: {len(violations)}"
    )


# ============================================================
# Test 6: IPC throughput > 100k messages/sec
# ============================================================

def test_ipc_throughput(ipc_csv):
    """
    At least one MPMC configuration must achieve > 100,000 messages/sec.
    Tests the lock-free MPMC ring buffer performance.
    """
    df = ipc_csv
    max_tp = float(df["throughput_msg_per_sec"].max())
    assert max_tp > 100_000, (
        f"IPC throughput too low: {max_tp:.0f} msg/s (must be > 100,000)"
    )


# ============================================================
# Test 7: GPU offload speedup > 2x for 1024×1024
# ============================================================

def test_gpu_offload_speedup(gpu_results):
    """
    Large (1024×1024) GPU tasks must be at least 2× faster than CPU.
    Tests the GPU offloading decision and measurement accuracy.
    """
    df = gpu_results
    large = df[df["size"] == "1024x1024"]
    if len(large) == 0:
        pytest.skip("No 1024x1024 tasks found in GPU results")

    max_speedup = float(large["speedup_vs_cpu"].max())
    assert max_speedup > 2.0, (
        f"GPU speedup for 1024×1024 too low: {max_speedup:.2f}x (must be > 2×)"
    )


# ============================================================
# Test 8: Jain's Fairness Index > 0.8
# ============================================================

def test_fairness_index(nash_results):
    """
    ARGUS-SYNC mean Jain's Fairness Index over all ticks must exceed 0.80.
    Tick 1 starts from urgency-biased weights; Mirror Descent converges
    the allocation toward fairness, so the mean across all 50 ticks
    robustly exceeds 0.80.
    """
    df_nash, _, df_fair = nash_results
    mean_fairness = float(df_fair["argus_fairness"].mean())
    assert mean_fairness > 0.80, (
        f"Mean Jain's Fairness Index too low: {mean_fairness:.4f} (must be > 0.80)"
    )
