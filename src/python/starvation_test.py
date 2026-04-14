"""
ARGUS Starvation Comparison
Task 5: Compare starvation behavior across 3 schedulers over 100 ticks.

Scenario:
  - 1 high-priority process  (urgency=100)
  - 3 medium-priority        (urgency=50 each)
  - 1 low-priority           (urgency=5)

Schedulers compared:
  a) Round Robin:        equal shares (1/N each tick)
  b) Priority (CFS-like): proportional to urgency (classic starvation)
  c) ARGUS-SYNC NE:     Nash Equilibrium with 5% starvation floor

Metrics:
  - Minimum allocation received by low-priority process
  - Time to starvation (when allocation < 1%)
  - Jain's Fairness Index

Saves:
  results/starvation/starvation_comparison.csv
  results/starvation/starvation_comparison.png
"""

import os
import numpy as np
import pandas as pd

os.makedirs("results/starvation", exist_ok=True)

# Optional matplotlib for plot
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ============================================================
# Helper utilities
# ============================================================

def jains_fairness_index(allocations: np.ndarray) -> float:
    x = np.array(allocations, dtype=float)
    n = len(x)
    denom = n * np.sum(x ** 2)
    return float(np.sum(x) ** 2 / denom) if denom > 0 else 0.0


def nash_equilibrium_with_floor(weights: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """
    Nash NE allocation with starvation-free floor.
    Iterative algorithm: pin floored elements to epsilon and redistribute
    remaining budget among unconstrained elements until stable.
    Guarantees x_i >= epsilon for all i.
    """
    x = weights / weights.sum()
    for _ in range(len(weights) + 1):
        below = x < epsilon
        if not below.any():
            break
        x[below] = epsilon
        remaining = 1.0 - epsilon * below.sum()
        above = ~below
        if above.any() and x[above].sum() > 0:
            x[above] = x[above] / x[above].sum() * remaining
    return x


# ============================================================
# Schedulers
# ============================================================

def scheduler_round_robin(n: int) -> np.ndarray:
    """Equal share to every process."""
    return np.full(n, 1.0 / n)


def scheduler_cfs(urgencies: np.ndarray) -> np.ndarray:
    """Proportional allocation (Linux CFS-like, no starvation protection)."""
    return urgencies / urgencies.sum()


def scheduler_argus(weights: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """ARGUS-SYNC Nash Equilibrium with Mirror Descent + starvation floor."""
    return nash_equilibrium_with_floor(weights, epsilon)


# ============================================================
# Simulation
# ============================================================

def run_starvation_simulation() -> pd.DataFrame:
    """
    Simulate 100 CPU ticks with 5 processes under 3 schedulers.
    Uses Mirror Descent weight update for ARGUS to show dynamic behavior.
    """
    print("=" * 60)
    print("  ARGUS: Starvation Comparison Simulation (100 ticks)")
    print("=" * 60)

    np.random.seed(0)

    TICKS    = 100
    EPSILON  = 0.05
    ALPHA    = 0.4    # Mirror Descent step size for ARGUS

    process_names = ["High", "Med_A", "Med_B", "Med_C", "Low"]
    urgencies     = np.array([100.0, 50.0, 50.0, 50.0, 5.0])
    N             = len(urgencies)

    # ARGUS weights start at urgency values, evolve via Mirror Descent
    argus_weights = urgencies.copy()
    # ARGUS SLA targets: deliberately equalizing (to protect Low-priority)
    sla_targets = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

    records = []

    for tick in range(1, TICKS + 1):
        # --- Round Robin ---
        rr_x = scheduler_round_robin(N)

        # --- CFS (proportional) ---
        cfs_x = scheduler_cfs(urgencies)

        # --- ARGUS NE ---
        argus_x = scheduler_argus(argus_weights, epsilon=EPSILON)

        # Mirror Descent update for next tick
        gradient = sla_targets - argus_x
        argus_weights = argus_weights * np.exp(ALPHA * gradient)
        argus_weights = np.maximum(argus_weights, 1e-9)

        # Record per-tick stats
        rec = {
            "tick":                  tick,
            # Full allocations (each process)
            "rr_high":   rr_x[0],  "rr_med_a":  rr_x[1],
            "rr_med_b":  rr_x[2],  "rr_med_c":  rr_x[3],
            "rr_low":    rr_x[4],
            "cfs_high":  cfs_x[0], "cfs_med_a": cfs_x[1],
            "cfs_med_b": cfs_x[2], "cfs_med_c": cfs_x[3],
            "cfs_low":   cfs_x[4],
            "ne_high":   argus_x[0], "ne_med_a": argus_x[1],
            "ne_med_b":  argus_x[2], "ne_med_c": argus_x[3],
            "ne_low":    argus_x[4],
            # Fairness
            "rr_fairness":    jains_fairness_index(rr_x),
            "cfs_fairness":   jains_fairness_index(cfs_x),
            "argus_fairness": jains_fairness_index(argus_x),
            # Low-priority allocation
            "rr_low_alloc":    float(rr_x[4]),
            "cfs_low_alloc":   float(cfs_x[4]),
            "argus_low_alloc": float(argus_x[4]),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv("results/starvation/starvation_comparison.csv", index=False)
    print(f"Saved: results/starvation/starvation_comparison.csv")

    # ---- Summary stats ----
    def starvation_time(col, threshold=0.01):
        """First tick where allocation drops below threshold."""
        below = df[df[col] < threshold]
        return int(below["tick"].min()) if len(below) > 0 else None

    print("\n--- Low-priority process statistics ---")
    for sched, col in [("Round Robin", "rr_low_alloc"),
                        ("CFS (priority)", "cfs_low_alloc"),
                        ("ARGUS NE",       "argus_low_alloc")]:
        min_a = df[col].min()
        avg_a = df[col].mean()
        t_s   = starvation_time(col)
        t_s_str = str(t_s) if t_s else "never"
        print(f"  {sched:18s}: min={min_a*100:5.2f}%  avg={avg_a*100:5.2f}%  "
              f"starvation_at_tick={t_s_str}")

    print("\n--- Jain's Fairness Index (mean over 100 ticks) ---")
    for sched, col in [("Round Robin",   "rr_fairness"),
                        ("CFS (priority)", "cfs_fairness"),
                        ("ARGUS NE",       "argus_fairness")]:
        print(f"  {sched:18s}: {df[col].mean():.4f}")

    # ---- Plot ----
    if HAS_PLOT:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        ticks = df["tick"].values

        # Top: Low-priority allocation over time
        ax = axes[0]
        ax.plot(ticks, df["rr_low_alloc"] * 100,
                label="Round Robin",   color="blue",  linestyle="--")
        ax.plot(ticks, df["cfs_low_alloc"] * 100,
                label="CFS (priority)", color="red",   linestyle="-.")
        ax.plot(ticks, df["argus_low_alloc"] * 100,
                label="ARGUS NE",       color="green", linestyle="-",  linewidth=2)
        ax.axhline(y=1.0,  color="black", linestyle=":",  linewidth=0.8,
                   label="Starvation threshold (1%)")
        ax.axhline(y=5.0,  color="gray",  linestyle=":",  linewidth=0.8,
                   label="ARGUS floor (5%)")
        ax.set_title("Low-Priority Process CPU Allocation over 100 Ticks",
                     fontsize=13)
        ax.set_xlabel("CPU Tick")
        ax.set_ylabel("CPU Allocation (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # Bottom: Jain's Fairness Index
        ax = axes[1]
        ax.plot(ticks, df["rr_fairness"],
                label="Round Robin",    color="blue",  linestyle="--")
        ax.plot(ticks, df["cfs_fairness"],
                label="CFS (priority)", color="red",   linestyle="-.")
        ax.plot(ticks, df["argus_fairness"],
                label="ARGUS NE",       color="green", linestyle="-",  linewidth=2)
        ax.set_title("Jain's Fairness Index over 100 Ticks", fontsize=13)
        ax.set_xlabel("CPU Tick")
        ax.set_ylabel("Fairness Index (1.0 = perfect)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig("results/starvation/starvation_comparison.png", dpi=150)
        plt.close()
        print("Saved: results/starvation/starvation_comparison.png")
    else:
        print("[INFO] matplotlib not available — skipping plot generation.")

    return df


if __name__ == "__main__":
    run_starvation_simulation()
