# ARGUS: Autonomous Resource Graph Unified Scheduler
### Game-Theoretic Distributed Microkernel for Edge AI


**Hardware Target:** NVIDIA GTX 1650 (4 GB VRAM) / WSL2  
**Language Stack:** Python · C11 · Rust (bare-metal)

ARGUS models the OS as a distributed game.  Processes are rational agents
competing for CPU time; their Nash Equilibrium defines a provably fair,
starvation-free allocation.  IPC is lock-free; GPU offloading is
Nash-scheduled.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Interpretation Boundary

ARGUS is strongest as a systems-research repository with:

- real scheduler simulations,
- real IPC benchmarks,
- real distributed-clock artifacts,
- and a bare-metal-oriented microkernel skeleton.

The repository contains meaningful empirical evidence, but the more ambitious
game-theoretic interpretation should be read as the motivating systems model,
not as a full formal proof package equivalent to a specialized theory paper.

## Result Snapshot

Representative committed artifacts:

- `results/scheduler/fairness_comparison.csv` shows ARGUS fairness rising from
  `0.7027` to `0.99999` over 50 ticks while the Lyapunov proxy shrinks toward 0.
- `results/latency/ipc_mpmc_benchmark.csv` and `results/latency/distributed_sync.csv`
  provide the main throughput and causal-order evidence.
- `results/scheduler/gpu_offload_results.csv` supports the GPU-sharing story.

## Why ARGUS is stronger after calibration

- It combines systems code, benchmark artifacts, tests, and a full paper.
- The fairness and convergence artifacts are concrete and easy to inspect.
- The repository is now framed as a serious systems-research platform with
  strong empirical signals, rather than as a universal theorem-backed OS
  replacement claim.

## Architecture

```
argus-os/
├── src/
│   ├── python/
│   │   ├── argus_sync_sim.py     # Nash Equilibrium scheduler (Mirror Descent)
│   │   ├── distributed_sync.py   # Lamport + Vector clocks
│   │   ├── gpu_offload.py        # Nash GPU edge-offloading scheduler
│   │   └── starvation_test.py    # ARGUS vs CFS vs Round Robin
│   ├── c/
│   │   └── argus_ipc.c           # Lock-free MPMC IPC (Vyukov algorithm)
│   └── rust/
│       └── main.rs               # Bare-metal microkernel skeleton
├── tests/
│   └── test_argus.py             # pytest suite (8 tests)
├── results/
│   ├── scheduler/
│   │   ├── nash_equilibrium.csv          # Legacy basic results
│   │   ├── nash_convergence.csv          # Mirror Descent convergence (50 ticks)
│   │   ├── lyapunov_convergence.csv      # Lyapunov V(t) curve
│   │   ├── fairness_comparison.csv       # ARGUS vs CFS vs RR
│   │   └── gpu_offload_results.csv       # GPU offloading latency/speedup
│   ├── latency/
│   │   ├── ipc_benchmark.csv             # Legacy SPSC results
│   │   ├── ipc_mpmc_benchmark.csv        # MPMC throughput (1P1C/2P2C/4P4C)
│   │   └── distributed_sync.csv          # Lamport/Vector clock events
│   └── starvation/
│       ├── starvation_comparison.csv     # 100-tick starvation comparison
│       └── starvation_comparison.png     # Plot
└── paper/
    └── argus_paper.tex                   # Full ACM paper (SOSP/EuroSys target)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib
```

### 2. Nash Equilibrium Scheduler
```bash
python3 src/python/argus_sync_sim.py
```
Runs 50-tick Mirror Descent NE simulation. Outputs:
- `results/scheduler/nash_convergence.csv`
- `results/scheduler/lyapunov_convergence.csv`
- `results/scheduler/fairness_comparison.csv`

### 3. Distributed Causal Consistency
```bash
python3 src/python/distributed_sync.py
```
Outputs: `results/latency/distributed_sync.csv`

### 4. Starvation Comparison
```bash
python3 src/python/starvation_test.py
```
Outputs: `results/starvation/starvation_comparison.{csv,png}`

### 5. GPU Edge Offloading
```bash
python3 src/python/gpu_offload.py
```
Outputs: `results/scheduler/gpu_offload_results.csv`

### 6. Lock-Free IPC (C11)
```bash
gcc -O3 -std=c11 -lpthread src/c/argus_ipc.c -o src/c/argus_ipc
./src/c/argus_ipc
```
Outputs: `results/latency/ipc_mpmc_benchmark.csv`

### 7. Run All Tests
```bash
python3 -m pytest tests/test_argus.py -v
```

## Experimental Results

### Nash Equilibrium Convergence

Mirror Descent converges from urgency-biased allocation to uniform SLA
targets (20 % each) within **17 ticks**.

| Tick | Lyapunov V | Fairness (Jain) | Min Alloc |
|-----:|----------:|----------------:|----------:|
|    1 |  0.21005  |  0.703          |  5.00 %   |
|    5 |  0.10230  |  0.842          |  5.00 %   |
|   10 |  0.04013  |  0.932          |  5.00 %   |
|   17 |  0.00993  |  0.981          |  5.00 %   |
|   50 |  0.000003 |  1.000          |  5.00 %   |

**Starvation-free guarantee:** every process receives ≥ 5 % CPU at every tick.

### Starvation Comparison (100 ticks, 5 processes)

| Scheduler        | Low-Prio Min | Low-Prio Avg | Jain Index | Starvation? |
|:----------------|-------------:|-------------:|-----------:|:-----------|
| Round Robin      |   20.00 %    |   20.00 %    |   1.0000   | Never       |
| CFS (priority)   |    1.96 %    |    1.96 %    |   0.7421   | Near (1.96 %) |
| **ARGUS-SYNC NE**|  **5.00 %**  |  **10.75 %** | **0.9292** | **Never**   |

ARGUS is the only scheduler that both **prevents starvation** and
**respects urgency differences**.  CFS starves the low-priority process
to 1.96 %.

### IPC Throughput (MPMC, WSL2 / C11)

| Configuration | Messages | Throughput   |
|:-------------|:--------:|:------------:|
| 1P / 1C      |  50,000  | **14.73 M/s** |
| 2P / 2C      | 100,000  |  6.34 M/s    |
| 4P / 4C      | 200,000  |  5.44 M/s    |

All configurations exceed the 100 k msg/s system requirement by 54–147×.  
Zero lock acquisitions; per-message Lamport timestamps for causal ordering.

### GPU Offloading Speedup (GTX 1650)

| Task           | Device   | CPU (ms) | GPU (ms) | Speedup   |
|:--------------|:--------:|---------:|---------:|:---------:|
| 64×64 matmul   | CPU      |   0.08   |   ---    |   1.0×    |
| 1024×1024      | GPU(sim) |  399     |   1.11   | **359×**  |
| 1024×1024      | GPU(sim) |  438     |   1.29   | **340×**  |
| **Mean large** |          |          |          | **345×**  |

*GPU latency here is a modeled/simulated estimate derived from the GTX 1650 FP32
peak path, while CPU latency is measured on the local WSL2 numpy stack. This
table should therefore be read as a scheduler-oriented comparative artifact, not
as a vendor-neutral production benchmark.*

Nash Equilibrium shares GPU time among concurrent large tasks — no task
is starved of GPU cycles regardless of urgency.

### Distributed Causal Consistency

- **60 events** (3 processes × 10 messages each, send + receive)
- **0 causality violations** (Lamport and Vector clock checks)
- Final Lamport clocks: P₀=44, P₁=44, P₂=43

## Test Suite (pytest)

| Test | Description | Status |
|:-----|:-----------|:------:|
| `test_nash_equilibrium_convergence` | V < 0.01 within 20 ticks | ✅ |
| `test_starvation_freedom` | min alloc ≥ 5 % always | ✅ |
| `test_lyapunov_decreasing` | V(t) non-increasing | ✅ |
| `test_lamport_clock_ordering` | recv_ts > send_ts always | ✅ |
| `test_vector_clock_consistency` | 0 causality violations | ✅ |
| `test_ipc_throughput` | > 100 k msg/s | ✅ |
| `test_gpu_offload_speedup` | GPU > 2× for 1024×1024 | ✅ |
| `test_fairness_index` | mean Jain index > 0.80 | ✅ |

All 8 tests pass in ~26 seconds.

## Game Theory Background

**Nash Equilibrium** (log utility):  
$$x_i^* = \frac{w_i}{\sum_j w_j}$$

**Mirror Descent update**:  
$$w_i(t+1) = w_i(t) \cdot \exp\!\bigl(\alpha \cdot (s_i - x_i(t))\bigr)$$

**Lyapunov function** (KL divergence):  
$$V(t) = \sum_i x_i(t)\log\frac{x_i(t)}{s_i} \xrightarrow{t\to\infty} 0$$

See `paper/argus_paper.tex` for full proofs.


