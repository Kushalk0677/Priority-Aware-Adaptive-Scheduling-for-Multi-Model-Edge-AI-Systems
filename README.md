# PAES — Priority-Aware Edge Scheduler

[![IEEE ESL](https://img.shields.io/badge/IEEE_ESL-Accepted-success.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Accepted for publication in IEEE Embedded Systems Letters (IEEE ESL)**  
> **Authors:** Kushal Khemani · Rushil Maniar · Ajinkyya Lokhande  
> **Repository:** [https://github.com/Kushalk0677/paes](https://github.com/Kushalk0677/paes)

---

## Video Abstract

[![PAES Video Abstract — IEEE Embedded Systems Letters](https://img.youtube.com/vi/BGSM55EgBn8/maxresdefault.jpg)](https://www.youtube.com/watch?v=BGSM55EgBn8)

> **Watch on YouTube:** [https://youtu.be/BGSM55EgBn8](https://youtu.be/BGSM55EgBn8)  
> A 2.9-minute walkthrough of PAES, highlighting why queue wait time dominates multi-model edge AI serving (>98%), the composite min-heap scoring heuristic ($\alpha P_i + \beta/L_i + \gamma/E_i$), and experimental results across 6 hardware platforms (including real CUDA GPU inference on the NVIDIA Jetson Orin Nano).  
> *(Local copy archived in repository: [`media/gavideo.mp4`](media/gavideo.mp4))*

---

## What PAES Does

Modern edge deployments run multiple AI models concurrently on a single CPU or single-executor edge processor. In this setting, **queue wait time — not inference — dominates total response time (>98%)**. Yet mainstream serving frameworks (TensorFlow Serving, TorchServe) default to FIFO.

PAES assigns each pending task a composite score:

```
Score(tᵢ) = α·Pᵢ + β·(1/Lᵢ) + γ·(1/Eᵢ)
```

where `P` = priority, `L` = expected latency (ms), `E` = expected energy (mJ), and `α/β/γ` are tunable weights (default: 1/1/1). Tasks are inserted into an O(log n) min-heap; the highest-scoring task executes next. No offline profiling required.

**Key results across 6 physical platforms, 60 total runs:**

| Metric | PAES | vs. FIFO | vs. QoS |
|--------|------|----------|---------|
| Queue wait — robot pipeline (i7) | 6,728 ± 183 ms | **−33.7%** | **−40.8%** |
| Queue wait — robot pipeline (Jetson) | 6,587 ± 185 ms | **−33.7%** | **−41.5%** |
| Queue wait — synthetic (600T, i7) | 45,400 ± 1,198 ms | **−32.6%** | comparable |
| Queue wait — synthetic (600T, Jetson) | 13,165 ± 370 ms | **−28.7%** | comparable |
| Deadline miss rate @ high load | **17.1%** | best of 7 | best of 7 |
| Scheduling overhead | **0.79 ± 0.03 µs** (x86) / **1.72 µs** (ARM) | 0.001% of p50 inference | faster than QoS |

---

## Repository Structure

```
paes/
├── scheduler.py              ← All 8 scheduler implementations + Task dataclass
├── experiments.py            ← Experiments 1–6 (synthetic workload) + Wilcoxon tests
├── exp_workload_realism.py   ← Robot pipeline experiment (685 tasks, 30s)
├── exp_overhead.py           ← Scheduling decision overhead measurement
├── run_all.py                ← Master runner (uses experiments.py + figures.py)
├── run_real_device.py        ← Self-contained physical device runner (7 experiments, statistical tests)
├── figures.py                ← Publication figure generation
├── requirements.txt
├── media/
│   └── gavideo.mp4           ← 1080p IEEE Video Abstract presentation (2.91 mins)
├── models/
│   └── model_zoo.py          ← Real model wrappers + calibrated fallbacks
├── results/
│   ├── jetson-orin-nano/     ← NVIDIA Jetson Orin Nano (6C ARM, CUDA 12.6, real CUDA inference)
│   ├── i7-1165G7/            ← Intel Core i7-1165G7 (4C, 16GB) — primary device
│   ├── core-ultra-5/         ← Intel Core Ultra 5 125H (14C, 32GB)
│   └── raspberry-pi-5/       ← Raspberry Pi 5 (Cortex-A76, 8GB)
├── figures/                  ← Pre-generated paper figures
└── version_history/          ← Archived repository snapshots
```

> Results for Core Ultra 9 275H and i5-520M are not in this public repository. These devices represent the hardware boundary conditions described in Section IV-F: the Core Ultra 9 runs too fast for scheduling order to matter; the i5-520M is compute-saturated regardless of policy.

---

## Hardware Tested

| Device | CPU / Architecture | Cores / Specs | RAM | Role in paper |
|--------|-------------------|---------------|-----|---------------|
| NVIDIA Jetson Orin Nano | 6-core ARM Cortex-A78AE | 6C @ 1.5 GHz, CUDA 12.6 | 8 GB | Genuine embedded edge GPU platform — real CUDA inference |
| Intel Core i7-1165G7 | Tiger Lake | 4C/8T, 2.8–4.7 GHz | 16 GB | Primary / productive operating range |
| Intel Core Ultra 5 125H | Meteor Lake | 14C, up to 4.5 GHz | 32 GB | Strongest PAES gains (−28.1% queue wait vs. FIFO) |
| Raspberry Pi 5 | Cortex-A76 | 4C, 2.4 GHz | 8 GB | Genuine embedded edge platform — ARM / constrained |
| Intel Core Ultra 9 275H | Arrow Lake | 24C, up to 5.4 GHz | 32 GB | High-end boundary: all schedulers converge at 0% miss rate |
| Intel Core i5-520M (2010) | Westmere | 2C/4T, 2.4 GHz | 4 GB | Legacy boundary: compute-saturated, >30% miss rate across all policies |

**Hardware boundary condition:** PAES adds measurable value only where queue depth regularly exceeds ~5 tasks. The i7-1165G7, Core Ultra 5, and Jetson Orin Nano represent the productive operating range for embedded and robotic edge deployments.

---

## Reproducing the Published Results

### Requirements

```bash
pip install numpy pandas matplotlib tqdm scipy

# Optional — enables real model inference instead of calibrated simulation:
pip install torch torchvision     # MobileNetV2, YOLOv5n, MiDaS
pip install ultralytics           # YOLOv5n
pip install openai-whisper        # Whisper Tiny
pip install transformers          # DistilBERT
```

Python 3.10+ required. Tested on Python 3.12.

### Reproduce Table II — synthetic workload queue wait and latency

```bash
python run_all.py --exp 1 2
```

Outputs: `results/exp1_latency.csv`, `results/exp2_deadline.csv`

The values in paper Table II are means and std.devs. computed across all 5 physical devices (n=10 runs per device, 50 total). Per-device single-run CSVs are provided in `results/i7-1165G7/`, `results/core-ultra-5/`, and `results/raspberry-pi-5/`.

### Reproduce Table III — sensitivity analysis (α/β/γ sweep)

```bash
python run_all.py --exp 5
```

Output: `results/exp5_sensitivity.csv`

### Reproduce Figure 2 — deadline miss rate vs. load level

```bash
python run_all.py --exp 2
```

### Reproduce robot pipeline results (Figure 8, Section IV-D)

```bash
python exp_workload_realism.py
```

Output: `results/exp_workload_realism.csv`

**Note on robot pipeline CSV format:** The columns `avg_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `avg_energy_mj`, and `throughput_tps` are identical across all 7 schedulers. This is correct and expected. In this experiment, task execution profiles (inference time, energy) are fixed at submission — scheduling order only determines when each task begins, not how long it runs. **Queue wait (`avg_wait_ms`) is the sole differentiator**, and PAES's 33.7% reduction (6,728 ms vs. FIFO's 10,155 ms) is the primary reported result.

### Reproduce Figure 7 — scheduling overhead

```bash
python exp_overhead.py
```

Output: `results/exp_overhead.csv`  
PAES: **0.79 ± 0.03 µs mean**, **1.40 µs P99**, **0.001% of p50 inference time**.

### Physical Device Runner (7 experiments, statistical tests)

```bash
python run_real_device.py                 # Full run — 7 experiments, 10 repeats, 600 tasks
python run_real_device.py --quick         # Reduced counts for fast validation
python run_real_device.py --exp 1 6       # Specific experiments (1-6) + robot pipeline (7)
python run_real_device.py --repeats 20    # More repeats for publication
python run_real_device.py --device my_pc  # Custom device label for output folder
```

Output: `results/<device_name>/` with CSVs, per-model wait breakdown, and `summary.json`.

### Full run (all experiments + figures)

```bash
python run_all.py           # ~15–20 min with real models installed
python run_all.py --quick   # ~2–3 min, reduced task counts, for validation
python run_all.py --exp 1 3 # run specific experiments only
python run_all.py --device i7-1165G7  # Save results to device-specific subfolder
```

---

## Real vs. Simulated Inference

The framework supports two execution modes, selected automatically per model:

**Real inference** — runs actual PyTorch / Transformers models when libraries are installed.

**Calibrated simulation** — Gaussian fallback using latency distributions calibrated to measured hardware means, validated within ±8% of physical measurements. Used when a library is not installed, or on memory-constrained devices where loading all 5 models simultaneously caused out-of-memory conditions.

**22% of tasks across the full 5-device experiment used calibrated simulation**, as disclosed in Section IV-A of the paper. This fallback was applied identically across all 7 schedulers — it does not affect relative scheduler comparisons. Per-device fallback-free confirmation on Core Ultra 5 and Raspberry Pi 5 shows relative ordering is unchanged.

---

## AI Models

| Model | Task | Priority | Deadline | Approx. Latency |
|-------|------|----------|----------|-----------------|
| YOLOv5n | Object detection | 3.0 (high) | 300 ms | ~80 ms |
| MobileNetV2 | Image classification | 2.0 | 200 ms | ~35 ms |
| Whisper Tiny | Speech recognition | 2.0 | 500 ms | ~150 ms |
| DistilBERT | NLP / sentiment | 1.5 | 400 ms | ~55 ms |
| MiDaS Small | Depth estimation | 1.0 (low) | 600 ms | ~110 ms |

---

## Baselines

| Scheduler | Description |
|-----------|-------------|
| FIFO | First-in first-out — default in TF-Serving, TorchServe |
| Round Robin | Equal interleaving, no priority |
| Static Priority | Fixed hand-assigned priority, no runtime adaptation |
| EDF | Earliest Deadline First (Liu & Layland 1973) — optimal under preemption; included to quantify the preemption-assumption violation gap |
| PQ+Deadline | Priority queue with deadline urgency bonus |
| QoS | Three-tier priority with intra-tier deadline ordering |
| Estimated-SJF | PAES with α=0, β=1, γ=0 — estimated Shortest Job First using latency-only ordering |
| **PAES** | **This work** |

All schedulers are non-preemptive — AI forward passes are atomic and cannot be interrupted.

---

## Scoring Term Analysis

At default weights α=β=γ=1 on the evaluation workload:

| Term | Range | Relative magnitude |
|------|-------|--------------------|
| αPᵢ | [1.0, 3.0] | ~100× dominant |
| β/Lᵢ | [0.007, 0.029] | primary latency driver |
| γ/Eᵢ | ≪ 0.01 | tie-breaker only |

Consequences (confirmed by ablation in Table III):
- Removing β produces the worst average latency (191.4 ms) — it is the primary driver of queue responsiveness.
- Setting α=0 achieves the best latency (150.1 ms) on the uniform synthetic workload by freeing β/L to govern ordering entirely — but causes priority starvation on asymmetric workloads.
- PAES incurs the highest per-task energy (1.28× FIFO via TDP proxy) because αPᵢ consistently promotes energy-intensive high-priority models. The γ/Eᵢ term is a tie-breaker, not a global energy minimizer.

For balanced three-way optimization, use the normalized variant (available in `scheduler.py`):

```python
score = alpha * (P / P_max) + beta * (L_min / L) + gamma * (E_min / E)
```

---

## Per-Device Results

### Intel Core i7-1165G7 — `results/i7-1165G7/`
*Primary device. 4 cores, 2.8–4.7 GHz, 16 GB RAM.*

**Synthetic workload — 600 tasks:**

| Scheduler | Avg Latency (ms) | Avg Queue Wait (ms) | Miss Rate |
|-----------|:---:|:---:|:---:|
| FIFO | 245.0 | 67,322 | 16.2% |
| Round Robin | 211.6 | 59,969 | 17.8% |
| Static Priority | 228.4 | 75,563 | 17.5% |
| EDF | 247.9 | 47,530 | 17.8% |
| PQ+Deadline | 257.4 | 55,015 | 18.2% |
| QoS | 233.5 | 46,513 | 17.8% |
| **PAES** | 237.2 | **45,400** | 18.0% |

**Robot pipeline — 685 tasks:** PAES queue wait **6,728 ms** vs. FIFO 10,155 ms (−33.7%), QoS 11,366 ms (−40.8%). All schedulers: 0% miss rate.

**Scheduling overhead:** PAES 0.787 µs mean, 1.4 µs P99 (0.001% of inference time).

---

### Intel Core Ultra 5 125H — `results/core-ultra-5/`
*14 cores, 32 GB RAM. PAES achieves its strongest results on this device.*

**Synthetic workload — 600 tasks:**

| Scheduler | Avg Latency (ms) | Avg Queue Wait (ms) | Miss Rate |
|-----------|:---:|:---:|:---:|
| FIFO | 85.5 | 25,931 | 0.2% |
| Round Robin | 84.8 | 24,981 | 0.0% |
| Static Priority | 82.5 | 24,844 | 0.0% |
| EDF | 80.4 | 18,911 | 0.0% |
| PQ+Deadline | 81.9 | 19,780 | 0.0% |
| QoS | 80.3 | 19,964 | 0.0% |
| **PAES** | **79.0** | **18,657** | 0.0% |

PAES: lowest latency (−7.6% vs. FIFO), lowest queue wait (−28.1% vs. FIFO), highest throughput (12.66 tps).

---

### NVIDIA Jetson Orin Nano — `results/jetson-orin-nano/`
*Genuine embedded edge GPU platform. 6-core ARM Cortex-A78AE, 8 GB RAM, CUDA 12.6. All five models evaluated with real CUDA inference.*

**Synthetic workload — 600 tasks:**

| Scheduler | Avg Latency (ms) | Avg Queue Wait (ms) | Miss Rate | Throughput (tps) |
|-----------|:---:|:---:|:---:|:---:|
| FIFO | 72.4 | 18,463 | 1.3% | 13.82 |
| Round Robin | 69.8 | 18,745 | 1.0% | 14.33 |
| Static Priority | 86.0 | 28,589 | 1.0% | 11.62 |
| EDF | 73.3 | 13,977 | 1.0% | 13.64 |
| PQ+Deadline | 76.3 | 15,667 | 1.0% | 13.11 |
| QoS | 58.5 | 11,915 | 1.0% | 17.10 |
| **PAES** | 65.9 | **13,165** | 1.0% | 15.18 |

PAES achieves a **28.7% queue wait reduction vs. FIFO** (13,165 ms vs. 18,463 ms).

**Robot pipeline — 685 tasks:** PAES queue wait **6,587 ms** vs. FIFO 9,943 ms (**−33.7%**), QoS 11,261 ms (**−41.5%**). All schedulers: 0% miss rate.

**Scheduling overhead:** PAES 1.72 ± 0.13 µs mean, 1.98 µs P99 (0.0022% of inference time on ARM).

---

### Raspberry Pi 5 — `results/raspberry-pi-5/`
*Cortex-A76, 4 cores @ 2.4 GHz, 8 GB RAM.*

**Synthetic workload — 600 tasks:** All schedulers converge at 0% miss rate and ~85–86 ms average latency. Queue wait is near-zero (2–4 ms) across all policies. This is the lower hardware boundary condition: the Pi 5 drains the 600-task queue as fast as tasks arrive, making scheduling order inconsequential at this load level — consistent with Section IV-F.

**Known data issue — robot pipeline:** The `exp_workload_realism.csv` for this device shows negative `avg_wait_ms` values. This is a clock reference bug in that specific run: `arrival_time` was captured before the scheduler loop started, producing negative queue wait deltas under fast execution. The synthetic experiment results (exp1–exp5) are unaffected and valid.

---

## Known Limitations

See Section V of the paper for full discussion. Summary:

- **Energy:** TDP-proxy estimates only. RAPL-based measurement planned.
- **Single-threaded:** One execution queue. Multi-core per-queue PAES with work-stealing is a planned extension.
- **Static profiles:** Task latency/energy estimates fixed at submission. Online EMA estimation planned.
- **Low-load miss rate:** PAES performs worst at low load (25.3%). A deadline-proximity bonus addresses this at threshold θ=150ms, with quantified improvement in Exp 2.
- **Statistical tests:** Evaluated using pairwise Wilcoxon signed-rank tests for queue wait (Exp 1).
- **GPU/NPU validation:** CPU and embedded GPU (NVIDIA Jetson Orin Nano CUDA) validated in this paper. Additional hardware acceleration targets (e.g., RK3588 NPU, Coral Edge TPU) remain future work.

---

## Citation

If you use PAES in your research, please cite our paper:

```bibtex
@article{khemani2026paes,
  title={PAES: Priority-Aware Adaptive Scheduling for Multi-Model Edge AI Systems},
  author={Khemani, Kushal and Maniar, Rushil and Lokhande, Ajinkyya},
  journal={IEEE Embedded Systems Letters},
  year={2026},
  publisher={IEEE}
}
```

---

## License

MIT License. See `LICENSE` for details.
