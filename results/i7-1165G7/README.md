# Results — Intel Core i7-1165G7

**CPU:** Tiger Lake, 4 cores / 8 threads, 2.8–4.7 GHz  
**RAM:** 16 GB  
**OS:** Windows 11 / Linux  
**Role:** Primary device. Represents the productive operating range for laptop-class edge deployments.

## Files

| File | Contents |
|------|----------|
| `exp1_latency.csv` | Avg/P50/P95/P99 latency, queue wait, miss rate, throughput, energy — 600 tasks, 7 schedulers |
| `exp2_deadline.csv` | Miss rate ± std.dev. at 4 load levels (30/80/160/300 tasks), n=10 repeats per level |
| `exp3_energy.csv` | Avg and total energy (TDP-proxy estimate) per scheduler |
| `exp4_burst.csv` | Pre/during/post-burst miss rate and P95 latency |
| `exp5_sensitivity.csv` | 7 α/β/γ weight configurations — latency, miss rate, energy |
| `exp_overhead.csv` | Scheduler decision overhead: mean, median, P99, max (µs) |
| `exp_workload_realism.csv` | Robot pipeline: 685 tasks, 6 sensor streams, 30s simulation |

## Key Numbers (Paper Tables)

**Table II headline (this device, single run):**
- PAES queue wait: 45,400 ms (−32.6% vs. FIFO 67,322 ms)
- PAES avg latency: 237.2 ms

**Figure 2 (deadline miss rate @ high load, this device):**
- PAES: 17.1% — lowest of 7 schedulers

**Robot pipeline (Section IV-D):**
- PAES queue wait: 6,728 ms (−33.7% vs. FIFO 10,155 ms; −40.8% vs. QoS 11,366 ms)
- All schedulers: 0% miss rate (workload within capacity)
