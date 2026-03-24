# Results — Raspberry Pi 5

**CPU:** ARM Cortex-A76, 4 cores @ 2.4 GHz  
**RAM:** 8 GB  
**Contributor:** Rushil Maniar  
**Role:** ARM / constrained platform. Lower hardware boundary condition.

## Key Numbers

All schedulers converge at 0% miss rate and ~85 ms average latency on the synthetic workload.
Queue wait is near-zero (2–4 ms) across all policies at 600-task load.

This is the lower hardware boundary condition described in Section IV-F: the Pi 5 processes
the 600-task queue as fast as tasks arrive, making scheduling order inconsequential at this
load level.

## Known Data Issue

`exp_workload_realism.csv` shows **negative `avg_wait_ms` values**. This is a clock reference
bug in that specific run: `arrival_time` was captured before the scheduler event loop began,
producing negative queue wait deltas under fast execution. The synthetic experiment results
(exp1–exp5) are unaffected and valid.

The robot pipeline result on this device is therefore excluded from the paper's reported
robot pipeline averages.

## Notes

- Some models used calibrated simulation fallbacks due to memory constraints when loading
  all 5 models simultaneously.
- Scheduling overhead on Pi 5: PAES 1.234 µs mean (1.23 µs reported in paper, Section IV-D).
