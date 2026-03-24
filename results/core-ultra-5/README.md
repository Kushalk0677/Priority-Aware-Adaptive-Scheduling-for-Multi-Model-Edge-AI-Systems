# Results — Intel Core Ultra 5 125H

**CPU:** Meteor Lake, 14 cores, up to 4.5 GHz  
**RAM:** 32 GB  
**Contributor:** Rushil Maniar  
**Role:** Strongest PAES results. High-core-count laptop-class platform.

## Key Numbers

PAES achieves its strongest results on this device:
- Avg latency: 78.96 ms (−7.6% vs. FIFO 85.45 ms)
- Queue wait: 18,657 ms (−28.1% vs. FIFO 25,931 ms)
- Throughput: 12.66 tps (highest of all schedulers)
- Miss rate: 0.0% (workload within capacity at 600-task load)

## Notes

- All 5 real models loaded successfully on this device (no simulation fallbacks in exp1–exp5).
- Scheduling overhead on this device: PAES 0.787 µs mean (similar to i7).
