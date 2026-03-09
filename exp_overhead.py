"""
exp_overhead.py — Measure scheduler decision overhead.

This answers the reviewer question:
  "Does PAES itself add significant latency?"

We time how long each scheduler takes to make a scheduling decision
(i.e. score + heappush + heappop) across 100,000 iterations.
"""

import time
import random
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from pathlib import Path

# p50 latency per model (ms) — used as expected_latency_ms in tasks
MODEL_PROFILES = [
    # name,                  priority  lat_ms  energy_mj  deadline_ms
    ("mobilenet_v2",         2.0,      35.0,   0.42,      200.0),
    ("yolov5n",              3.0,      80.0,   1.02,      300.0),
    ("whisper_tiny",         2.0,      150.0,  2.03,      500.0),
    ("distilbert_sentiment", 1.5,      55.0,   0.62,      400.0),
    ("midas_small",          1.0,      110.0,  1.32,      600.0),
]

N_ITERATIONS = 100_000   # number of scheduling decisions to time
WARMUP       = 1_000     # warmup iterations (not counted)

SCHEDULERS = ["fifo", "round_robin", "static_priority", "edf", "pq_deadline", "qos", "paes"]
LABELS = {
    "fifo":            "FIFO",
    "round_robin":     "Round Robin",
    "static_priority": "Static Priority",
    "edf":             "EDF",
    "pq_deadline":     "PQ+Deadline",
    "qos":             "QoS",
    "paes":            "PAES (ours)",
}
COLORS = {
    "fifo":            "#e74c3c",
    "round_robin":     "#f39c12",
    "static_priority": "#3498db",
    "edf":             "#9b59b6",
    "pq_deadline":     "#1abc9c",
    "qos":             "#e67e22",
    "paes":            "#2ecc71",
}


# ── Minimal task + scoring (mirrors scheduler.py) ────────────────────────────

class LightTask:
    __slots__ = ["name","priority","lat","energy","deadline","arrival","counter"]
    def __init__(self, name, priority, lat, energy, deadline, counter):
        self.name     = name
        self.priority = priority
        self.lat      = lat
        self.energy   = energy
        self.deadline = deadline
        self.arrival  = time.perf_counter()
        self.counter  = counter

    def paes_score(self, a=1.0, b=1.0, g=1.0):
        return a*self.priority + b*(1/max(self.lat,1e-6)) + g*(1/max(self.energy,1e-6))


def make_random_task(counter):
    name, pri, lat, energy, deadline = random.choice(MODEL_PROFILES)
    return LightTask(name, pri, lat, energy, deadline, counter)


QOS_HIGH = 2.5
QOS_MED  = 1.5

def score_for_mode(task, mode, rr_index):
    if mode == "fifo":
        return task.arrival
    elif mode == "round_robin":
        return float(rr_index)
    elif mode == "static_priority":
        return -task.priority
    elif mode == "edf":
        return task.arrival + task.deadline / 1000.0
    elif mode == "pq_deadline":
        now = task.arrival  # use arrival as proxy for current time in overhead test
        ttd = max((task.arrival + task.deadline/1000.0) - now, 1e-6)
        return -(task.priority + 1.0/ttd)
    elif mode == "qos":
        now = task.arrival
        ttd = max((task.arrival + task.deadline/1000.0) - now, 1e-6)
        urgency = 1.0 / ttd
        tier = 0 if task.priority >= QOS_HIGH else (1 if task.priority >= QOS_MED else 2)
        return (tier, -urgency)
    elif mode == "paes":
        return -task.paes_score()


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark_scheduler(mode: str, n: int = N_ITERATIONS):
    """
    Repeatedly time a single scheduling decision:
    submit (heappush) + select (heappop)
    Returns list of decision times in microseconds.
    """
    queue   = []
    counter = 0
    rr_idx  = 0
    times   = []

    # Pre-fill queue so heappop always has something
    for _ in range(50):
        t = make_random_task(counter)
        counter += 1
        s = score_for_mode(t, mode, rr_idx)
        heapq.heappush(queue, (s, counter, t))

    # Warmup
    for _ in range(WARMUP):
        t = make_random_task(counter)
        counter += 1
        s = score_for_mode(t, mode, rr_idx)
        heapq.heappush(queue, (s, counter, t))
        heapq.heappop(queue)

    # Timed loop
    for _ in range(n):
        t = make_random_task(counter)
        counter += 1

        t0 = time.perf_counter_ns()
        s  = score_for_mode(t, mode, rr_idx)
        heapq.heappush(queue, (s, counter, t))
        heapq.heappop(queue)
        t1 = time.perf_counter_ns()

        times.append((t1 - t0) / 1_000)  # ns → µs
        if mode == "round_robin":
            rr_idx += 1

    return times


# ── Run ──────────────────────────────────────────────────────────────────────

print("="*60)
print("Scheduler Overhead Measurement")
print(f"  {N_ITERATIONS:,} decisions per scheduler")
print("="*60)

results = {}
for mode in SCHEDULERS:
    print(f"  Benchmarking {mode}...", end=" ", flush=True)
    times = benchmark_scheduler(mode)
    results[mode] = {
        "mean_us":   round(statistics.mean(times), 3),
        "median_us": round(statistics.median(times), 3),
        "p99_us":    round(float(np.percentile(times, 99)), 3),
        "max_us":    round(max(times), 3),
        "std_us":    round(statistics.stdev(times), 3),
    }
    # As % of your real p50 inference latency (77ms for PAES)
    pct = results[mode]["mean_us"] / (77.0 * 1000) * 100
    results[mode]["pct_of_inference"] = round(pct, 4)
    print(f"mean={results[mode]['mean_us']}µs  ({pct:.4f}% of inference)")

# ── Table ─────────────────────────────────────────────────────────────────────

df = pd.DataFrame(results).T
df.index.name = "scheduler"
print("\n  Overhead Results (µs):")
print(df.to_string())

Path("results").mkdir(exist_ok=True)
df.to_csv("results/exp_overhead.csv")
print("\n  Saved → results/exp_overhead.csv")

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

# Mean + p99 grouped bar
x      = np.arange(len(SCHEDULERS))
width  = 0.35
means  = [results[m]["mean_us"]  for m in SCHEDULERS]
p99s   = [results[m]["p99_us"]   for m in SCHEDULERS]

b1 = axes[0].bar(x - width/2, means, width,
                  label="Mean", color=[COLORS[m] for m in SCHEDULERS],
                  edgecolor="white")
b2 = axes[0].bar(x + width/2, p99s, width,
                  label="P99",  color=[COLORS[m] for m in SCHEDULERS],
                  alpha=0.5, edgecolor="white", hatch="//")

for bar in b1:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
for bar in b2:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

axes[0].set_xticks(x)
axes[0].set_xticklabels([LABELS[m] for m in SCHEDULERS], rotation=20, ha="right")
axes[0].set_ylabel("Decision Time (µs)")
axes[0].set_title("Scheduler Decision Overhead\n(mean and P99 — lower is better)")
axes[0].legend()

# Overhead as % of inference
pcts = [results[m]["pct_of_inference"] for m in SCHEDULERS]
bars = axes[1].bar(x, pcts,
                    color=[COLORS[m] for m in SCHEDULERS], edgecolor="white", width=0.6)
axes[1].axhline(0.01, color="red", linestyle="--", linewidth=1,
                label="0.01% threshold")
for bar, v in zip(bars, pcts):
    axes[1].text(bar.get_x()+bar.get_width()/2, v*1.05,
                 f"{v:.4f}%", ha="center", va="bottom", fontsize=7)
axes[1].set_xticks(x)
axes[1].set_xticklabels([LABELS[m] for m in SCHEDULERS], rotation=20, ha="right")
axes[1].set_ylabel("Overhead (% of p50 inference latency)")
axes[1].set_title("Overhead as % of Inference Time\n(all schedulers negligible)")
axes[1].legend()

fig.suptitle("Figure 7 — Scheduler Decision Overhead", fontweight="bold")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/fig7_overhead.png", dpi=150, bbox_inches="tight")
print("  Saved → figures/fig7_overhead.png")
plt.close()

print("\n overhead experiment complete.")
print(f"  PAES overhead: {results['paes']['mean_us']}µs mean")
print(f"  = {results['paes']['pct_of_inference']}% of real inference time")
