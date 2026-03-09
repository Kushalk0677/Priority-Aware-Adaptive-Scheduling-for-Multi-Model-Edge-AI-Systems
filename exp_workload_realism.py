"""
exp_workload_realism.py — Realistic robot pipeline workload experiment.

Instead of uniform random task arrivals, this simulates a real-world
edge AI pipeline: a robot assistant with three concurrent input streams.

Pipeline:
  Camera  → object detection (bursty, every ~100ms with variance)
  Mic     → speech recognition (triggered, Poisson arrivals)
  Planner → LLM reasoning (periodic, every ~2s)

Calibrated to hardware results (exp1_latency.csv):
PAES p50=77ms, avg=214ms, avg_wait=41,375ms
"""

import time
import random
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable
import heapq

random.seed(42)
np.random.seed(42)

def sim_mobilenet():
    """Image classification ~35ms"""
    time.sleep(random.gauss(35, 6) / 1000)
    return 35.0, 0.42

def sim_yolo():
    """Object detection ~80ms"""
    time.sleep(random.gauss(80, 12) / 1000)
    return 80.0, 1.02

def sim_whisper():
    """Speech recognition ~150ms"""
    time.sleep(random.gauss(150, 20) / 1000)
    return 150.0, 2.03

def sim_distilbert():
    """NLP/planning ~55ms"""
    time.sleep(random.gauss(55, 8) / 1000)
    return 55.0, 0.62

def sim_midas():
    """Depth estimation ~110ms"""
    time.sleep(random.gauss(110, 15) / 1000)
    return 110.0, 1.32


MODEL_FNS = {
    "yolov5n":              (sim_yolo,       3.0, 80.0,  1.02, 300.0),
    "mobilenet_v2":         (sim_mobilenet,  2.0, 35.0,  0.42, 200.0),
    "whisper_tiny":         (sim_whisper,    2.0, 150.0, 2.03, 500.0),
    "distilbert_sentiment": (sim_distilbert, 1.5, 55.0,  0.62, 400.0),
    "midas_small":          (sim_midas,      1.0, 110.0, 1.32, 600.0),
}

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


# ── Task ──────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    model_name:          str
    priority:            float
    expected_latency_ms: float
    expected_energy_mj:  float
    deadline_ms:         float
    run_fn:              Callable
    arrival_time:        float = field(default_factory=time.perf_counter)
    source:              str   = "unknown"   # camera / mic / planner

    def paes_score(self, a=1.0, b=1.0, g=1.0):
        return (a * self.priority
                + b * (1/max(self.expected_latency_ms, 1e-6))
                + g * (1/max(self.expected_energy_mj,  1e-6)))


# ── Robot pipeline workload generator ─────────────────────────────────────────

def generate_robot_pipeline(duration_sec: float = 30.0) -> list[Task]:
    """
    Simulate a robot assistant over `duration_sec` seconds.

    Three concurrent streams:
      Camera  → YOLO detections every ~100ms (bursty)
      Mic     → Whisper whenever speech detected (Poisson, avg 1/3s)
      Planner → DistilBERT reasoning every ~2s
      Depth   → MiDaS background depth scan every ~500ms
      Classify→ MobileNet quick classification every ~200ms
    """
    tasks   = []
    t       = 0.0

    # Stream 1: Camera — object detection, bursty (100ms ± 30ms)
    cam_t = 0.0
    while cam_t < duration_sec:
        fn, pri, lat, energy, deadline = MODEL_FNS["yolov5n"]
        tasks.append(Task("yolov5n", pri, lat, energy, deadline, fn,
                          arrival_time=cam_t, source="camera"))
        # Occasional burst: 3-5 rapid frames
        if random.random() < 0.15:
            burst_n = random.randint(2, 4)
            for i in range(burst_n):
                bt = cam_t + (i+1) * random.uniform(10, 30) / 1000
                if bt < duration_sec:
                    tasks.append(Task("yolov5n", pri, lat, energy, deadline, fn,
                                      arrival_time=bt, source="camera_burst"))
        cam_t += random.gauss(0.10, 0.03)

    # Stream 2: Microphone — speech, Poisson (avg every 3s)
    mic_t = random.expovariate(1/3.0)
    while mic_t < duration_sec:
        fn, pri, lat, energy, deadline = MODEL_FNS["whisper_tiny"]
        tasks.append(Task("whisper_tiny", pri, lat, energy, deadline, fn,
                          arrival_time=mic_t, source="microphone"))
        mic_t += random.expovariate(1/3.0)

    # Stream 3: Planner — periodic reasoning every ~2s
    plan_t = 0.5
    while plan_t < duration_sec:
        fn, pri, lat, energy, deadline = MODEL_FNS["distilbert_sentiment"]
        tasks.append(Task("distilbert_sentiment", pri, lat, energy, deadline, fn,
                          arrival_time=plan_t, source="planner"))
        plan_t += random.gauss(2.0, 0.3)

    # Stream 4: Depth sensor — background scan every ~500ms
    depth_t = 0.2
    while depth_t < duration_sec:
        fn, pri, lat, energy, deadline = MODEL_FNS["midas_small"]
        tasks.append(Task("midas_small", pri, lat, energy, deadline, fn,
                          arrival_time=depth_t, source="depth_sensor"))
        depth_t += random.gauss(0.5, 0.08)

    # Stream 5: Quick classification every ~200ms
    cls_t = 0.05
    while cls_t < duration_sec:
        fn, pri, lat, energy, deadline = MODEL_FNS["mobilenet_v2"]
        tasks.append(Task("mobilenet_v2", pri, lat, energy, deadline, fn,
                          arrival_time=cls_t, source="classifier"))
        cls_t += random.gauss(0.2, 0.04)

    # Sort by arrival time
    tasks.sort(key=lambda t: t.arrival_time)
    return tasks


# ── Scheduler runner ──────────────────────────────────────────────────────────

def run_realistic_workload(mode: str, tasks: list[Task],
                            alpha=1.0, beta=1.0, gamma=1.0) -> dict:
    q       = []
    counter = 0
    rr_idx  = 0
    results = []

    QOS_HIGH = 2.5
    QOS_MED  = 1.5

    def score(task):
        nonlocal rr_idx
        now = time.perf_counter()
        if mode == "fifo":
            return task.arrival_time
        elif mode == "round_robin":
            s = float(rr_idx); rr_idx += 1; return s
        elif mode == "static_priority":
            return -task.priority
        elif mode == "edf":
            return task.arrival_time + task.deadline_ms / 1000.0
        elif mode == "pq_deadline":
            time_to_ddl = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            return -(task.priority + 1.0/time_to_ddl)
        elif mode == "qos":
            time_to_ddl = max((task.arrival_time + task.deadline_ms/1000.0) - now, 1e-6)
            urgency = 1.0 / time_to_ddl
            tier = 0 if task.priority >= QOS_HIGH else (1 if task.priority >= QOS_MED else 2)
            return (tier, -urgency)
        elif mode == "paes":
            return -task.paes_score(alpha, beta, gamma)

    for t in tasks:
        heapq.heappush(q, (score(t), counter, t))
        counter += 1

    while q:
        _, _, task = heapq.heappop(q)
        exec_start = time.perf_counter()
        wait_ms    = (exec_start - task.arrival_time) * 1000
        lat_ms, energy_mj = task.run_fn()
        results.append({
            "model":         task.model_name,
            "source":        task.source,
            "latency_ms":    lat_ms,
            "energy_mj":     energy_mj,
            "wait_ms":       wait_ms,
            "deadline_ms":   task.deadline_ms,
            "missed":        lat_ms > task.deadline_ms,
            "scheduler":     mode,
        })

    df = pd.DataFrame(results)
    return {
        "scheduler":     mode,
        "n_tasks":       len(results),
        "avg_latency_ms":round(df["latency_ms"].mean(), 2),
        "p95_latency_ms":round(float(np.percentile(df["latency_ms"], 95)), 2),
        "p99_latency_ms":round(float(np.percentile(df["latency_ms"], 99)), 2),
        "avg_wait_ms":   round(df["wait_ms"].mean(), 2),
        "miss_rate":     round(df["missed"].mean(), 4),
        "avg_energy_mj": round(df["energy_mj"].mean(), 4),
        "throughput_tps":round(len(results) / max(df["latency_ms"].sum()/1000, 1e-6), 2),
    }, df


# ── Main ──────────────────────────────────────────────────────────────────────

DURATION = 30.0   # seconds of simulated robot operation

print("="*60)
print("Workload Realism Experiment — Robot Pipeline Simulation")
print(f"  Simulating {DURATION}s of robot operation per scheduler")
print("="*60)

# Generate tasks once, reuse for all schedulers
base_tasks = generate_robot_pipeline(DURATION)
print(f"\n  Generated {len(base_tasks)} tasks from robot pipeline")

source_counts = {}
for t in base_tasks:
    source_counts[t.source] = source_counts.get(t.source, 0) + 1
for src, cnt in sorted(source_counts.items()):
    print(f"    {src:<20} {cnt} tasks")

all_stats = []
all_dfs   = {}

for mode in SCHEDULERS:
    print(f"\n  Running {mode}...", end=" ", flush=True)
    # Rebuild tasks with fresh arrival times offset from now
    tasks = []
    base_t = time.perf_counter()
    for t in base_tasks:
        fn, pri, lat, energy, deadline = MODEL_FNS[t.model_name]
        tasks.append(Task(
            model_name=t.model_name, priority=t.priority,
            expected_latency_ms=t.expected_latency_ms,
            expected_energy_mj=t.expected_energy_mj,
            deadline_ms=t.deadline_ms, run_fn=fn,
            arrival_time=base_t + t.arrival_time,
            source=t.source,
        ))

    stats, df = run_realistic_workload(mode, tasks)
    all_stats.append(stats)
    all_dfs[mode] = df
    print(f"avg_lat={stats['avg_latency_ms']}ms  "
          f"miss={stats['miss_rate']:.1%}  "
          f"wait={stats['avg_wait_ms']:.0f}ms")

# ── Results ───────────────────────────────────────────────────────────────────

df_stats = pd.DataFrame(all_stats).set_index("scheduler")
print("\n  Results:")
print(df_stats[["avg_latency_ms","p95_latency_ms","avg_wait_ms",
                "miss_rate","throughput_tps"]].to_string())

Path("results").mkdir(exist_ok=True)
df_stats.to_csv("results/exp_workload_realism.csv")
print("\n  Saved → results/exp_workload_realism.csv")

# Per-source miss rate for PAES vs FIFO
print("\n  Per-source miss rate (PAES vs FIFO):")
for src in ["camera","camera_burst","microphone","planner","depth_sensor","classifier"]:
    paes_sub = all_dfs["paes"][all_dfs["paes"]["source"]==src]
    fifo_sub = all_dfs["fifo"][all_dfs["fifo"]["source"]==src]
    if len(paes_sub) > 0 and len(fifo_sub) > 0:
        print(f"    {src:<20} PAES={paes_sub['missed'].mean():.1%}  "
              f"FIFO={fifo_sub['missed'].mean():.1%}")

# ── Figures ───────────────────────────────────────────────────────────────────

Path("figures").mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

sched_list = SCHEDULERS
x = np.arange(len(sched_list))
xlabels = [LABELS[s] for s in sched_list]
bar_colors = [COLORS[s] for s in sched_list]

# Plot 1: Avg latency
bars = axes[0].bar(x, [df_stats.loc[s,"avg_latency_ms"] for s in sched_list],
                    color=bar_colors, edgecolor="white", width=0.6)
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                 f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
axes[0].set_xticks(x); axes[0].set_xticklabels(xlabels, rotation=15, ha="right")
axes[0].set_title("Avg Latency (ms) — Robot Pipeline\n↓ lower is better")
axes[0].set_ylabel("ms")

# Plot 2: Miss rate
bars = axes[1].bar(x, [df_stats.loc[s,"miss_rate"]*100 for s in sched_list],
                    color=bar_colors, edgecolor="white", width=0.6)
for b in bars:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                 f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
axes[1].set_xticks(x); axes[1].set_xticklabels(xlabels, rotation=15, ha="right")
axes[1].set_title("Deadline Miss Rate — Robot Pipeline\n↓ lower is better")
axes[1].set_ylabel("%")

# Plot 3: Avg queue wait
bars = axes[2].bar(x, [df_stats.loc[s,"avg_wait_ms"] for s in sched_list],
                    color=bar_colors, edgecolor="white", width=0.6)
for b in bars:
    axes[2].text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                 f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
axes[2].set_xticks(x); axes[2].set_xticklabels(xlabels, rotation=15, ha="right")
axes[2].set_title("Avg Queue Wait (ms) — Robot Pipeline\n↓ lower is better")
axes[2].set_ylabel("ms")

# Plot 4: Per-source miss rate heatmap (PAES vs FIFO)
sources = ["camera","camera_burst","microphone","planner","depth_sensor","classifier"]
source_labels = ["Camera\n(detection)","Camera\n(burst)","Microphone\n(speech)",
                  "Planner\n(reasoning)","Depth\n(sensor)","Classifier\n(quick)"]
paes_miss = []
fifo_miss = []
for src in sources:
    p = all_dfs["paes"][all_dfs["paes"]["source"]==src]["missed"].mean()
    f = all_dfs["fifo"][all_dfs["fifo"]["source"]==src]["missed"].mean()
    paes_miss.append(p*100 if not np.isnan(p) else 0)
    fifo_miss.append(f*100 if not np.isnan(f) else 0)

xs = np.arange(len(sources))
w  = 0.35
axes[3].bar(xs - w/2, fifo_miss, w, label="FIFO",       color=COLORS["fifo"],  edgecolor="white")
axes[3].bar(xs + w/2, paes_miss, w, label="PAES (ours)", color=COLORS["paes"], edgecolor="white")
axes[3].set_xticks(xs)
axes[3].set_xticklabels(source_labels, fontsize=9)
axes[3].set_title("Miss Rate by Task Source\nPAES vs FIFO")
axes[3].set_ylabel("%")
axes[3].legend()

fig.suptitle("Figure 8 — Realistic Robot Pipeline Workload Evaluation",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("figures/fig8_workload_realism.png", dpi=150, bbox_inches="tight")
print("\n  Saved → figures/fig8_workload_realism.png")
plt.close()

print("\n✓ Workload realism experiment complete.")
print(f"  PAES avg wait: {df_stats.loc['paes','avg_wait_ms']:.0f}ms  "
      f"vs FIFO: {df_stats.loc['fifo','avg_wait_ms']:.0f}ms")