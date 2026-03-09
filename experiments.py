"""
experiments.py 

  Exp 1: Latency under multi-model load
  Exp 2: Real-time deadline miss rate vs load level
  Exp 3: Energy consumption per task
  Exp 4: Burst workload recovery
  Exp 5: Sensitivity analysis (α/β/γ sweep)
"""

import random
import time
import numpy as np
import pandas as pd
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

from scheduler import Scheduler, Task

SCHEDULERS = ["fifo", "round_robin", "static_priority", "edf", "pq_deadline", "qos", "paes"]


# ── Task factory ──────────────────────────────────────────────────────────────

def make_task(model_name: str, model_instance, input_data=None) -> Task:
    return Task(
        model_name          = model_name,
        priority            = model_instance.priority,
        expected_latency_ms = _estimate_latency(model_instance),
        expected_energy_mj  = _estimate_energy(model_instance),
        deadline_ms         = model_instance.deadline_ms,
        run_fn              = model_instance.run,
        input_data          = input_data or model_instance.make_dummy_input(),
        arrival_time        = time.perf_counter(),
    )

LATENCY_PRIORS = {
    "mobilenet_v2":         35.0,
    "yolov5n":              80.0,
    "whisper_tiny":        150.0,
    "distilbert_sentiment": 55.0,
    "midas_small":         110.0,
}
ENERGY_PRIORS = {
    "mobilenet_v2":         0.42,
    "yolov5n":              1.02,
    "whisper_tiny":         2.03,
    "distilbert_sentiment": 0.62,
    "midas_small":          1.32,
}

def _estimate_latency(m): return LATENCY_PRIORS.get(m.name, 100.0)
def _estimate_energy(m):  return ENERGY_PRIORS.get(m.name, 1.0)


def build_task_batch(models: dict, n: int, weights=None) -> list[Task]:
    """Generate n tasks, sampling models according to weights."""
    names = list(models.keys())
    if weights is None:
        weights = [1.0] * len(names)
    chosen = random.choices(names, weights=weights, k=n)
    return [make_task(name, models[name]) for name in chosen]


def run_scheduler_on_tasks(mode: str, tasks: list[Task],
                            alpha=1.0, beta=1.0, gamma=1.0) -> dict:
    sched = Scheduler(mode=mode, alpha=alpha, beta=beta, gamma=gamma)
    for t in tasks:
        sched.submit(t)
    sched.run_all()
    return sched.stats(), sched


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Latency Under Multi-Model Load
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_1_latency(models: dict, n_tasks=300) -> pd.DataFrame:
    """
    Run all 4 schedulers on the same mixed workload.
    Measures avg, p50, p95, p99 latency and throughput.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1 — Latency Under Multi-Model Load")
    print(f"  {n_tasks} tasks, {len(models)} models, 7 schedulers")
    print(f"{'='*60}")

    # Same task set for all schedulers (fair comparison)
    base_tasks = build_task_batch(models, n_tasks)

    rows = []
    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        # Deep-copy tasks so each scheduler gets fresh arrival times
        tasks = [Task(
            model_name          = t.model_name,
            priority            = t.priority,
            expected_latency_ms = t.expected_latency_ms,
            expected_energy_mj  = t.expected_energy_mj,
            deadline_ms         = t.deadline_ms,
            run_fn              = t.run_fn,
            input_data          = t.input_data,
            arrival_time        = time.perf_counter(),
        ) for t in base_tasks]

        stats, _ = run_scheduler_on_tasks(mode, tasks)
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("scheduler")
    print("\n  Results:")
    print(df[["avg_latency_ms","p95_latency_ms","p99_latency_ms",
              "throughput_tps","miss_rate"]].to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Deadline Miss Rate vs Load Level
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_2_deadline(models: dict, repeats=3) -> pd.DataFrame:
    """
    Sweep task submission rate from low to extreme.
    At each load level, measure deadline miss rate for each scheduler.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 2 — Deadline Miss Rate vs Load Level")
    print(f"{'='*60}")

    load_levels = {
        "low":     30,
        "medium":  80,
        "high":    160,
        "extreme": 300,
    }

    rows = []
    for load_name, n in load_levels.items():
        for mode in SCHEDULERS:
            miss_rates = []
            for _ in range(repeats):
                tasks = build_task_batch(models, n)
                stats, _ = run_scheduler_on_tasks(mode, tasks)
                miss_rates.append(stats["miss_rate"])
            rows.append({
                "load_level": load_name,
                "n_tasks":    n,
                "scheduler":  mode,
                "miss_rate":  round(float(np.mean(miss_rates)), 4),
                "miss_rate_std": round(float(np.std(miss_rates)), 4),
            })
        print(f"  Load '{load_name}' ({n} tasks) — done")

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="load_level", columns="scheduler", values="miss_rate")
    pivot = pivot.reindex(["low","medium","high","extreme"])
    print("\n  Miss Rate by Load:")
    print(pivot.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Energy Consumption Per Task
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_3_energy(models: dict, n_tasks=200) -> pd.DataFrame:
    """
    Measure average energy (mJ) per completed task for each scheduler.
    PAES should waste less energy by avoiding stalled or idle cycles.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 3 — Energy Consumption Per Task")
    print(f"{'='*60}")

    rows = []
    base_tasks = build_task_batch(models, n_tasks)

    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        tasks = [Task(
            model_name          = t.model_name,
            priority            = t.priority,
            expected_latency_ms = t.expected_latency_ms,
            expected_energy_mj  = t.expected_energy_mj,
            deadline_ms         = t.deadline_ms,
            run_fn              = t.run_fn,
            input_data          = t.input_data,
            arrival_time        = time.perf_counter(),
        ) for t in base_tasks]

        stats, _ = run_scheduler_on_tasks(mode, tasks)
        rows.append({
            "scheduler":       mode,
            "avg_energy_mj":   stats["avg_energy_mj"],
            "total_energy_mj": stats["total_energy_mj"],
            "n_tasks":         stats["n_tasks"],
        })

    df = pd.DataFrame(rows).set_index("scheduler")
    # Normalize to FIFO = 1.0x
    baseline = df.loc["fifo", "avg_energy_mj"]
    df["relative_energy"] = (df["avg_energy_mj"] / baseline).round(3)
    print("\n  Energy Results:")
    print(df.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Burst Workload Recovery
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_4_burst(models: dict) -> pd.DataFrame:
    """
    Simulate: normal → burst → normal load.
    Measure queue depth and miss rate during each phase.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 4 — Burst Workload Recovery")
    print(f"{'='*60}")

    phases = [
        ("pre_burst",    40),
        ("burst",       160),
        ("post_burst",   40),
    ]

    rows = []
    for mode in SCHEDULERS:
        phase_stats = []
        for phase_name, n in phases:
            tasks = build_task_batch(models, n)
            stats, sched = run_scheduler_on_tasks(mode, tasks)
            phase_stats.append({
                "phase":      phase_name,
                "miss_rate":  stats["miss_rate"],
                "avg_lat":    stats["avg_latency_ms"],
                "p95_lat":    stats["p95_latency_ms"],
            })

        # Recovery = how much p95 latency increases during burst vs pre_burst
        pre_p95   = phase_stats[0]["p95_lat"]
        burst_p95 = phase_stats[1]["p95_lat"]
        post_p95  = phase_stats[2]["p95_lat"]
        recovery  = (post_p95 - pre_p95) / max(pre_p95, 1e-6)  # % overshoot

        rows.append({
            "scheduler":            mode,
            "pre_burst_miss_rate":  phase_stats[0]["miss_rate"],
            "burst_miss_rate":      phase_stats[1]["miss_rate"],
            "post_burst_miss_rate": phase_stats[2]["miss_rate"],
            "pre_p95_ms":           pre_p95,
            "burst_p95_ms":         burst_p95,
            "post_p95_ms":          post_p95,
            "recovery_overshoot":   round(recovery, 4),
        })
        print(f"  {mode}: burst miss={phase_stats[1]['miss_rate']:.1%}, "
              f"recovery overshoot={recovery:.1%}")

    df = pd.DataFrame(rows).set_index("scheduler")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5 — PAES Sensitivity Analysis (α/β/γ sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_5_sensitivity(models: dict, n_tasks=150) -> pd.DataFrame:
    """
    Vary α, β, γ and see how PAES performance changes.
    This shows which dimension of the scoring function matters most.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 5 — PAES Sensitivity Analysis (α/β/γ sweep)")
    print(f"{'='*60}")

    configs = [
        # label           α    β    γ
        ("balanced",      1.0, 1.0, 1.0),
        ("priority-heavy",3.0, 1.0, 1.0),
        ("latency-heavy", 1.0, 3.0, 1.0),
        ("energy-heavy",  1.0, 1.0, 3.0),
        ("no-priority",   0.0, 1.0, 1.0),
        ("no-latency",    1.0, 0.0, 1.0),
        ("no-energy",     1.0, 1.0, 0.0),
    ]

    base_tasks = build_task_batch(models, n_tasks)
    rows = []

    for label, a, b, g in tqdm(configs, desc="Configs"):
        tasks = [Task(
            model_name          = t.model_name,
            priority            = t.priority,
            expected_latency_ms = t.expected_latency_ms,
            expected_energy_mj  = t.expected_energy_mj,
            deadline_ms         = t.deadline_ms,
            run_fn              = t.run_fn,
            input_data          = t.input_data,
            arrival_time        = time.perf_counter(),
        ) for t in base_tasks]

        stats, _ = run_scheduler_on_tasks("paes", tasks, alpha=a, beta=b, gamma=g)
        rows.append({
            "config":         label,
            "alpha":          a, "beta": b, "gamma": g,
            "avg_latency_ms": stats["avg_latency_ms"],
            "p95_latency_ms": stats["p95_latency_ms"],
            "miss_rate":      stats["miss_rate"],
            "avg_energy_mj":  stats["avg_energy_mj"],
        })

    df = pd.DataFrame(rows).set_index("config")
    print("\n  Sensitivity Results:")
    print(df[["avg_latency_ms","p95_latency_ms","miss_rate","avg_energy_mj"]].to_string())
    return df
