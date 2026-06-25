"""
experiments.py — v2

  Exp 1: Latency under multi-model load
          + estimated-SJF added as proper 8th baseline (α=γ=0, β=1)
          + Wilcoxon signed-rank tests vs PAES for all key metrics
  Exp 2: Real-time deadline miss rate vs load level
          + deadline proximity bonus variant (Fix [3]) shown at low load
  Exp 3: Energy consumption per task
  Exp 4: Burst workload recovery
  Exp 5: Sensitivity analysis (α/β/γ sweep)
  Exp 6: Arrival distribution sensitivity (NEW)
          — tests Poisson, bursty, and uniform distributions
          — addresses generalizability critique

Changes from v1:
  - SCHEDULERS list now includes "estimated_sjf" (PAES with α=γ=0)
    as a full proper baseline with its own row in all tables
  - run_scheduler_on_tasks() dispatches estimated_sjf correctly
  - wilcoxon_vs_paes() helper added for statistical tests
  - experiment_2_deadline() shows low-load miss rate with/without
    deadline proximity bonus to demonstrate Fix [3] effect
  - experiment_6_arrival_sensitivity() added (new)
"""

import random
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats as scipy_stats

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

from scheduler import Scheduler, Task

# ── Baseline list ─────────────────────────────────────────────────────────────
# estimated_sjf = PAES with α=γ=0, β=1 — latency-only scheduling using
# estimated (not exact) runtimes. Added as proper 8th baseline to address
# the critique that the latency term's independent contribution was not
# isolated in Table II of the original paper.
SCHEDULERS = [
    "fifo", "round_robin", "static_priority", "edf",
    "pq_deadline", "qos", "paes", "estimated_sjf",
]

SCHEDULER_LABELS = {
    "fifo":           "FIFO",
    "round_robin":    "Round Robin",
    "static_priority":"Static Priority",
    "edf":            "EDF",
    "pq_deadline":    "PQ+Deadline",
    "qos":            "QoS",
    "paes":           "PAES (ours)",
    "estimated_sjf":  "Est.-SJF (α=γ=0)",
}


# ── Task factory ──────────────────────────────────────────────────────────────

LATENCY_PRIORS = {
    "mobilenet_v2":          35.0,
    "yolov5n":               80.0,
    "whisper_tiny":         150.0,
    "distilbert_sentiment":  55.0,
    "midas_small":          110.0,
}
ENERGY_PRIORS = {
    "mobilenet_v2":          0.42,
    "yolov5n":               1.02,
    "whisper_tiny":          2.03,
    "distilbert_sentiment":  0.62,
    "midas_small":           1.32,
}

def _estimate_latency(m): return LATENCY_PRIORS.get(m.name, 100.0)
def _estimate_energy(m):  return ENERGY_PRIORS.get(m.name, 1.0)


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


def build_task_batch(models: dict, n: int, weights=None) -> list[Task]:
    names = list(models.keys())
    if weights is None:
        weights = [1.0] * len(names)
    chosen = random.choices(names, weights=weights, k=n)
    return [make_task(name, models[name]) for name in chosen]


def run_scheduler_on_tasks(mode: str, tasks: list[Task],
                            alpha=1.0, beta=1.0, gamma=1.0,
                            deadline_bonus=False) -> tuple:
    """
    Run a scheduler on a task list and return (stats_dict, scheduler_obj).

    Special cases:
      mode="estimated_sjf" → PAES with α=0, β=1, γ=0
        This is estimated-SJF: orders purely by 1/L_i using estimated
        latencies, no priority or energy. Added as proper baseline.
      deadline_bonus=True  → enables PAES deadline proximity bonus
        (PAES_DEADLINE_BONUS_THETA=150ms); used in Exp 2 low-load
        comparison to quantify Fix [3] effect.
    """
    if mode == "estimated_sjf":
        alpha, beta, gamma = 0.0, 1.0, 0.0
        mode = "paes"   # reuse PAES heap with SJF weights

    sched = Scheduler(mode=mode, alpha=alpha, beta=beta, gamma=gamma)

    if deadline_bonus and mode == "paes":
        # Temporarily lower theta to activate bonus more aggressively
        # at low load — used only in the bonus comparison experiment
        import scheduler as sched_module
        orig_theta = sched_module.PAES_DEADLINE_BONUS_THETA
        sched_module.PAES_DEADLINE_BONUS_THETA = 150.0
        for t in tasks:
            sched.submit(t)
        sched.run_all()
        sched_module.PAES_DEADLINE_BONUS_THETA = orig_theta
    else:
        for t in tasks:
            sched.submit(t)
        sched.run_all()

    return sched.stats(), sched


# ── Statistical helper ────────────────────────────────────────────────────────

def wilcoxon_vs_paes(metric_key: str, per_run_data: dict,
                     alpha_level: float = 0.05) -> dict:
    """
    Run pairwise Wilcoxon signed-rank tests comparing each scheduler
    against PAES on a given metric.

    per_run_data: dict[scheduler_name -> list of per-run metric values]
    Returns dict[scheduler_name -> {statistic, pvalue, significant, direction}]
    """
    paes_vals = np.array(per_run_data.get("paes", []))
    results = {}
    for mode, vals in per_run_data.items():
        if mode == "paes":
            continue
        arr = np.array(vals)
        if len(arr) < 2 or len(paes_vals) < 2:
            results[mode] = {"pvalue": None, "significant": False}
            continue
        # Use alternative="two-sided" — we report direction separately
        try:
            stat, pval = scipy_stats.wilcoxon(paes_vals, arr,
                                               alternative="two-sided")
        except ValueError:
            pval = 1.0
            stat = 0.0
        direction = "PAES<" if np.mean(paes_vals) < np.mean(arr) else "PAES>"
        results[mode] = {
            "statistic":   round(float(stat), 3),
            "pvalue":      round(float(pval), 4),
            "significant": pval < alpha_level,
            "direction":   direction,
            "paes_mean":   round(float(np.mean(paes_vals)), 2),
            "other_mean":  round(float(np.mean(arr)), 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Latency Under Multi-Model Load
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_1_latency(models: dict, n_tasks=300,
                          repeats=10) -> tuple[pd.DataFrame, dict]:
    """
    Run all 8 schedulers (including estimated-SJF) on the same mixed
    workload. Reports avg/p50/p95/p99 latency, queue wait, priority-
    weighted queue wait, throughput, and miss rate.

    Returns (results_df, wilcoxon_results).
    Wilcoxon tests compare each scheduler vs PAES on queue_wait_ms.
    """
    print(f"\n{'='*64}")
    print(f"Experiment 1 — Latency Under Multi-Model Load")
    print(f"  {n_tasks} tasks, {len(models)} models, {len(SCHEDULERS)} schedulers")
    print(f"  + estimated-SJF as proper 8th baseline")
    print(f"  + Wilcoxon signed-rank tests vs PAES")
    print(f"{'='*64}")

    rows = []
    # Collect per-run values for Wilcoxon
    per_run = defaultdict(lambda: defaultdict(list))

    base_tasks = build_task_batch(models, n_tasks)

    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        label = SCHEDULER_LABELS[mode]
        run_stats = []
        for r in range(repeats):
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
            run_stats.append(stats)
            sched_key = "paes" if mode == "estimated_sjf" else mode
            # Store under original mode name
            per_run[mode]["avg_wait_ms"].append(stats["avg_wait_ms"])
            per_run[mode]["miss_rate"].append(stats["miss_rate"])
            per_run[mode]["avg_latency_ms"].append(stats["avg_latency_ms"])

        def mean(k): return round(float(np.mean([s[k] for s in run_stats])), 2)
        def std(k):  return round(float(np.std( [s[k] for s in run_stats])), 2)

        rows.append({
            "scheduler":    mode,
            "label":        label,
            "n_tasks":      n_tasks,
            "avg_latency_ms":   mean("avg_latency_ms"),
            "lat_std":          std("avg_latency_ms"),
            "p50_latency_ms":   mean("p50_latency_ms"),
            "p95_latency_ms":   mean("p95_latency_ms"),
            "p99_latency_ms":   mean("p99_latency_ms"),
            "avg_wait_ms":      mean("avg_wait_ms"),
            "wait_std":         std("avg_wait_ms"),
            "priority_weighted_avg_wait_ms": mean("priority_weighted_avg_wait_ms"),
            "miss_rate":        mean("miss_rate"),
            "miss_std":         std("miss_rate"),
            "throughput_tps":   mean("throughput_tps"),
            "avg_energy_mj":    mean("avg_energy_mj"),
        })

    df = pd.DataFrame(rows).set_index("scheduler")

    # Wilcoxon: queue wait
    wilcoxon_wait = wilcoxon_vs_paes(
        "avg_wait_ms",
        {m: per_run[m]["avg_wait_ms"] for m in SCHEDULERS}
    )

    print(f"\n  {'Scheduler':<20} {'AvgLat':>10} {'QueueWait':>12} "
          f"{'PW-Wait':>10} {'MissRate':>10}")
    print("  " + "-"*66)
    for mode in SCHEDULERS:
        r = df.loc[mode]
        mk = " ◀" if mode == "paes" else ""
        print(f"  {SCHEDULER_LABELS[mode]:<20} {r['avg_latency_ms']:>10.1f} "
              f"{r['avg_wait_ms']:>12.1f} "
              f"{r['priority_weighted_avg_wait_ms']:>10.1f} "
              f"{r['miss_rate']:>10.4f}{mk}")

    print(f"\n  Wilcoxon tests (queue wait vs PAES):")
    print(f"  {'Scheduler':<20} {'p-value':>10} {'Sig':>6} {'Direction':>12}")
    print("  " + "-"*52)
    for mode, res in wilcoxon_wait.items():
        if res["pvalue"] is not None:
            sig = "YES" if res["significant"] else "no"
            print(f"  {SCHEDULER_LABELS[mode]:<20} {res['pvalue']:>10.4f} "
                  f"{sig:>6} {res['direction']:>12}")

    return df, {"queue_wait": wilcoxon_wait}


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Deadline Miss Rate vs Load Level
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_2_deadline(models: dict, repeats=3) -> pd.DataFrame:
    """
    Sweep task count from low to extreme. Shows miss rate for all 8
    schedulers. At low load specifically, shows PAES with and without
    the deadline proximity bonus (Fix [3]) to quantify its effect.
    """
    print(f"\n{'='*64}")
    print(f"Experiment 2 — Deadline Miss Rate vs Load Level")
    print(f"  + estimated-SJF as proper 8th baseline")
    print(f"  + Low-load comparison: PAES with/without deadline bonus")
    print(f"{'='*64}")

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
                "load_level":    load_name,
                "n_tasks":       n,
                "scheduler":     mode,
                "label":         SCHEDULER_LABELS[mode],
                "miss_rate":     round(float(np.mean(miss_rates)), 4),
                "miss_rate_std": round(float(np.std(miss_rates)),  4),
            })
        print(f"  Load '{load_name}' ({n} tasks) done")

    df = pd.DataFrame(rows)

    # ── Low-load bonus comparison ─────────────────────────────────────────────
    print(f"\n  Low-load (30 tasks) — PAES without vs with deadline bonus:")
    print(f"  (addresses critique: bonus was identified in v1 but not evaluated)")
    no_bonus_rates, bonus_rates = [], []
    for _ in range(repeats * 3):
        tasks = build_task_batch(models, 30)
        s1, _ = run_scheduler_on_tasks("paes", tasks, deadline_bonus=False)
        tasks = build_task_batch(models, 30)
        s2, _ = run_scheduler_on_tasks("paes", tasks, deadline_bonus=True)
        no_bonus_rates.append(s1["miss_rate"])
        bonus_rates.append(s2["miss_rate"])

    nb_mean = np.mean(no_bonus_rates)
    b_mean  = np.mean(bonus_rates)
    try:
        _, pval = scipy_stats.wilcoxon(no_bonus_rates, bonus_rates,
                                        alternative="two-sided")
        sig = f"p={pval:.4f}"
    except ValueError:
        sig = "n/a"

    print(f"    PAES no bonus:   {nb_mean:.4f}")
    print(f"    PAES with bonus: {b_mean:.4f}  ({sig})")
    improvement = (nb_mean - b_mean) / max(nb_mean, 1e-6) * 100
    print(f"    Improvement:     {improvement:+.1f}%")

    pivot = df.pivot(index="load_level", columns="scheduler",
                     values="miss_rate")
    pivot = pivot.reindex(["low","medium","high","extreme"])
    print("\n  Miss Rate by Load:")
    print(pivot[[m for m in SCHEDULERS]].to_string())

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Energy Consumption Per Task
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_3_energy(models: dict, n_tasks=200) -> pd.DataFrame:
    print(f"\n{'='*64}")
    print(f"Experiment 3 — Energy Consumption Per Task")
    print(f"  Note: TDP-proxy estimates, not hardware measurements")
    print(f"  estimated-SJF included as 8th baseline")
    print(f"{'='*64}")

    rows = []
    base_tasks = build_task_batch(models, n_tasks)

    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        tasks = [Task(
            model_name=t.model_name, priority=t.priority,
            expected_latency_ms=t.expected_latency_ms,
            expected_energy_mj=t.expected_energy_mj,
            deadline_ms=t.deadline_ms, run_fn=t.run_fn,
            input_data=t.input_data,
            arrival_time=time.perf_counter(),
        ) for t in base_tasks]
        stats, _ = run_scheduler_on_tasks(mode, tasks)
        rows.append({
            "scheduler":       mode,
            "label":           SCHEDULER_LABELS[mode],
            "avg_energy_mj":   stats["avg_energy_mj"],
            "total_energy_mj": stats["total_energy_mj"],
            "n_tasks":         stats["n_tasks"],
        })

    df = pd.DataFrame(rows).set_index("scheduler")
    baseline = df.loc["fifo", "avg_energy_mj"]
    df["relative_energy"] = (df["avg_energy_mj"] / baseline).round(3)
    print("\n  Energy Results (TDP-proxy, FIFO=1.0x):")
    print(df[["avg_energy_mj","relative_energy"]].to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Burst Workload Recovery
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_4_burst(models: dict) -> pd.DataFrame:
    print(f"\n{'='*64}")
    print(f"Experiment 4 — Burst Workload Recovery")
    print(f"{'='*64}")

    phases = [("pre_burst", 40), ("burst", 160), ("post_burst", 40)]
    rows = []

    for mode in SCHEDULERS:
        phase_stats = []
        for phase_name, n in phases:
            tasks = build_task_batch(models, n)
            stats, _ = run_scheduler_on_tasks(mode, tasks)
            phase_stats.append({
                "phase":      phase_name,
                "miss_rate":  stats["miss_rate"],
                "avg_lat":    stats["avg_latency_ms"],
                "p95_lat":    stats["p95_latency_ms"],
            })
        pre_p95   = phase_stats[0]["p95_lat"]
        burst_p95 = phase_stats[1]["p95_lat"]
        post_p95  = phase_stats[2]["p95_lat"]
        recovery  = (post_p95 - pre_p95) / max(pre_p95, 1e-6)

        rows.append({
            "scheduler":            mode,
            "label":                SCHEDULER_LABELS[mode],
            "pre_burst_miss_rate":  phase_stats[0]["miss_rate"],
            "burst_miss_rate":      phase_stats[1]["miss_rate"],
            "post_burst_miss_rate": phase_stats[2]["miss_rate"],
            "pre_p95_ms":           pre_p95,
            "burst_p95_ms":         burst_p95,
            "post_p95_ms":          post_p95,
            "recovery_overshoot":   round(recovery, 4),
        })
        print(f"  {SCHEDULER_LABELS[mode]:<22} burst_miss="
              f"{phase_stats[1]['miss_rate']:.1%}  "
              f"recovery={recovery:.1%}")

    return pd.DataFrame(rows).set_index("scheduler")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5 — PAES Sensitivity Analysis (α/β/γ sweep)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_5_sensitivity(models: dict, n_tasks=150) -> pd.DataFrame:
    print(f"\n{'='*64}")
    print(f"Experiment 5 — PAES Sensitivity Analysis (α/β/γ sweep)")
    print(f"{'='*64}")

    configs = [
        ("balanced",       1.0, 1.0, 1.0),
        ("priority-heavy", 3.0, 1.0, 1.0),
        ("latency-heavy",  1.0, 3.0, 1.0),
        ("energy-heavy",   1.0, 1.0, 3.0),
        ("no-priority",    0.0, 1.0, 1.0),
        ("no-latency",     1.0, 0.0, 1.0),
        ("no-energy",      1.0, 1.0, 0.0),
    ]

    base_tasks = build_task_batch(models, n_tasks)
    rows = []

    for label, a, b, g in tqdm(configs, desc="Configs"):
        tasks = [Task(
            model_name=t.model_name, priority=t.priority,
            expected_latency_ms=t.expected_latency_ms,
            expected_energy_mj=t.expected_energy_mj,
            deadline_ms=t.deadline_ms, run_fn=t.run_fn,
            input_data=t.input_data,
            arrival_time=time.perf_counter(),
        ) for t in base_tasks]

        stats, _ = run_scheduler_on_tasks("paes", tasks,
                                           alpha=a, beta=b, gamma=g)
        rows.append({
            "config":         label,
            "alpha": a, "beta": b, "gamma": g,
            "avg_latency_ms": stats["avg_latency_ms"],
            "p95_latency_ms": stats["p95_latency_ms"],
            "miss_rate":      stats["miss_rate"],
            "avg_energy_mj":  stats["avg_energy_mj"],
            "priority_weighted_avg_wait_ms":
                              stats["priority_weighted_avg_wait_ms"],
        })

    df = pd.DataFrame(rows).set_index("config")
    print("\n  Sensitivity Results:")
    print(df[["avg_latency_ms","p95_latency_ms","miss_rate",
              "avg_energy_mj","priority_weighted_avg_wait_ms"]].to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 6 — Arrival Distribution Sensitivity (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_6_arrival_sensitivity(models: dict,
                                      n_tasks=200,
                                      repeats=5) -> pd.DataFrame:
    """
    Tests PAES and baselines across three arrival distributions:

      uniform   — tasks submitted all at once (original paper setup)
      poisson   — inter-arrival times exponentially distributed (λ=1/mean_lat)
      bursty    — alternating quiet periods and high-rate bursts

    Addresses critique: results were tested only on uniform arrival.
    If PAES's advantage holds across distributions, generalizability
    is strengthened.
    """
    print(f"\n{'='*64}")
    print(f"Experiment 6 — Arrival Distribution Sensitivity (NEW)")
    print(f"  Tests: uniform, Poisson, bursty")
    print(f"  Addresses generalizability critique")
    print(f"{'='*64}")

    mean_lat_ms = np.mean(list(LATENCY_PRIORS.values()))  # ~86ms

    def build_uniform(n):
        """All tasks arrive at t=0 (original paper setup)."""
        t_now = time.perf_counter()
        tasks = build_task_batch(models, n)
        for t in tasks:
            t.arrival_time = t_now
        return tasks

    def build_poisson(n):
        """Inter-arrival times ~ Exp(λ) where λ is fast enough to build a queue.
        Rate = 5 tasks/sec so tasks arrive faster than they can be processed."""
        t_now = time.perf_counter()
        names = list(models.keys())
        tasks = []
        t = t_now
        lam = 5.0  # tasks per second — faster than inference, builds queue
        for _ in range(n):
            name = random.choice(names)
            gap  = random.expovariate(lam)
            t   += gap
            m    = models[name]
            tasks.append(Task(
                model_name=name, priority=m.priority,
                expected_latency_ms=_estimate_latency(m),
                expected_energy_mj=_estimate_energy(m),
                deadline_ms=m.deadline_ms, run_fn=m.run,
                input_data=m.make_dummy_input(),
                arrival_time=t,
            ))
        return tasks

    def build_bursty(n):
        """Alternating quiet (5 tasks) and burst (30 tasks) windows."""
        t_now  = time.perf_counter()
        names  = list(models.keys())
        tasks  = []
        t      = t_now
        placed = 0
        while placed < n:
            # Quiet window
            quiet_n = min(5, n - placed)
            for _ in range(quiet_n):
                name = random.choice(names)
                m    = models[name]
                tasks.append(Task(
                    model_name=name, priority=m.priority,
                    expected_latency_ms=_estimate_latency(m),
                    expected_energy_mj=_estimate_energy(m),
                    deadline_ms=m.deadline_ms, run_fn=m.run,
                    input_data=m.make_dummy_input(),
                    arrival_time=t,
                ))
                t += random.uniform(0.05, 0.15)  # 50–150ms gaps
            placed += quiet_n
            if placed >= n:
                break
            # Burst window — 30 tasks arrive rapidly
            burst_n = min(30, n - placed)
            for i in range(burst_n):
                name = random.choice(names)
                m    = models[name]
                tasks.append(Task(
                    model_name=name, priority=m.priority,
                    expected_latency_ms=_estimate_latency(m),
                    expected_energy_mj=_estimate_energy(m),
                    deadline_ms=m.deadline_ms, run_fn=m.run,
                    input_data=m.make_dummy_input(),
                    arrival_time=t + i * 0.002,  # 2ms gaps in burst
                ))
            t += burst_n * 0.002 + 0.5  # quiet after burst
            placed += burst_n
        return tasks

    distributions = {
        "uniform": build_uniform,
        "poisson": build_poisson,
        "bursty":  build_bursty,
    }

    rows = []
    for dist_name, builder in distributions.items():
        for mode in SCHEDULERS:
            avg_waits, miss_rates, pw_waits = [], [], []
            for r in range(repeats):
                random.seed(r * 17)
                tasks = builder(n_tasks)
                stats, _ = run_scheduler_on_tasks(mode, tasks)
                avg_waits.append(stats["avg_wait_ms"])
                miss_rates.append(stats["miss_rate"])
                pw_waits.append(stats["priority_weighted_avg_wait_ms"])
            rows.append({
                "distribution": dist_name,
                "scheduler":    mode,
                "label":        SCHEDULER_LABELS[mode],
                "avg_wait_ms":  round(float(np.mean(avg_waits)),  2),
                "wait_std":     round(float(np.std(avg_waits)),   2),
                "pw_wait_ms":   round(float(np.mean(pw_waits)),   2),
                "miss_rate":    round(float(np.mean(miss_rates)), 4),
                "miss_std":     round(float(np.std(miss_rates)),  4),
            })
        print(f"  Distribution '{dist_name}' done")

    df = pd.DataFrame(rows)

    # Print comparison
    for dist_name in distributions:
        sub = df[df["distribution"] == dist_name].set_index("scheduler")
        fifo_w = sub.loc["fifo", "avg_wait_ms"]
        paes_w = sub.loc["paes", "avg_wait_ms"]
        sjf_w  = sub.loc["estimated_sjf", "avg_wait_ms"]
        red    = (fifo_w - paes_w) / max(fifo_w, 1e-6) * 100
        sjf_red = (fifo_w - sjf_w) / max(fifo_w, 1e-6) * 100
        print(f"\n  [{dist_name}] PAES vs FIFO: {red:+.1f}%  "
              f"| Est-SJF vs FIFO: {sjf_red:+.1f}%  "
              f"| PAES miss: {sub.loc['paes','miss_rate']:.4f}")

    return df
