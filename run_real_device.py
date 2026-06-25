"""
run_real_device.py
==================
Runs all PAES v2 experiments on the current physical device using real
model inference. No simulation, no speed scaling, no Gaussian fallbacks
unless a library is genuinely not installed.

New in v2 vs original paper:
  [1] deadline_missed uses total_response_ms (queue_wait + inference)
      vs deadline — not inference alone
  [2] priority_weighted_avg_wait_ms headline metric — YOLOv5 (pri=3.0)
      counts 3x more than MiDaS (pri=1.0)
  [3] PAES deadline proximity bonus — urgency spike within theta=150ms
  [4] per_model_stats() with avg_wait_ms per model (PM vs servant breakdown)

Usage:
  python run_real_device.py                  # all experiments
  python run_real_device.py --exp 1 2        # specific experiments
  python run_real_device.py --quick          # reduced task counts
  python run_real_device.py --repeats 10     # more repeats for publication

Requirements:
  pip install numpy pandas matplotlib tqdm scipy
  pip install torch torchvision              # MobileNetV2, YOLOv5n, MiDaS
  pip install ultralytics                    # YOLOv5n
  pip install openai-whisper                 # Whisper Tiny
  pip install transformers                   # DistilBERT

Output:
  results/<device_name>/exp1_latency.csv
  results/<device_name>/exp2_deadline.csv
  results/<device_name>/exp3_energy.csv
  results/<device_name>/exp4_burst.csv
  results/<device_name>/exp5_sensitivity.csv
  results/<device_name>/exp_workload_realism.csv
  results/<device_name>/exp_per_model_wait.csv   ← new in v2
  results/<device_name>/summary_v2.json
  figures/<device_name>/                          ← all plots
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scheduler import Scheduler, Task, VALID_MODES
from models.model_zoo import load_models

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

# ── Device identification ─────────────────────────────────────────────────────

def get_device_name():
    node = platform.node().replace(" ", "_").replace("/", "_")[:20]
    cpu  = platform.processor() or platform.machine()
    safe = "".join(c for c in cpu if c.isalnum() or c in "-_")[:30]
    return f"{node}_{safe}" if safe else node

DEVICE_NAME = get_device_name()

SCHEDULERS = list(VALID_MODES)

# ── Task factory ──────────────────────────────────────────────────────────────

LATENCY_PRIORS = {
    "mobilenet_v2":          35.0,
    "yolov5n":               80.0,
    "whisper_tiny":         150.0,
    "distilbert_sentiment":  55.0,
    "midas_small":          110.0,
}
ENERGY_PRIORS = {
    "mobilenet_v2":         446.0,
    "yolov5n":             1020.0,
    "whisper_tiny":        1912.0,
    "distilbert_sentiment": 701.0,
    "midas_small":         1402.0,
}

def make_task(name, model) -> Task:
    return Task(
        model_name          = name,
        priority            = model.priority,
        expected_latency_ms = LATENCY_PRIORS.get(name, 100.0),
        expected_energy_mj  = ENERGY_PRIORS.get(name, 1000.0),
        deadline_ms         = model.deadline_ms,
        run_fn              = model.run,
        input_data          = model.make_dummy_input(),
        arrival_time        = time.perf_counter(),
    )

def build_batch(models, n, seed=None) -> list[Task]:
    if seed is not None:
        random.seed(seed)
    names  = list(models.keys())
    chosen = random.choices(names, k=n)
    return [make_task(name, models[name]) for name in chosen]

def run_sched(mode, tasks, alpha=1.0, beta=1.0, gamma=1.0):
    s = Scheduler(mode, alpha=alpha, beta=beta, gamma=gamma)
    for t in tasks:
        s.submit(t)
    s.run_all()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Latency and Queue Wait (600 tasks)
# ══════════════════════════════════════════════════════════════════════════════

def exp1_latency(models, n_tasks, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 1 — Latency & Queue Wait ({n_tasks} tasks, {repeats} repeats)")
    print(f"{'='*64}")

    rows = []
    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        avg_waits, pw_waits, latencies, miss_rates = [], [], [], []
        for r in range(repeats):
            tasks = build_batch(models, n_tasks, seed=r * 77)
            s = run_sched(mode, tasks)
            st = s.stats()
            avg_waits.append(st["avg_wait_ms"])
            pw_waits.append(st["priority_weighted_avg_wait_ms"])
            latencies.append(st["avg_latency_ms"])
            miss_rates.append(st["miss_rate"])
        rows.append({
            "scheduler":                     mode,
            "n_tasks":                       n_tasks,
            "avg_latency_ms":                round(float(np.mean(latencies)),  2),
            "avg_latency_std":               round(float(np.std(latencies)),   2),
            "avg_wait_ms":                   round(float(np.mean(avg_waits)),  2),
            "avg_wait_std":                  round(float(np.std(avg_waits)),   2),
            "priority_weighted_avg_wait_ms": round(float(np.mean(pw_waits)),   2),
            "pw_wait_std":                   round(float(np.std(pw_waits)),    2),
            "miss_rate":                     round(float(np.mean(miss_rates)), 4),
            "miss_rate_std":                 round(float(np.std(miss_rates)),  4),
        })

    df = pd.DataFrame(rows).set_index("scheduler")
    df.to_csv(out_dir / "exp1_latency.csv")
    print(f"\n  {'Scheduler':<18} {'AvgWait':>12} {'PW-Wait':>12} {'AvgLat':>10} {'Miss':>8}")
    print("  " + "-"*64)
    for mode in SCHEDULERS:
        mk = " ◀" if mode == "paes" else ""
        r  = df.loc[mode]
        print(f"  {mode:<18} {r['avg_wait_ms']:>12.1f} "
              f"{r['priority_weighted_avg_wait_ms']:>12.1f} "
              f"{r['avg_latency_ms']:>10.1f} {r['miss_rate']:>8.4f}{mk}")
    print(f"\n  Saved → {out_dir}/exp1_latency.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Miss Rate vs Load Level
# ══════════════════════════════════════════════════════════════════════════════

def exp2_deadline(models, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 2 — Miss Rate vs Load ({repeats} repeats per level)")
    print(f"{'='*64}")

    load_levels = {"low": 30, "medium": 80, "high": 160, "extreme": 300}
    rows = []

    for load_name, n in load_levels.items():
        for mode in SCHEDULERS:
            miss_rates = []
            for r in range(repeats):
                tasks = build_batch(models, n, seed=r * 13)
                s = run_sched(mode, tasks)
                miss_rates.append(s.stats()["miss_rate"])
            rows.append({
                "load_level":    load_name,
                "n_tasks":       n,
                "scheduler":     mode,
                "miss_rate":     round(float(np.mean(miss_rates)), 4),
                "miss_rate_std": round(float(np.std(miss_rates)),  4),
            })
        print(f"  Load '{load_name}' ({n} tasks) done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp2_deadline.csv", index=False)
    pivot = df.pivot(index="load_level", columns="scheduler", values="miss_rate")
    pivot = pivot.reindex(["low", "medium", "high", "extreme"])
    print("\n  Miss Rate by Load:")
    print(pivot.to_string())
    print(f"\n  Saved → {out_dir}/exp2_deadline.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Energy per Task
# ══════════════════════════════════════════════════════════════════════════════

def exp3_energy(models, n_tasks, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 3 — Energy per Task ({n_tasks} tasks)")
    print(f"{'='*64}")

    rows = []
    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        energies = []
        for r in range(repeats):
            tasks = build_batch(models, n_tasks, seed=r * 55)
            s = run_sched(mode, tasks)
            energies.append(s.stats()["avg_energy_mj"])
        rows.append({
            "scheduler":     mode,
            "avg_energy_mj": round(float(np.mean(energies)), 4),
            "energy_std":    round(float(np.std(energies)),  4),
        })

    df = pd.DataFrame(rows).set_index("scheduler")
    baseline = df.loc["fifo", "avg_energy_mj"]
    df["relative_energy"] = (df["avg_energy_mj"] / baseline).round(3)
    df.to_csv(out_dir / "exp3_energy.csv")
    print("\n  Energy Results:")
    print(df.to_string())
    print(f"\n  Saved → {out_dir}/exp3_energy.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Burst Recovery
# ══════════════════════════════════════════════════════════════════════════════

def exp4_burst(models, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 4 — Burst Workload Recovery")
    print(f"{'='*64}")

    phases = [("pre_burst", 40), ("burst", 160), ("post_burst", 40)]
    rows = []

    for mode in SCHEDULERS:
        phase_results = []
        for phase_name, n in phases:
            mrs, p95s = [], []
            for r in range(repeats):
                tasks = build_batch(models, n, seed=r * 9)
                s = run_sched(mode, tasks)
                st = s.stats()
                mrs.append(st["miss_rate"])
                p95s.append(st["p95_latency_ms"])
            phase_results.append({
                "phase":    phase_name,
                "miss":     float(np.mean(mrs)),
                "p95":      float(np.mean(p95s)),
            })

        pre_p95   = phase_results[0]["p95"]
        burst_p95 = phase_results[1]["p95"]
        post_p95  = phase_results[2]["p95"]
        recovery  = (post_p95 - pre_p95) / max(pre_p95, 1e-6)

        rows.append({
            "scheduler":            mode,
            "pre_burst_miss":       round(phase_results[0]["miss"],  4),
            "burst_miss":           round(phase_results[1]["miss"],  4),
            "post_burst_miss":      round(phase_results[2]["miss"],  4),
            "pre_p95_ms":           round(pre_p95,                   2),
            "burst_p95_ms":         round(burst_p95,                 2),
            "post_p95_ms":          round(post_p95,                  2),
            "recovery_overshoot":   round(recovery,                  4),
        })
        print(f"  {mode:<18} burst_miss={phase_results[1]['miss']:.1%}  "
              f"recovery_overshoot={recovery:.1%}")

    df = pd.DataFrame(rows).set_index("scheduler")
    df.to_csv(out_dir / "exp4_burst.csv")
    print(f"\n  Saved → {out_dir}/exp4_burst.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 5 — PAES Sensitivity (α/β/γ sweep)
# ══════════════════════════════════════════════════════════════════════════════

def exp5_sensitivity(models, n_tasks, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 5 — PAES Sensitivity Analysis")
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

    rows = []
    for label, a, b, g in tqdm(configs, desc="Configs"):
        lats, energies, miss_rates, pw_waits = [], [], [], []
        for r in range(repeats):
            tasks = build_batch(models, n_tasks, seed=r * 31)
            s = run_sched("paes", tasks, alpha=a, beta=b, gamma=g)
            st = s.stats()
            lats.append(st["avg_latency_ms"])
            energies.append(st["avg_energy_mj"])
            miss_rates.append(st["miss_rate"])
            pw_waits.append(st["priority_weighted_avg_wait_ms"])
        rows.append({
            "config":                        label,
            "alpha": a, "beta": b, "gamma": g,
            "avg_latency_ms":                round(float(np.mean(lats)),       2),
            "miss_rate":                     round(float(np.mean(miss_rates)), 4),
            "avg_energy_mj":                 round(float(np.mean(energies)),   4),
            "priority_weighted_avg_wait_ms": round(float(np.mean(pw_waits)),   2),
        })

    df = pd.DataFrame(rows).set_index("config")
    df.to_csv(out_dir / "exp5_sensitivity.csv")
    print("\n  Sensitivity Results:")
    print(df[["avg_latency_ms", "miss_rate",
              "avg_energy_mj", "priority_weighted_avg_wait_ms"]].to_string())
    print(f"\n  Saved → {out_dir}/exp5_sensitivity.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 6 (NEW) — Per-Model Wait Breakdown
# ══════════════════════════════════════════════════════════════════════════════

def exp6_per_model_wait(models, n_tasks, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Exp 6 (v2 new) — Per-Model Wait Breakdown ({n_tasks} tasks)")
    print(f"  Reveals PM vs servant dynamic hidden by unweighted average")
    print(f"{'='*64}")

    all_per_model = {}

    for mode in tqdm(SCHEDULERS, desc="Schedulers"):
        model_waits  = {}
        model_misses = {}
        model_totals = {}

        for r in range(repeats):
            tasks = build_batch(models, n_tasks, seed=r * 42)
            s = run_sched(mode, tasks)
            pm = s.per_model_stats()
            for model_name, stats in pm.items():
                if model_name not in model_waits:
                    model_waits[model_name]  = []
                    model_misses[model_name] = []
                    model_totals[model_name] = []
                model_waits[model_name].append(stats["avg_wait_ms"])
                model_misses[model_name].append(stats["miss_rate"])
                model_totals[model_name].append(stats["avg_total_response_ms"])

        all_per_model[mode] = {
            m: {
                "avg_wait_ms":           round(float(np.mean(model_waits[m])),  1),
                "avg_wait_std":          round(float(np.std(model_waits[m])),   1),
                "avg_total_response_ms": round(float(np.mean(model_totals[m])), 1),
                "miss_rate":             round(float(np.mean(model_misses[m])), 4),
            }
            for m in model_waits
        }

    # Build flat CSV
    rows = []
    priority_map = {
        "yolov5n": 3.0, "mobilenet_v2": 2.0, "whisper_tiny": 2.0,
        "distilbert_sentiment": 1.5, "midas_small": 1.0,
    }
    for mode in SCHEDULERS:
        for model_name, stats in all_per_model[mode].items():
            rows.append({
                "scheduler":             mode,
                "model":                 model_name,
                "priority":              priority_map.get(model_name, 1.0),
                "avg_wait_ms":           stats["avg_wait_ms"],
                "avg_wait_std":          stats["avg_wait_std"],
                "avg_total_response_ms": stats["avg_total_response_ms"],
                "miss_rate":             stats["miss_rate"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp_per_model_wait.csv", index=False)

    # Print PAES vs FIFO comparison
    print(f"\n  Per-model wait: PAES vs FIFO")
    print(f"  {'Model':<28} {'Pri':>5} {'FIFO Wait':>12} {'PAES Wait':>12} "
          f"{'Diff%':>9} {'PAES Miss':>10}")
    print("  " + "-"*80)

    model_order = ["yolov5n", "mobilenet_v2", "whisper_tiny",
                   "distilbert_sentiment", "midas_small"]
    for model_name in model_order:
        if model_name not in all_per_model.get("fifo", {}):
            continue
        f    = all_per_model["fifo"][model_name]["avg_wait_ms"]
        p    = all_per_model["paes"][model_name]["avg_wait_ms"]
        miss = all_per_model["paes"][model_name]["miss_rate"]
        pri  = priority_map.get(model_name, 1.0)
        pct  = ((p - f) / max(f, 1e-6)) * 100
        tag  = "▼ better" if pct < 0 else "▲ worse"
        print(f"  {model_name:<28} {pri:>5.1f} {f:>12.1f} {p:>12.1f} "
              f"{pct:>+8.1f}% {tag}  {miss:>8.4f}")

    print(f"\n  Saved → {out_dir}/exp_per_model_wait.csv")
    return df, all_per_model


# ══════════════════════════════════════════════════════════════════════════════
# Robot Pipeline — realistic 30s workload
# ══════════════════════════════════════════════════════════════════════════════

def exp_robot_pipeline(models, repeats, out_dir, fig_dir):
    print(f"\n{'='*64}")
    print(f"  Robot Pipeline — 30s realistic workload ({repeats} repeats)")
    print(f"{'='*64}")

    MODEL_PROFILES = {
        name: (model.priority, LATENCY_PRIORS[name],
               ENERGY_PRIORS[name], model.deadline_ms, model.run)
        for name, model in models.items()
    }

    def generate_pipeline(duration=30.0):
        tasks = []
        # Camera → YOLO every ~100ms
        t = 0.0
        while t < duration:
            pri, lat, energy, deadline, run_fn = MODEL_PROFILES["yolov5n"]
            tasks.append(("yolov5n", pri, lat, energy, deadline, run_fn, t))
            if random.random() < 0.15:
                for i in range(random.randint(2, 4)):
                    bt = t + (i+1) * random.uniform(0.01, 0.03)
                    if bt < duration:
                        tasks.append(("yolov5n", pri, lat, energy, deadline, run_fn, bt))
            t += max(random.gauss(0.10, 0.03), 0.02)
        # Mic → Whisper Poisson
        t = random.expovariate(1/3.0)
        while t < duration:
            pri, lat, energy, deadline, run_fn = MODEL_PROFILES["whisper_tiny"]
            tasks.append(("whisper_tiny", pri, lat, energy, deadline, run_fn, t))
            t += random.expovariate(1/3.0)
        # Planner → DistilBERT every ~2s
        t = 0.5
        while t < duration:
            pri, lat, energy, deadline, run_fn = MODEL_PROFILES["distilbert_sentiment"]
            tasks.append(("distilbert_sentiment", pri, lat, energy, deadline, run_fn, t))
            t += max(random.gauss(2.0, 0.3), 0.5)
        # Depth → MiDaS every ~500ms
        t = 0.2
        while t < duration:
            pri, lat, energy, deadline, run_fn = MODEL_PROFILES["midas_small"]
            tasks.append(("midas_small", pri, lat, energy, deadline, run_fn, t))
            t += max(random.gauss(0.5, 0.08), 0.1)
        # Classifier → MobileNet every ~200ms
        t = 0.05
        while t < duration:
            pri, lat, energy, deadline, run_fn = MODEL_PROFILES["mobilenet_v2"]
            tasks.append(("mobilenet_v2", pri, lat, energy, deadline, run_fn, t))
            t += max(random.gauss(0.2, 0.04), 0.05)
        tasks.sort(key=lambda x: x[6])
        return tasks

    rows = []
    for mode in SCHEDULERS:
        waits, pw_waits, miss_rates = [], [], []
        for r in range(repeats):
            random.seed(r * 7)
            pipeline_tasks = generate_pipeline()
            base_t = time.perf_counter()
            task_objs = []
            for name, pri, lat, energy, deadline, run_fn, arrival_offset in pipeline_tasks:
                task_objs.append(Task(
                    model_name=name, priority=pri,
                    expected_latency_ms=lat, expected_energy_mj=energy,
                    deadline_ms=deadline, run_fn=run_fn,
                    arrival_time=base_t + arrival_offset,
                ))
            s = run_sched(mode, task_objs)
            st = s.stats()
            waits.append(st["avg_wait_ms"])
            pw_waits.append(st["priority_weighted_avg_wait_ms"])
            miss_rates.append(st["miss_rate"])

        rows.append({
            "scheduler":                     mode,
            "n_tasks":                       len(pipeline_tasks),
            "avg_wait_ms":                   round(float(np.mean(waits)),      2),
            "avg_wait_std":                  round(float(np.std(waits)),       2),
            "priority_weighted_avg_wait_ms": round(float(np.mean(pw_waits)),   2),
            "pw_wait_std":                   round(float(np.std(pw_waits)),    2),
            "miss_rate":                     round(float(np.mean(miss_rates)), 4),
        })
        print(f"  {mode:<18} wait={rows[-1]['avg_wait_ms']:.0f}ms  "
              f"pw_wait={rows[-1]['priority_weighted_avg_wait_ms']:.0f}ms  "
              f"miss={rows[-1]['miss_rate']:.1%}")

    df = pd.DataFrame(rows).set_index("scheduler")
    df.to_csv(out_dir / "exp_workload_realism.csv")

    fifo_w = df.loc["fifo", "avg_wait_ms"]
    paes_w = df.loc["paes", "avg_wait_ms"]
    print(f"\n  PAES vs FIFO: {((fifo_w-paes_w)/max(fifo_w,1e-6)*100):+.1f}% queue wait")
    print(f"  Saved → {out_dir}/exp_workload_realism.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     nargs="+", type=int, default=[1,2,3,4,5,6,7],
                        help="Experiments to run (1-6, 7=robot pipeline)")
    parser.add_argument("--quick",   action="store_true",
                        help="Reduced task counts for fast validation")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device",  type=str, default=DEVICE_NAME,
                        help="Override device name for output folder")
    args = parser.parse_args()

    n_tasks = 200 if args.quick else 600
    repeats = 2   if args.quick else args.repeats

    out_dir = Path("results") / args.device
    fig_dir = Path("figures") / args.device
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  PAES v2 — Real Device Experiments")
    print(f"  Device : {args.device}")
    print(f"  Tasks  : {n_tasks} per experiment")
    print(f"  Repeats: {repeats}")
    print(f"  Output : {out_dir}/")
    print(f"{'='*64}")

    print("\nLoading models...")
    models = load_models()

    summary = {"device": args.device, "n_tasks": n_tasks, "repeats": repeats}

    if 1 in args.exp:
        df = exp1_latency(models, n_tasks, repeats, out_dir, fig_dir)
        summary["exp1_paes_avg_wait"]  = float(df.loc["paes", "avg_wait_ms"])
        summary["exp1_fifo_avg_wait"]  = float(df.loc["fifo", "avg_wait_ms"])
        summary["exp1_paes_pw_wait"]   = float(df.loc["paes", "priority_weighted_avg_wait_ms"])
        summary["exp1_fifo_pw_wait"]   = float(df.loc["fifo", "priority_weighted_avg_wait_ms"])

    if 2 in args.exp:
        exp2_deadline(models, repeats, out_dir, fig_dir)

    if 3 in args.exp:
        exp3_energy(models, n_tasks, repeats, out_dir, fig_dir)

    if 4 in args.exp:
        exp4_burst(models, repeats, out_dir, fig_dir)

    if 5 in args.exp:
        exp5_sensitivity(models, n_tasks, repeats, out_dir, fig_dir)

    if 6 in args.exp:
        exp6_per_model_wait(models, n_tasks, repeats, out_dir, fig_dir)

    if 7 in args.exp:
        df = exp_robot_pipeline(models, repeats, out_dir, fig_dir)
        summary["robot_paes_wait"] = float(df.loc["paes", "avg_wait_ms"])
        summary["robot_fifo_wait"] = float(df.loc["fifo", "avg_wait_ms"])

    with open(out_dir / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  All experiments complete.")
    print(f"  Results saved to: {out_dir}/")
    print(f"{'='*64}\n")
