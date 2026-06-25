"""
run_all_experiments_v2.py
=========================
Runs all PAES v2 experiments and saves results + Wilcoxon stats.

Usage:
  python run_all_experiments_v2.py              # full run (real inference)
  python run_all_experiments_v2.py --quick      # fast validation (no sleep)
  python run_all_experiments_v2.py --exp 1 2 6  # specific experiments

The --quick flag patches model_zoo to skip time.sleep() so the
full experiment suite runs in ~2 mins on simulator. Use this for
code validation; use full run with real inference for publication.
"""

import argparse
import os
import sys
import random
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--quick",   action="store_true")
parser.add_argument("--exp",     nargs="+", type=int, default=[1,2,3,4,5,6])
parser.add_argument("--repeats", type=int,  default=10)
parser.add_argument("--tasks",   type=int,  default=300)
parser.add_argument("--device",  type=str,  default="i7-1165G7")
args = parser.parse_args()

# ── Fast mode: patch simulators to use 1ms fixed sleep ───────────────────────
if args.quick:
    print("⚡ Quick mode — using 1ms fixed latency (preserves relative timing)")
    import models.model_zoo as mz
    _orig_gauss = __import__('random').gauss
    # Replace gauss with fixed means so latency ratios are preserved
    import random as _rnd
    _orig_gauss = _rnd.gauss
    _rnd.gauss = lambda mu, sigma: mu   # deterministic, no variance
    # Scale sleep down 100x so 80ms -> 0.8ms, 150ms -> 1.5ms
    _orig_sleep = __import__('time').sleep
    import time as _time
    _time.sleep = lambda x: _orig_sleep(x / 100.0)
    n_tasks = args.tasks if args.tasks != 300 else 200
    repeats = 3
else:
    n_tasks = args.tasks
    repeats = args.repeats

random.seed(42)
np.random.seed(42)

from models.model_zoo import load_models
from experiments import (
    experiment_1_latency,
    experiment_2_deadline,
    experiment_3_energy,
    experiment_4_burst,
    experiment_5_sensitivity,
    experiment_6_arrival_sensitivity,
    SCHEDULERS, SCHEDULER_LABELS,
    wilcoxon_vs_paes,
)

out_dir = Path("results") / args.device
fig_dir = Path("figures") / args.device
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*64}")
print(f"  PAES v2 — All Experiments")
print(f"  Device : {args.device}")
print(f"  Tasks  : {n_tasks} | Repeats: {repeats}")
print(f"  Mode   : {'quick (no sleep)' if args.quick else 'full (real inference)'}")
print(f"  Schedulers ({len(SCHEDULERS)}): {[SCHEDULER_LABELS[s] for s in SCHEDULERS]}")
print(f"{'='*64}")

print("\nLoading models...")
models = load_models()

summary = {
    "device": args.device,
    "n_tasks": n_tasks,
    "repeats": repeats,
    "quick": args.quick,
    "schedulers": SCHEDULERS,
    "experiments": {}
}

# ══════════════════════════════════════════════════════════════════════════════
if 1 in args.exp:
    df1, wilcox1 = experiment_1_latency(models, n_tasks=n_tasks,
                                         repeats=repeats)
    df1.to_csv(out_dir / "exp1_latency_v2.csv")

    # Save Wilcoxon results
    wilcox_rows = []
    for mode, res in wilcox1["queue_wait"].items():
        wilcox_rows.append({
            "metric": "avg_wait_ms",
            "baseline": mode,
            "baseline_label": SCHEDULER_LABELS.get(mode, mode),
            **res
        })
    pd.DataFrame(wilcox_rows).to_csv(
        out_dir / "exp1_wilcoxon_v2.csv", index=False)

    summary["experiments"]["exp1"] = {
        "paes_avg_wait":   float(df1.loc["paes","avg_wait_ms"]),
        "fifo_avg_wait":   float(df1.loc["fifo","avg_wait_ms"]),
        "sjf_avg_wait":    float(df1.loc["estimated_sjf","avg_wait_ms"]),
        "paes_pw_wait":    float(df1.loc["paes","priority_weighted_avg_wait_ms"]),
        "fifo_pw_wait":    float(df1.loc["fifo","priority_weighted_avg_wait_ms"]),
        "paes_miss":       float(df1.loc["paes","miss_rate"]),
        "paes_vs_fifo_pct": round(
            (df1.loc["fifo","avg_wait_ms"] - df1.loc["paes","avg_wait_ms"])
            / df1.loc["fifo","avg_wait_ms"] * 100, 2),
        "paes_vs_sjf_pct": round(
            (df1.loc["estimated_sjf","avg_wait_ms"] - df1.loc["paes","avg_wait_ms"])
            / df1.loc["estimated_sjf","avg_wait_ms"] * 100, 2),
    }
    print(f"\n  ✓ Exp 1 saved → {out_dir}/exp1_latency_v2.csv")
    print(f"              → {out_dir}/exp1_wilcoxon_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
if 2 in args.exp:
    df2 = experiment_2_deadline(models, repeats=repeats)
    df2.to_csv(out_dir / "exp2_deadline_v2.csv", index=False)

    # Extract key stats
    high_load = df2[df2["load_level"]=="high"].set_index("scheduler")
    low_load  = df2[df2["load_level"]=="low"].set_index("scheduler")
    summary["experiments"]["exp2"] = {
        "paes_high_miss":  float(high_load.loc["paes","miss_rate"]),
        "fifo_high_miss":  float(high_load.loc["fifo","miss_rate"]),
        "paes_low_miss":   float(low_load.loc["paes","miss_rate"]),
        "fifo_low_miss":   float(low_load.loc["fifo","miss_rate"]),
    }
    print(f"\n  ✓ Exp 2 saved → {out_dir}/exp2_deadline_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
if 3 in args.exp:
    df3 = experiment_3_energy(models, n_tasks=n_tasks)
    df3.to_csv(out_dir / "exp3_energy_v2.csv")
    summary["experiments"]["exp3"] = {
        "paes_relative_energy": float(df3.loc["paes","relative_energy"]),
        "sjf_relative_energy":  float(df3.loc["estimated_sjf","relative_energy"]),
    }
    print(f"\n  ✓ Exp 3 saved → {out_dir}/exp3_energy_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
if 4 in args.exp:
    df4 = experiment_4_burst(models)
    df4.to_csv(out_dir / "exp4_burst_v2.csv")
    summary["experiments"]["exp4"] = {
        "paes_burst_miss": float(df4.loc["paes","burst_miss_rate"]),
        "qos_burst_miss":  float(df4.loc["qos","burst_miss_rate"]),
        "sjf_burst_miss":  float(df4.loc["estimated_sjf","burst_miss_rate"]),
    }
    print(f"\n  ✓ Exp 4 saved → {out_dir}/exp4_burst_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
if 5 in args.exp:
    df5 = experiment_5_sensitivity(models, n_tasks=n_tasks)
    df5.to_csv(out_dir / "exp5_sensitivity_v2.csv")
    summary["experiments"]["exp5"] = {
        "no_latency_avg_lat": float(df5.loc["no-latency","avg_latency_ms"]),
        "balanced_avg_lat":   float(df5.loc["balanced","avg_latency_ms"]),
        "no_energy_energy":   float(df5.loc["no-energy","avg_energy_mj"]),
        "balanced_energy":    float(df5.loc["balanced","avg_energy_mj"]),
    }
    print(f"\n  ✓ Exp 5 saved → {out_dir}/exp5_sensitivity_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
if 6 in args.exp:
    df6 = experiment_6_arrival_sensitivity(
        models, n_tasks=n_tasks, repeats=repeats)
    df6.to_csv(out_dir / "exp6_arrival_sensitivity_v2.csv", index=False)

    dist_summary = {}
    for dist in ["uniform","poisson","bursty"]:
        sub = df6[df6["distribution"]==dist].set_index("scheduler")
        fifo_w = sub.loc["fifo","avg_wait_ms"]
        paes_w = sub.loc["paes","avg_wait_ms"]
        sjf_w  = sub.loc["estimated_sjf","avg_wait_ms"]
        if fifo_w > 1.0:  # only report if FIFO wait is meaningful
            paes_red = (fifo_w - paes_w) / fifo_w * 100
            sjf_red  = (fifo_w - sjf_w)  / fifo_w * 100
        else:
            paes_red = sjf_red = float("nan")
        dist_summary[dist] = {
            "paes_vs_fifo_pct": round(paes_red, 2) if not np.isnan(paes_red) else None,
            "sjf_vs_fifo_pct":  round(sjf_red,  2) if not np.isnan(sjf_red)  else None,
            "paes_miss":        float(sub.loc["paes","miss_rate"]),
        }
    summary["experiments"]["exp6"] = dist_summary
    print(f"\n  ✓ Exp 6 saved → {out_dir}/exp6_arrival_sensitivity_v2.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Save summary JSON
with open(out_dir / "summary_v2.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*64}")
print(f"  All experiments complete.")
print(f"  Results: {out_dir}/")
print(f"\n  Key findings:")
if "exp1" in summary["experiments"]:
    e = summary["experiments"]["exp1"]
    print(f"    PAES vs FIFO (queue wait):    {e['paes_vs_fifo_pct']:+.1f}%")
    print(f"    PAES vs Est-SJF (queue wait): {e['paes_vs_sjf_pct']:+.1f}%")
    print(f"    PAES PW-wait:  {e['paes_pw_wait']:.0f}ms  "
          f"(FIFO: {e['fifo_pw_wait']:.0f}ms)")
if "exp6" in summary["experiments"]:
    print(f"    Arrival sensitivity:")
    for dist, d in summary["experiments"]["exp6"].items():
        paes_s = f"{d['paes_vs_fifo_pct']:+.1f}%" if d['paes_vs_fifo_pct'] is not None else "n/a (low FIFO wait)"
        sjf_s  = f"{d['sjf_vs_fifo_pct']:+.1f}%"  if d['sjf_vs_fifo_pct']  is not None else "n/a"
        print(f"      {dist:<10} PAES {paes_s} vs FIFO  | EstSJF {sjf_s} vs FIFO")
print(f"{'='*64}\n")
