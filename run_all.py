"""
run_all.py 

Usage:
    python run_all.py              # full run (~15-20 min)
    python run_all.py --quick      # fast run for testing (~2-3 min)
    python run_all.py --exp 1 2    # run only experiments 1 and 2

Results saved to:
    results/   ← CSV files for each experiment
    figures/   ← PNG figures ready for the paper
"""

import argparse
import json
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
random.seed(42)
np.random.seed(42)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="PAES Experiment Runner")
parser.add_argument("--quick",  action="store_true",
                    help="Run with reduced task counts for a quick test")
parser.add_argument("--exp",    nargs="+", type=int, default=[1,2,3,4,5],
                    help="Which experiments to run (default: all)")
parser.add_argument("--models", nargs="+",
                    default=["mobilenet_v2","yolov5n","whisper_tiny",
                             "distilbert_sentiment","midas_small"],
                    help="Which models to include")
args = parser.parse_args()

QUICK = args.quick
EXPS  = args.exp

# Scale task counts
SCALE = 0.25 if QUICK else 1.0
N_MAIN    = 600
N_REPEATS = 5

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("\n" + "█"*60)
print("  PAES — Priority-Aware Edge Scheduler")
print("  Full Experimental Evaluation")
print("█"*60)
if QUICK:
    print("  [QUICK MODE] Running with reduced task counts")
print(f"  Experiments: {EXPS}")
print(f"  Models:      {args.models}")
print(f"  N tasks:     {N_MAIN} (main experiments)")
print()

# ── Load models ───────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent / "models"))
from model_zoo import load_models
models = load_models(selection=args.models if args.models else None)

if not models:
    print("ERROR: No models loaded.")
    sys.exit(1)

# ── Import experiment functions ───────────────────────────────────────────────

from experiments import (experiment_1_latency, experiment_2_deadline,
                         experiment_3_energy,  experiment_4_burst,
                         experiment_5_sensitivity)
from figures import (fig1_latency, fig2_deadline, fig3_energy,
                     fig4_burst, fig5_sensitivity, fig6_per_model,
                     summary_table)
from scheduler import Scheduler, Task

wall_start = time.time()
results_store = {}

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("RUNNING EXPERIMENTS")
print("─"*60)

# ── Experiment 1 ──────────────────────────────────────────────────────────────
if 1 in EXPS:
    df1 = experiment_1_latency(models, n_tasks=N_MAIN)
    df1.to_csv(RESULTS_DIR / "exp1_latency.csv")
    results_store["latency"] = df1
    print("  ✓ Exp 1 saved → results/exp1_latency.csv")

# ── Experiment 2 ──────────────────────────────────────────────────────────────
if 2 in EXPS:
    df2 = experiment_2_deadline(models, repeats=N_REPEATS)
    df2.to_csv(RESULTS_DIR / "exp2_deadline.csv", index=False)
    results_store["deadline"] = df2
    print("  ✓ Exp 2 saved → results/exp2_deadline.csv")

# ── Experiment 3 ──────────────────────────────────────────────────────────────
if 3 in EXPS:
    df3 = experiment_3_energy(models, n_tasks=N_MAIN)
    df3.to_csv(RESULTS_DIR / "exp3_energy.csv")
    results_store["energy"] = df3
    print("  ✓ Exp 3 saved → results/exp3_energy.csv")

# ── Experiment 4 ──────────────────────────────────────────────────────────────
if 4 in EXPS:
    df4 = experiment_4_burst(models)
    df4.to_csv(RESULTS_DIR / "exp4_burst.csv")
    results_store["burst"] = df4
    print("  ✓ Exp 4 saved → results/exp4_burst.csv")

# ── Experiment 5 ──────────────────────────────────────────────────────────────
if 5 in EXPS:
    df5 = experiment_5_sensitivity(models, n_tasks=int(150 * SCALE))
    df5.to_csv(RESULTS_DIR / "exp5_sensitivity.csv")
    results_store["sensitivity"] = df5
    print("  ✓ Exp 5 saved → results/exp5_sensitivity.csv")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("GENERATING FIGURES")
print("─"*60)

if "latency" in results_store:
    fig1_latency(results_store["latency"])

if "deadline" in results_store:
    fig2_deadline(results_store["deadline"])

if "energy" in results_store:
    fig3_energy(results_store["energy"])

if "burst" in results_store:
    fig4_burst(results_store["burst"])

if "sensitivity" in results_store:
    fig5_sensitivity(results_store["sensitivity"])

# Per-model breakdown (uses Exp 1's raw data via fresh run on small n)
print("\n  Generating per-model heatmap...")
per_model_data = {}
from experiments import SCHEDULERS, build_task_batch
from scheduler import Task as _Task
import time as _time

small_tasks = build_task_batch(models, 60)
for mode in SCHEDULERS:
    s = Scheduler(mode=mode)
    for t in small_tasks:
        tx = _Task(
            model_name=t.model_name, priority=t.priority,
            expected_latency_ms=t.expected_latency_ms,
            expected_energy_mj=t.expected_energy_mj,
            deadline_ms=t.deadline_ms, run_fn=t.run_fn,
            input_data=t.input_data, arrival_time=_time.perf_counter()
        )
        s.submit(tx)
    s.run_all()
    per_model_data[mode] = s.per_model_stats()

fig6_per_model(per_model_data)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*60)
print("SUMMARY TABLE")
print("─"*60)

if "latency" in results_store and "energy" in results_store and "deadline" in results_store:
    summary_table(results_store["latency"],
                  results_store["energy"],
                  results_store["deadline"])

# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - wall_start
print(f"\n{'█'*60}")
print(f"  All done in {elapsed/60:.1f} minutes")
print(f"  Figures → figures/")
print(f"  Raw CSVs → results/")
print(f"{'█'*60}\n")
