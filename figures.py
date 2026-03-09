"""
figures.py

All figures use a consistent style suitable for IEEE double-column format.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "fifo":            "#e74c3c",
    "round_robin":     "#f39c12",
    "static_priority": "#3498db",
    "edf":             "#9b59b6",
    "pq_deadline":     "#1abc9c",
    "qos":             "#e67e22",
    "paes":            "#2ecc71",
}
LABELS = {
    "fifo":            "FIFO",
    "round_robin":     "Round Robin",
    "static_priority": "Static Priority",
    "edf":             "EDF",
    "pq_deadline":     "PQ+Deadline",
    "qos":             "QoS",
    "paes":            "PAES (ours)",
}
HATCHES = {
    "fifo":            "",
    "round_robin":     "//",
    "static_priority": "..",
    "edf":             "\\\\",
    "pq_deadline":     "xx",
    "qos":             "oo",
    "paes":            "++",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

SCHEDULERS = ["fifo", "round_robin", "static_priority", "edf", "pq_deadline", "qos", "paes"]


# ── Fig 1: Latency Comparison (grouped bar) ───────────────────────────────────

def fig1_latency(df_latency: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [
        ("avg_latency_ms", "Average Latency (ms)",   "ms"),
        ("p95_latency_ms", "P95 Latency (ms)",        "ms"),
        ("throughput_tps", "Throughput (tasks/sec)",  "tasks/s"),
    ]

    for ax, (col, title, unit) in zip(axes, metrics):
        bars = ax.bar(
            [LABELS[s] for s in SCHEDULERS],
            [df_latency.loc[s, col] for s in SCHEDULERS],
            color=[COLORS[s] for s in SCHEDULERS],
            hatch=[HATCHES[s] for s in SCHEDULERS],
            edgecolor="white",
            linewidth=0.8,
            width=0.6,
        )
        # Annotate bar tops
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ax.get_ylim()[1]*0.01,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], rotation=15, ha="right")

    # Highlight PAES improvement
    axes[0].set_title("Average Latency (ms)\n↓ lower is better")
    axes[2].set_title("Throughput (tasks/s)\n↑ higher is better")

    fig.suptitle("Figure 1 — Latency & Throughput Under Multi-Model Load",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUTDIR / "fig1_latency.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Fig 2: Deadline Miss Rate vs Load (line chart) ───────────────────────────

def fig2_deadline(df_deadline: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    load_order = ["low", "medium", "high", "extreme"]
    load_sizes = df_deadline.drop_duplicates("load_level") \
                             .set_index("load_level")["n_tasks"].to_dict()

    x_labels = [f"{l}\n({load_sizes[l]} tasks)" for l in load_order]

    for mode in SCHEDULERS:
        sub = df_deadline[df_deadline["scheduler"] == mode]
        sub = sub.set_index("load_level").reindex(load_order)
        ax.plot(x_labels,
                sub["miss_rate"] * 100,
                marker="o",
                linewidth=2,
                markersize=7,
                color=COLORS[mode],
                label=LABELS[mode])
        # Error band
        if "miss_rate_std" in sub.columns:
            ax.fill_between(
                x_labels,
                (sub["miss_rate"] - sub["miss_rate_std"]) * 100,
                (sub["miss_rate"] + sub["miss_rate_std"]) * 100,
                alpha=0.12,
                color=COLORS[mode],
            )

    ax.set_ylabel("Deadline Miss Rate (%)")
    ax.set_xlabel("Load Level")
    ax.set_title("Figure 2 — Deadline Miss Rate vs Load Level\n"
                 "(lower is better; shaded = ±1 std dev)")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = OUTDIR / "fig2_deadline.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Fig 3: Energy Per Task ────────────────────────────────────────────────────

def fig3_energy(df_energy: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Absolute energy
    bars = axes[0].bar(
        [LABELS[s] for s in SCHEDULERS],
        [df_energy.loc[s, "avg_energy_mj"] for s in SCHEDULERS],
        color=[COLORS[s] for s in SCHEDULERS],
        hatch=[HATCHES[s] for s in SCHEDULERS],
        edgecolor="white",
        width=0.6,
    )
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() * 1.02,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Avg Energy / Task (mJ)\n↓ lower is better")
    axes[0].set_ylabel("mJ")
    axes[0].set_xticklabels([LABELS[s] for s in SCHEDULERS], rotation=15, ha="right")

    # Relative (FIFO = 1.0)
    fifo_e = df_energy.loc["fifo", "avg_energy_mj"]
    rel = [df_energy.loc[s, "avg_energy_mj"] / fifo_e for s in SCHEDULERS]
    bars2 = axes[1].bar(
        [LABELS[s] for s in SCHEDULERS],
        rel,
        color=[COLORS[s] for s in SCHEDULERS],
        hatch=[HATCHES[s] for s in SCHEDULERS],
        edgecolor="white",
        width=0.6,
    )
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="FIFO baseline")
    for bar, v in zip(bars2, rel):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v * 1.02, f"{v:.2f}×",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Relative Energy (FIFO = 1.0×)\n↓ lower is better")
    axes[1].set_ylabel("Relative energy")
    axes[1].set_xticklabels([LABELS[s] for s in SCHEDULERS], rotation=15, ha="right")
    axes[1].legend()

    fig.suptitle("Figure 3 — Energy Consumption Per Task", fontweight="bold")
    plt.tight_layout()
    path = OUTDIR / "fig3_energy.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Fig 4: Burst Recovery ─────────────────────────────────────────────────────

def fig4_burst(df_burst: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    phases = ["pre_burst_miss_rate", "burst_miss_rate", "post_burst_miss_rate"]
    phase_labels = ["Pre-Burst", "Burst", "Post-Burst"]

    x = np.arange(len(phase_labels))
    width = 0.2

    for i, mode in enumerate(SCHEDULERS):
        vals = [df_burst.loc[mode, p] * 100 for p in phases]
        offset = (i - 1.5) * width
        axes[0].bar(x + offset, vals, width,
                    label=LABELS[mode],
                    color=COLORS[mode],
                    hatch=HATCHES[mode],
                    edgecolor="white")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phase_labels)
    axes[0].set_ylabel("Deadline Miss Rate (%)")
    axes[0].set_title("Miss Rate by Phase\n↓ lower is better")
    axes[0].legend(fontsize=9)

    # P95 latency across phases
    phases_lat = ["pre_p95_ms", "burst_p95_ms", "post_p95_ms"]
    for i, mode in enumerate(SCHEDULERS):
        vals = [df_burst.loc[mode, p] for p in phases_lat]
        offset = (i - 1.5) * width
        axes[1].bar(x + offset, vals, width,
                    label=LABELS[mode],
                    color=COLORS[mode],
                    hatch=HATCHES[mode],
                    edgecolor="white")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phase_labels)
    axes[1].set_ylabel("P95 Latency (ms)")
    axes[1].set_title("P95 Latency by Phase\n↓ lower is better")

    fig.suptitle("Figure 4 — Burst Workload Recovery", fontweight="bold")
    plt.tight_layout()
    path = OUTDIR / "fig4_burst.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Fig 5: Sensitivity Analysis ───────────────────────────────────────────────

def fig5_sensitivity(df_sens: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    configs = df_sens.index.tolist()
    x = np.arange(len(configs))
    pal = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))

    for ax, (col, title) in zip(axes, [
        ("avg_latency_ms", "Avg Latency (ms)\n↓ lower is better"),
        ("miss_rate",      "Miss Rate\n↓ lower is better"),
        ("avg_energy_mj",  "Avg Energy/Task (mJ)\n↓ lower is better"),
    ]):
        bars = ax.bar(x, df_sens[col], color=pal, edgecolor="white", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=9)
        ax.set_title(title)
        # Highlight balanced config
        bars[0].set_edgecolor("#e74c3c")
        bars[0].set_linewidth(2.5)

    fig.suptitle("Figure 5 — PAES Sensitivity Analysis (α/β/γ weight sweep)\n"
                 "Red outline = balanced (α=β=γ=1)", fontweight="bold")
    plt.tight_layout()
    path = OUTDIR / "fig5_sensitivity.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Fig 6: Per-model breakdown heatmap ───────────────────────────────────────

def fig6_per_model(per_model_data: dict):
    """
    Heatmap: rows = models, columns = schedulers, values = avg latency
    """
    schedulers = SCHEDULERS
    models_seen = set()
    for mode_data in per_model_data.values():
        models_seen.update(mode_data.keys())
    model_names = sorted(models_seen)

    matrix = np.zeros((len(model_names), len(schedulers)))
    for j, mode in enumerate(schedulers):
        for i, mname in enumerate(model_names):
            matrix[i, j] = per_model_data[mode].get(mname, {}).get("avg_latency_ms", 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(model_names)*0.8 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(schedulers)))
    ax.set_xticklabels([LABELS[s] for s in schedulers], rotation=15, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(schedulers)):
            ax.text(j, i, f"{matrix[i,j]:.0f}ms",
                    ha="center", va="center", fontsize=9,
                    color="black" if matrix[i,j] < matrix.max()*0.7 else "white")

    plt.colorbar(im, ax=ax, label="Avg Latency (ms)")
    ax.set_title("Figure 6 — Per-Model Avg Latency Heatmap\n"
                 "(greener = faster for that model/scheduler combo)")

    plt.tight_layout()
    path = OUTDIR / "fig6_per_model_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(df_latency: pd.DataFrame, df_energy: pd.DataFrame,
                  df_deadline_high: pd.DataFrame):
    """
    Produce a single compact summary table (for paper Table 1).
    """
    rows = []
    for mode in SCHEDULERS:
        miss = df_deadline_high[
            (df_deadline_high["scheduler"] == mode) &
            (df_deadline_high["load_level"] == "high")
        ]["miss_rate"].values

        rows.append({
            "Scheduler":       LABELS[mode],
            "Avg Lat (ms)":    df_latency.loc[mode, "avg_latency_ms"],
            "P95 Lat (ms)":    df_latency.loc[mode, "p95_latency_ms"],
            "Throughput (t/s)":df_latency.loc[mode, "throughput_tps"],
            "Miss Rate @high": f"{float(miss[0])*100:.1f}%" if len(miss) else "—",
            "Rel Energy":      f"{df_energy.loc[mode,'avg_energy_mj'] / df_energy.loc['fifo','avg_energy_mj']:.2f}×",
        })

    df = pd.DataFrame(rows).set_index("Scheduler")
    print("\n" + "="*70)
    print("TABLE 1 — Summary Results")
    print("="*70)
    print(df.to_string())
    print("="*70)
    return df