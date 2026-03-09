# PAES — Priority-Aware Edge Scheduler
### Experimental Evaluation Suite

Full research codebase for the paper:
**"Priority-Aware Adaptive Scheduling for Multi-Model Edge AI Systems"**

This repository contains the experimental framework for evaluating **PAES**, a scheduler designed for multi-model AI inference on edge devices.

## Project Structure

```
paes/
├── run_all.py          ← Master runner (start here)
├── scheduler.py        ← FIFO / Round Robin / Static Priority / PAES / EDF / QoS / Deadline Priority Queue
├── experiments.py      ← All 5 experiments
├── figures.py          ← Publication-quality figure generation
├── requirements.txt    ← Dependencies
├── models/
│   └── model_zoo.py    ← Model wrappers (real + simulation fallbacks)
├── results/            ← CSV output per experiment
└── figures/            ← PNG figures for the paper
```


## Setup

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install core dependencies
pip install numpy pandas matplotlib tqdm scipy

# 3. (Optional) Install real models
pip install torch torchvision    # for MobileNetV2, YOLO, MiDaS
pip install ultralytics          # for YOLOv5n
pip install openai-whisper       # for Whisper tiny
pip install transformers         # for DistilBERT
```

Without real models installed, every model falls back to a calibrated
**simulator** that uses realistic timing distributions. The scheduler
experiments are fully valid either way.



## Running

### Quick test (~2–3 min)
```bash
python run_all.py --quick
```

### Full experiment run (~15–20 min)
```bash
python run_all.py
```

### Specific experiments only
```bash
python run_all.py --exp 1 2      # Latency + deadline experiments only
python run_all.py --exp 5        # Sensitivity analysis only
```

### Specific models only
```bash
python run_all.py --models mobilenet_v2 yolov5n whisper_tiny
```



## Experiments

| # | Name                        | What it measures                              |
|---|-----------------------------|-----------------------------------------------|
| 1 | Latency Under Load          | Avg/P95/P99 latency, throughput               |
| 2 | Deadline Miss Rate vs Load  | Miss rate at low/medium/high/extreme load      |
| 3 | Energy Per Task             | mJ/task, relative to FIFO baseline            |
| 4 | Burst Workload Recovery     | Miss rate & P95 lat before/during/after burst |
| 5 | PAES Sensitivity (α/β/γ)   | How weight tuning affects all metrics         |


## Models

| Model                  | Task                  | Priority | Deadline |
|------------------------|-----------------------|----------|----------|
| YOLOv5n                | Object detection      | 3 (high) | 300 ms   |
| MobileNetV2            | Image classification  | 2        | 200 ms   |
| Whisper Tiny           | Speech recognition    | 2        | 500 ms   |
| DistilBERT Sentiment   | NLP inference         | 1.5      | 400 ms   |
| MiDaS Small            | Depth estimation      | 1 (low)  | 600 ms   |

## Simulation vs Real Models

The framework supports two modes:

1. **Real inference mode** – runs actual models using PyTorch / Transformers.
2. **Simulation mode** – uses calibrated latency distributions when models are unavailable.

Both modes produce comparable scheduling behavior, allowing experiments to run on standard laptops.

## PAES Algorithm

PAES assigns a score to each pending task:

Score(task) = α·Priority + β·(1/ExpectedLatency) + γ·(1/ExpectedEnergy)

The scheduler selects the task with the highest score.

This allows the scheduler to balance:
- task importance
- execution speed
- energy cost


## Outputs

After running, check:
- `figures/fig1_latency.png`        — bar charts (use in Section 5.1)
- `figures/fig2_deadline.png`       — line chart with error bands (Section 5.2)
- `figures/fig3_energy.png`         — absolute + relative energy (Section 5.3)
- `figures/fig4_burst.png`          — grouped bars by phase (Section 5.4)
- `figures/fig5_sensitivity.png`    — weight sweep (Section 5.5)
- `figures/fig6_per_model_heatmap.png` — per-model breakdown
- `results/*.csv`                   — raw data for your paper tables



## Scaling Up for Publication

To get paper-quality statistics:

1. Set `N_MAIN = 2000` in `run_all.py`
2. Set `N_REPEATS = 5` in `run_all.py`
3. Install real models (PyTorch + ultralytics + whisper)
4. Run on your target hardware (Raspberry Pi / Jetson Nano if available)

## Reproducibility

All experiments are deterministic given a fixed random seed.  
Raw CSV outputs are stored in `results/` and can be used to regenerate all figures in `figures/`.

## Citation

If you use this code, please cite:

Priority-Aware Adaptive Scheduling for Multi-Model Edge AI Systems (2026)