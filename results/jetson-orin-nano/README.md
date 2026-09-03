# Jetson PAES Raw Results: CPU-era vs CUDA

Date: 2026-05-16
Host: soren-edge Jetson Orin
Repo: paes (https://github.com/Kushalk0677/paes)

## Layout

- cpu/run_all/: CSVs from `python3 run_all.py --device soren-edge-full`
- cpu/standalone/: CPU-era standalone CSVs for `exp_workload_realism.py` and `exp_overhead.py`
- cpu/logs/: CPU-era script logs
- cuda/run_all/: CSVs from `./run_with_jetson_cuda.sh python run_all.py --device soren-edge-cuda-full`
- cuda/standalone/: CUDA-env standalone CSVs for `exp_workload_realism.py` and `exp_overhead.py`
- cuda/logs/: CUDA run logs plus tegrastats GPU utilization log
- report/: Markdown summary report for Kushal

## Caveat

The CPU-era run used real CPU MobileNetV2, DistilBERT, and MiDaS, but YOLOv5n and Whisper Tiny fell back to simulation. The CUDA run used all five real model wrappers on CUDA.
