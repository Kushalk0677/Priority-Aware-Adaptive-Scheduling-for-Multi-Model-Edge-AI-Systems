"""
model_zoo.py — Real model wrappers with simulation fallbacks.

Each model exposes a single `.run(input)` method and reports:
  - actual measured latency
  - estimated energy (via CPU proxy)

If a library isn't installed, falls back to a calibrated simulator
so the scheduler experiments still run correctly.
"""

import time
import random
import platform
import numpy as np

# ── CPU energy proxy (mJ estimate) ───────────────────────────────────────────
# Maps CPU usage fraction × time × TDP to millijoules.
# Typical laptop TDP ~15W; desktop ~65W. We default to 15W.
TDP_WATTS = 15.0

def estimate_energy_mj(elapsed_sec: float, cpu_fraction: float = 0.8) -> float:
    return TDP_WATTS * cpu_fraction * elapsed_sec * 1000  # W × s × 1000 → mJ


# ── Base class ────────────────────────────────────────────────────────────────

class BaseModel:
    name: str = "base"
    priority: float = 1.0
    deadline_ms: float = 1000.0

    def run(self, input_data=None):
        raise NotImplementedError

    def make_dummy_input(self):
        return None


# ── 1. Image Classification — MobileNetV2 ────────────────────────────────────

class MobileNetModel(BaseModel):
    name = "mobilenet_v2"
    priority = 2.0
    deadline_ms = 200.0

    def __init__(self):
        self.real = False
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T
            self.model = models.mobilenet_v2(weights=None)
            self.model.eval()
            self.transform = T.Compose([T.ToTensor(),
                                        T.Resize((224, 224)),
                                        T.Normalize([0.485,0.456,0.406],
                                                    [0.229,0.224,0.225])])
            self.torch = torch
            self.real = True
            print(f"  ✓ {self.name}: using real PyTorch model")
        except ImportError:
            print(f"  ⚠ {self.name}: PyTorch not found — using simulator")

    def make_dummy_input(self):
        if self.real:
            return self.torch.randn(1, 3, 224, 224)
        return None

    def run(self, input_data=None):
        start = time.perf_counter()
        if self.real:
            with self.torch.no_grad():
                inp = input_data if input_data is not None else self.make_dummy_input()
                _ = self.model(inp)
        else:
            # Calibrated simulation: MobileNetV2 ~20-50ms on CPU
            time.sleep(random.gauss(35, 8) / 1000)
        elapsed = time.perf_counter() - start
        return elapsed * 1000, estimate_energy_mj(elapsed, 0.7)


# ── 2. Object Detection — YOLOv5n ────────────────────────────────────────────

class YOLOModel(BaseModel):
    name = "yolov5n"
    priority = 3.0
    deadline_ms = 300.0

    def __init__(self):
        self.real = False
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n',
                                        pretrained=False, verbose=False)
            self.model.eval()
            self.torch = torch
            self.real = True
            print(f"  ✓ {self.name}: using real YOLOv5n model")
        except Exception:
            print(f"  ⚠ {self.name}: ultralytics not found — using simulator")

    def make_dummy_input(self):
        if self.real:
            return self.torch.randn(1, 3, 416, 416)
        return np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

    def run(self, input_data=None):
        start = time.perf_counter()
        if self.real:
            with self.torch.no_grad():
                inp = input_data if input_data is not None else self.make_dummy_input()
                _ = self.model(inp)
        else:
            # YOLOv5n CPU ~60-120ms
            time.sleep(random.gauss(80, 15) / 1000)
        elapsed = time.perf_counter() - start
        return elapsed * 1000, estimate_energy_mj(elapsed, 0.85)


# ── 3. Speech Recognition — Whisper Tiny ─────────────────────────────────────

class WhisperModel(BaseModel):
    name = "whisper_tiny"
    priority = 2.0
    deadline_ms = 500.0

    def __init__(self):
        self.real = False
        try:
            import whisper
            import numpy as np
            self.model = whisper.load_model("tiny")
            self.np = np
            self.real = True
            print(f"  ✓ {self.name}: using real Whisper tiny model")
        except ImportError:
            print(f"  ⚠ {self.name}: openai-whisper not found — using simulator")

    def make_dummy_input(self):
        # 1 second of random audio at 16kHz
        return np.random.randn(16000).astype(np.float32)

    def run(self, input_data=None):
        start = time.perf_counter()
        if self.real:
            audio = input_data if input_data is not None else self.make_dummy_input()
            _ = self.model.transcribe(audio, fp16=False)
        else:
            # Whisper tiny on CPU ~100-200ms per second of audio
            time.sleep(random.gauss(150, 25) / 1000)
        elapsed = time.perf_counter() - start
        return elapsed * 1000, estimate_energy_mj(elapsed, 0.9)


# ── 4. NLP Sentiment — DistilBERT ────────────────────────────────────────────

class SentimentModel(BaseModel):
    name = "distilbert_sentiment"
    priority = 1.5
    deadline_ms = 400.0

    def __init__(self):
        self.real = False
        try:
            from transformers import pipeline
            self.pipe = pipeline("sentiment-analysis",
                                  model="distilbert-base-uncased-finetuned-sst-2-english",
                                  device=-1)
            self.real = True
            print(f"  ✓ {self.name}: using real DistilBERT model")
        except ImportError:
            print(f"  ⚠ {self.name}: transformers not found — using simulator")

    def make_dummy_input(self):
        phrases = [
            "The system is performing well under load.",
            "Edge AI scheduling is a complex problem.",
            "The robot successfully avoided the obstacle.",
            "Latency spikes are causing task failures.",
        ]
        return random.choice(phrases)

    def run(self, input_data=None):
        start = time.perf_counter()
        if self.real:
            text = input_data if input_data is not None else self.make_dummy_input()
            _ = self.pipe(text)
        else:
            # DistilBERT CPU ~40-80ms
            time.sleep(random.gauss(55, 12) / 1000)
        elapsed = time.perf_counter() - start
        return elapsed * 1000, estimate_energy_mj(elapsed, 0.75)


# ── 5. Depth Estimation — MiDaS Small ────────────────────────────────────────

class DepthModel(BaseModel):
    name = "midas_small"
    priority = 1.0
    deadline_ms = 600.0

    def __init__(self):
        self.real = False
        try:
            import torch
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                                         verbose=False)
            self.model.eval()
            self.torch = torch
            self.real = True
            print(f"  ✓ {self.name}: using real MiDaS model")
        except Exception:
            print(f"  ⚠ {self.name}: MiDaS not available — using simulator")

    def make_dummy_input(self):
        if self.real:
            return self.torch.randn(1, 3, 256, 256)
        return None

    def run(self, input_data=None):
        start = time.perf_counter()
        if self.real:
            with self.torch.no_grad():
                inp = input_data if input_data is not None else self.make_dummy_input()
                _ = self.model(inp)
        else:
            # MiDaS small ~80-150ms
            time.sleep(random.gauss(110, 20) / 1000)
        elapsed = time.perf_counter() - start
        return elapsed * 1000, estimate_energy_mj(elapsed, 0.8)


# ── Model Registry ────────────────────────────────────────────────────────────

def load_models(selection=None):
    """
    Load and return a dict of model instances.
    selection: list of names to load, or None for all.
    """
    all_models = {
        "mobilenet_v2":         MobileNetModel,
        "yolov5n":              YOLOModel,
        "whisper_tiny":         WhisperModel,
        "distilbert_sentiment": SentimentModel,
        "midas_small":          DepthModel,
    }
    if selection:
        all_models = {k: v for k, v in all_models.items() if k in selection}

    print("\nLoading models...")
    loaded = {}
    for name, cls in all_models.items():
        loaded[name] = cls()
    print(f"  → {len(loaded)} models ready\n")
    return loaded
