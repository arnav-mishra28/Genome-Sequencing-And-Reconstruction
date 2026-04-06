"""
trainer.py
==========
Shared training utilities: early stopping, gradient accumulation,
mixed precision, checkpoint management, and logging.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from config.settings import MODEL_DIR, GRADIENT_CLIP


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class TrainingLogger:
    """JSON-based training logger."""
    def __init__(self, name: str, save_dir: str = None):
        self.name     = name
        self.save_dir = save_dir or MODEL_DIR
        self.history: List[Dict] = []
        self.start_time = time.time()

    def log(self, epoch: int, metrics: Dict):
        entry = {"epoch": epoch, "time": time.time() - self.start_time}
        entry.update(metrics)
        self.history.append(entry)

    def save(self):
        path = os.path.join(self.save_dir, f"{self.name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  [LOG] History saved → {path}")

    def summary(self) -> Dict:
        if not self.history:
            return {}
        last = self.history[-1]
        best_loss = min(h.get("val_loss", float("inf"))
                        for h in self.history)
        return {
            "name":       self.name,
            "epochs":     len(self.history),
            "final_loss": last.get("loss", None),
            "best_val":   best_loss,
            "elapsed_s":  round(time.time() - self.start_time, 1),
        }


def save_checkpoint(model: nn.Module, name: str,
                    extra: Optional[Dict] = None):
    """Save model checkpoint."""
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    state = {"model": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  [CKPT] Saved → {path}")
    return path


def load_checkpoint(model: nn.Module, name: str,
                    device: torch.device = None) -> nn.Module:
    """Load model from checkpoint."""
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    if not os.path.exists(path):
        print(f"  [CKPT WARN] {path} not found — using untrained model.")
        return model
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    print(f"  [CKPT] Loaded ← {path}")
    return model
