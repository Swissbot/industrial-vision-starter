from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

@dataclass
class TrainConfig:
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 8
    lr: float = 3e-4
    train_steps: int = 300   # steps per epoch (synthetic)
    val_steps: int = 50
    seed: int = 42
    device: str = "auto"     # auto|cpu|cuda
    out_dir: str = "runs/ivs"
    amp: bool = True         # automatic mixed precision on CUDA

def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def resolve_torch_device(device: str):
    """
    device: auto|cpu|cuda
    """
    import torch
    d = (device or "auto").lower()
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def overlay_mask_bgr(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask = (mask01 > 0.5).astype(np.uint8)
    color = np.zeros_like(img_bgr)
    color[:, :, 1] = 255  # green overlay
    out = img_bgr.copy()
    out[mask == 1] = cv2.addWeighted(out[mask == 1], 1 - alpha, color[mask == 1], alpha, 0)
    return out

def mask_to_bbox(mask01: np.ndarray):
    mask = (mask01 > 0.5).astype(np.uint8)
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)
