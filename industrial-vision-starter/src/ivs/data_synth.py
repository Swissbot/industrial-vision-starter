from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class SynthParams:
    image_size: int = 256
    min_shapes: int = 1
    max_shapes: int = 4

def _rand_bg(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    base = rng.integers(40, 120)
    noise = rng.normal(0, 18, size=(h, w, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    # vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    dist = dist / dist.max()
    vign = (1.0 - 0.35 * dist).astype(np.float32)
    img = (img.astype(np.float32) * vign[..., None]).clip(0, 255).astype(np.uint8)
    return img

def synth_sample(rng: np.random.Generator, p: SynthParams):
    """
    Creates an image and a binary mask for 'target part' (e.g., screw head / component).
    This is intentionally synthetic: enough to show a complete pipeline without real data.
    """
    s = p.image_size
    img = _rand_bg(rng, s, s)
    mask = np.zeros((s, s), dtype=np.uint8)

    n = int(rng.integers(p.min_shapes, p.max_shapes + 1))
    for _ in range(n):
        kind = rng.choice(["circle", "rect", "ring"])
        cx, cy = int(rng.integers(40, s - 40)), int(rng.integers(40, s - 40))
        col = int(rng.integers(130, 230))

        if kind == "circle":
            r = int(rng.integers(16, 42))
            cv2.circle(img, (cx, cy), r, (col, col, col), -1)
            cv2.circle(mask, (cx, cy), r, 255, -1)

        elif kind == "rect":
            w = int(rng.integers(30, 80))
            h = int(rng.integers(20, 70))
            ang = float(rng.integers(0, 180))
            rect = ((cx, cy), (w, h), ang)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillConvexPoly(img, box, (col, col, col))
            cv2.fillConvexPoly(mask, box, 255)

        else:  # ring (screw-like)
            r_outer = int(rng.integers(20, 52))
            r_inner = int(max(8, r_outer - rng.integers(8, 18)))
            cv2.circle(img, (cx, cy), r_outer, (col, col, col), -1)
            cv2.circle(img, (cx, cy), r_inner, (int(col * 0.4),) * 3, -1)
            cv2.circle(mask, (cx, cy), r_outer, 255, -1)

    # add some blur and edges
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if rng.random() < 0.20:
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=int(rng.integers(-10, 10)))

    return img, mask
