from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_synth import SynthParams, synth_sample

@dataclass
class SyntheticDatasetConfig:
    image_size: int = 256
    length: int = 1000
    seed: int = 42

class SyntheticSegDataset(Dataset):
    def __init__(self, cfg: SyntheticDatasetConfig):
        self.cfg = cfg

    def __len__(self):
        return self.cfg.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.cfg.seed + idx)
        img_bgr, mask = synth_sample(rng, SynthParams(image_size=self.cfg.image_size))

        # BGR uint8 -> float32 CHW [0..1]
        img = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # 1xHxW
        return img, y
