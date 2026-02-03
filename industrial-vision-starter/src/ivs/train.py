# src/ivs/train.py
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console

from .dataset import SyntheticDatasetConfig, SyntheticSegDataset
from .model_unet import UNetSmall
from .utils import TrainConfig, ensure_dir, set_seed, resolve_torch_device

console = Console()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1 - (num / den)
    return dice.mean()


def train(cfg: TrainConfig) -> Path:
    """
    Train a small U-Net on a synthetic segmentation dataset.

    CPU/GPU:
      - device=auto|cpu|cuda
      - AMP is enabled only when running on CUDA (no warnings on CPU).
    """
    set_seed(cfg.seed)
    out = ensure_dir(cfg.out_dir)
    run_dir = ensure_dir(out / time.strftime("%Y%m%d_%H%M%S"))
    ckpt_path = run_dir / "model.pt"

    device = resolve_torch_device(cfg.device)
    use_amp = bool(cfg.amp) and device.type == "cuda"

    # Robust AMP handling:
    # - On CUDA: use torch.amp autocast + GradScaler
    # - On CPU: no autocast, no scaler
    amp_ctx = torch.amp.autocast("cuda", enabled=True) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    console.print(f"[bold]Device:[/bold] {device}  | AMP: {use_amp}")

    ds_train = SyntheticSegDataset(
        SyntheticDatasetConfig(
            image_size=cfg.image_size,
            length=cfg.train_steps * cfg.batch_size,
            seed=cfg.seed,
        )
    )
    ds_val = SyntheticSegDataset(
        SyntheticDatasetConfig(
            image_size=cfg.image_size,
            length=cfg.val_steps * cfg.batch_size,
            seed=cfg.seed + 999,
        )
    )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = UNetSmall().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    best = 1e9

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0

        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = model(x)
                loss = 0.6 * bce(logits, y) + 0.4 * dice_loss(logits, y)

            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tr_loss += float(loss.item())

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with amp_ctx:
                    logits = model(x)
                    loss = 0.6 * bce(logits, y) + 0.4 * dice_loss(logits, y)
                va_loss += float(loss.item())

        tr_loss /= max(1, len(dl_train))
        va_loss /= max(1, len(dl_val))
        console.print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    console.print(f"[green]Saved:[/green] {ckpt_path}")
    return ckpt_path
