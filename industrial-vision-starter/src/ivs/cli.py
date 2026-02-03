# src/ivs/cli.py
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import typer
from rich import print as rprint

from .data_synth import SynthParams, synth_sample
from .export_onnx import export_onnx as export_onnx_fn
from .infer import infer_onnx, infer_torch
from .train import train as train_fn
from .utils import TrainConfig

app = typer.Typer(add_completion=False)


@app.command()
def synth(out_dir: str = "data/demo", n: int = 20, image_size: int = 256, seed: int = 42):
    """
    Generate synthetic images + masks for quick demos (no real data needed).
    """
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n):
        img, mask = synth_sample(rng, SynthParams(image_size=image_size))
        cv2.imwrite(str(out / "images" / f"{i:05d}.png"), img)
        cv2.imwrite(str(out / "masks" / f"{i:05d}.png"), mask)

    rprint(f"[green]Wrote[/green] {n} samples to {out}")


@app.command()
def train(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 3e-4,
    image_size: int = 256,
    device: str = "auto",  # auto|cpu|cuda
    amp: bool = True,  # AMP used only on CUDA
    out_dir: str = "runs/ivs",
):
    """
    Train a small U-Net on the synthetic dataset.
    Creates a run folder under out_dir and stores model.pt there.
    """
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        device=device,
        amp=amp,
        out_dir=out_dir,
    )
    ckpt = train_fn(cfg)
    rprint(f"[bold]Checkpoint:[/bold] {ckpt}")


@app.command("export-onnx")
def export_onnx(
    ckpt: str,
    out_path: str = "",
    image_size: int = 256,
):
    """
    Export ONNX.

    Default behavior:
      - If out_path is not provided, export next to the checkpoint:
        runs/.../<timestamp>/model.onnx (+ model.onnx.data if needed)
    """
    ckpt_p = Path(ckpt)
    if not out_path:
        out_path = str(ckpt_p.parent / "model.onnx")

    out = export_onnx_fn(ckpt, out_path, image_size=image_size)
    rprint(f"[green]Exported[/green] {out}")


@app.command()
def infer(
    image: str,
    ckpt: str,
    out: str = "",
    image_size: int = 256,
    backend: str = "torch",  # torch|onnx
    device: str = "auto",  # auto|cpu|cuda
    onnx_path: str = "",
):
    """
    Run inference on a single image.

    Default behavior:
      - If out is not provided, write output next to checkpoint:
          runs/.../<timestamp>/out.png           (torch)
          runs/.../<timestamp>/out_onnx.png      (onnx)

      - If backend is onnx and onnx_path is not provided, use:
          runs/.../<timestamp>/model.onnx
    """
    ckpt_p = Path(ckpt)
    run_dir = ckpt_p.parent

    if not out:
        out = str(run_dir / ("out_onnx.png" if backend == "onnx" else "out.png"))

    if backend == "onnx":
        if not onnx_path:
            onnx_path = str(run_dir / "model.onnx")
        p = infer_onnx(image, onnx_path, out, image_size=image_size, device=device)
    else:
        p = infer_torch(image, ckpt, out, image_size=image_size, device=device)

    rprint(f"[green]Wrote[/green] {p}")


@app.command()
def bench(
    onnx_path: str,
    n: int = 100,
    image_size: int = 256,
    device: str = "auto",
):
    """
    Benchmark ONNXRuntime inference throughput.

    Note:
      - Provide the ONNX path explicitly (recommended: runs/.../<timestamp>/model.onnx)
      - device: auto|cpu|cuda (cuda requires onnxruntime-gpu and proper CUDA libs)
    """
    import onnxruntime as ort

    x = np.random.rand(1, 3, image_size, image_size).astype(np.float32)

    providers = ["CPUExecutionProvider"]
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)

    # warmup
    for _ in range(10):
        _ = sess.run(["logits"], {"image": x})[0]

    t0 = time.time()
    for _ in range(n):
        _ = sess.run(["logits"], {"image": x})[0]
    dt = time.time() - t0

    rprint(f"[bold]ONNX[/bold] {n/dt:.1f} FPS ({providers[0]}, batch=1, {image_size}x{image_size})")


def main():
    app()


if __name__ == "__main__":
    main()
