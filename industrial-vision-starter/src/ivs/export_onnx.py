from __future__ import annotations
from pathlib import Path
import torch
from .model_unet import UNetSmall

def export_onnx(ckpt: str, out_path: str = "model.onnx", image_size: int = 256) -> Path:
    ckpt = Path(ckpt)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = UNetSmall()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        out.as_posix(),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return out
