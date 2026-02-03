from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
import onnxruntime as ort
from .model_unet import UNetSmall
from .utils import overlay_mask_bgr, mask_to_bbox, resolve_torch_device

def infer_torch(image_path: str, ckpt_path: str, out_path: str = "out.png",
               image_size: int = 256, device: str = "auto"):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    dev = resolve_torch_device(device)
    x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
    x = x.to(dev)

    model = UNetSmall().to(dev)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        mask = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    vis = overlay_mask_bgr(img, mask)
    bb = mask_to_bbox(mask)
    if bb:
        x0, y0, x1, y1 = bb
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)
    return Path(out_path)

def _ort_providers(device: str):
    d = (device or "auto").lower()
    if d == "cpu":
        return ["CPUExecutionProvider"]
    if d == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def infer_onnx(image_path: str, onnx_path: str, out_path: str = "out_onnx.png",
              image_size: int = 256, device: str = "auto"):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1x3xHxW
    providers = _ort_providers(device)
    sess = ort.InferenceSession(onnx_path, providers=providers)

    logits = sess.run(["logits"], {"image": x})[0]
    mask = 1 / (1 + np.exp(-logits[0, 0]))

    vis = overlay_mask_bgr(img, mask)
    bb = mask_to_bbox(mask)
    if bb:
        x0, y0, x1, y1 = bb
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)
    return Path(out_path)
