"""Grad-CAM: mapa de calor de las zonas de la hoja que más pesaron en el diagnóstico.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Funciona con cualquier modelo de timm porque usa ``forward_features`` (mapa de
características de la última capa convolucional) y ``forward_head`` (clasificador).
Sirve tanto para EfficientNet-B0 como para ResNet50 sin elegir capas a mano.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def logits_and_cam(model: torch.nn.Module, x: torch.Tensor,
                   mask: torch.Tensor | None = None) -> tuple[torch.Tensor, np.ndarray]:
    """Una sola pasada hacia adelante y atrás: devuelve los logits y el mapa Grad-CAM.

    ``mask`` (opcional) suma -inf a las clases excluidas solo para elegir la clase a
    explicar (la ganadora después del filtro por cultivo); los logits devueltos son
    los originales. El mapa queda en [0, 1] con la resolución espacial de la última
    capa (7x7 para una entrada de 224).
    """
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        scores = logits[0] if mask is None else logits[0] + mask
        target = int(scores.argmax())
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()

    weights = feats.grad[0].mean(dim=(1, 2))                      # importancia de cada canal
    cam = torch.relu((weights[:, None, None] * feats[0]).sum(0))  # combinación ponderada
    cam = cam.detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return logits.detach(), cam


def _colormap(v: np.ndarray) -> np.ndarray:
    """Paleta tipo "jet" (azul → verde → amarillo → rojo) sin depender de matplotlib."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_png(img: Image.Image, cam: np.ndarray, img_size: int, alpha: float = 0.45) -> bytes:
    """Superpone el mapa sobre el mismo recorte que vio el modelo y devuelve un PNG."""
    crop = T.Compose([T.Resize(int(img_size * 256 / 224)), T.CenterCrop(img_size)])(img.convert("RGB"))
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(crop.size, Image.BILINEAR)
    heat_rgb = _colormap(np.asarray(heat, dtype=np.float32) / 255)
    base = np.asarray(crop, dtype=np.float32)
    blended = ((1 - alpha) * base + alpha * heat_rgb).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()
