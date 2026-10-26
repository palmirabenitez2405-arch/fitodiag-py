"""Creación, guardado y carga de modelos (EfficientNet-B0 y ResNet50 vía timm)."""

from __future__ import annotations

from pathlib import Path

import timm
import torch

ARCHS = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
}


def build_model(arch: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2) -> torch.nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Arquitectura no soportada: {arch}. Opciones: {list(ARCHS)}")
    return timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def head_parameters(model: torch.nn.Module):
    """Parámetros de la capa de clasificación (para la etapa de calentamiento)."""
    return model.get_classifier().parameters()


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_parameters(model):
        p.requires_grad = True


def save_checkpoint(path: str | Path, model: torch.nn.Module, arch: str, classes: list[str],
                    img_size: int, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "classes": classes,
        "img_size": img_size,
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
