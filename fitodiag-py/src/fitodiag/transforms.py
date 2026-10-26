"""Transformaciones de imagen compartidas por el entrenamiento y la API.

La API usa exactamente ``eval_transform``: así la imagen que llega del celular
se procesa igual que las imágenes de validación.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int = 224, augment: bool = True) -> T.Compose:
    """Aumento de datos pensado para fotos de campo tomadas con smartphone.

    - RandomResizedCrop: encuadres distintos y hojas a distinta distancia.
    - Flips y rotación: la hoja puede estar en cualquier orientación.
    - ColorJitter fuerte: luz de mediodía, sombra, cielo nublado.
    - GaussianBlur: fotos algo movidas o desenfocadas.
    - RandomErasing: oclusiones (dedos, otras hojas).
    """
    if not augment:
        return eval_transform(img_size)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
