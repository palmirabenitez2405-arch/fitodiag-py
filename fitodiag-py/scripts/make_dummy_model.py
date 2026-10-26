"""Crea un modelo con pesos aleatorios para probar la API/UI/Docker sin entrenar.

Las predicciones no tienen sentido: sirve solo para verificar que todo el sistema
funciona de punta a punta antes de tener el modelo real (semanas 1-3).

Uso: python scripts/make_dummy_model.py --out models/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import CLASSES  # noqa: E402
from fitodiag.model import build_model, save_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/best_model.pt")
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    args = ap.parse_args()
    model = build_model(args.arch, len(CLASSES), pretrained=False)
    save_checkpoint(args.out, model, args.arch, CLASSES, 224, {"nota": "modelo de prueba, pesos aleatorios"})
    print(f"Modelo de prueba guardado en {args.out}")


if __name__ == "__main__":
    main()
