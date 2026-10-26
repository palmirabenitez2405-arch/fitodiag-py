"""Exporta el dataset de soja de Hugging Face a carpetas por clase.

Fuente: Project-AgML/MH_SoyaHealthVision_disease_classification_leaf
(reformateo de Shinde & Attar, Mendeley Data, doi:10.17632/hkbgh5s3b7.1).

Uso: python scripts/export_hf_soy.py --out data/raw/soja
"""

from __future__ import annotations

import argparse
from pathlib import Path

HF_ID = "Project-AgML/MH_SoyaHealthVision_disease_classification_leaf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/soja")
    args = ap.parse_args()

    from datasets import load_dataset  # import tardío: solo hace falta acá

    out = Path(args.out)
    ds = load_dataset(HF_ID)
    n = 0
    for split_name, split in ds.items():
        label_names = split.features["label"].names
        for i, row in enumerate(split):
            label = label_names[row["label"]]
            dest = out / label
            dest.mkdir(parents=True, exist_ok=True)
            img = row["image"].convert("RGB")
            img.save(dest / f"{split_name}_{i:05d}.jpg", quality=95)
            n += 1
    print(f"Exportadas {n} imágenes de soja en {out}")


if __name__ == "__main__":
    main()
