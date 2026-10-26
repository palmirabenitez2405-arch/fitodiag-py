"""Unifica los tres datasets públicos en un único dataset limpio de 15 clases.

Pasos (cada uno queda registrado en data/processed/stats.json):
  1. Recorre data/raw/{soja,mandioca,sesamo} y asigna cada imagen a una clase.
  2. Descarta imágenes ilegibles o muy chicas (< 64 px de lado).
  3. Descarta duplicados exactos (hash MD5 del archivo).
  4. Limita cada clase a --max-per-class imágenes (la mandioca tiene ~13.000 de mosaico).
  5. Redimensiona el lado mayor a --resize px y guarda en JPEG.
  6. Divide en train/val/test estratificado (70/15/15, semilla fija).
  7. Escribe manifest.csv, stats.json y actualiza la tabla de data/README.md.

Uso:
  python scripts/prepare_data.py --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import (  # noqa: E402
    CASSAVA_LABELS, CLASSES, SESAME_KEYWORDS, SOY_KEYWORDS, match_keywords,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIDE = 64


def list_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            yield p


def collect_soy(root: Path) -> list[tuple[Path, str]]:
    items = []
    for p in list_images(root):
        cls = match_keywords(list(p.relative_to(root).parts[:-1]), SOY_KEYWORDS)
        if cls:
            items.append((p, cls))
    return items


def collect_cassava(root: Path) -> list[tuple[Path, str]]:
    csv_path = root / "train.csv"
    if not csv_path.exists():
        return []
    items = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            p = root / "train_images" / row["image_id"]
            if p.exists():
                items.append((p, CASSAVA_LABELS[int(row["label"])]))
    return items


def collect_sesame(root: Path) -> list[tuple[Path, str]]:
    items = []
    for p in list_images(root):
        parts = list(p.relative_to(root).parts[:-1])
        if any("augment" in part.lower() for part in parts):
            continue  # solo originales: las aumentadas generarían fuga entre splits
        cls = match_keywords(parts, SESAME_KEYWORDS)
        if cls:
            items.append((p, cls))
    return items


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_split(items: list[tuple[Path, str]], val: float, test: float, seed: int):
    by_class = defaultdict(list)
    for it in items:
        by_class[it[1]].append(it)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        group = by_class[cls]
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test))
        n_val = max(1, round(n * val))
        splits["test"] += group[:n_test]
        splits["val"] += group[n_test:n_test + n_val]
        splits["train"] += group[n_test + n_val:]
    return splits


def update_readme(readme: Path, table_md: str) -> None:
    start, end = "<!-- CONTEOS:INICIO -->", "<!-- CONTEOS:FIN -->"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end)[1]
        readme.write_text(f"{before}{start}\n{table_md}\n{end}{after}", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--resize", type=int, default=512, help="lado mayor en px")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    rng = random.Random(args.seed)

    collected = (
        collect_soy(raw / "soja") + collect_cassava(raw / "mandioca") + collect_sesame(raw / "sesamo")
    )
    if not collected:
        sys.exit(f"No se encontraron imágenes en {raw}. Corré primero scripts/download_data.sh")
    raw_counts = Counter(c for _, c in collected)
    print(f"Imágenes encontradas: {len(collected)}")

    # Limpieza: ilegibles, muy chicas y duplicados exactos
    seen, clean = set(), []
    dropped = Counter()
    for p, cls in collected:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if min(im.size) < MIN_SIDE:
                    dropped["muy_chica"] += 1
                    continue
        except (UnidentifiedImageError, OSError):
            dropped["ilegible"] += 1
            continue
        h = md5(p)
        if h in seen:
            dropped["duplicada"] += 1
            continue
        seen.add(h)
        clean.append((p, cls))

    # Tope por clase (submuestreo aleatorio reproducible)
    by_class = defaultdict(list)
    for it in clean:
        by_class[it[1]].append(it)
    capped = []
    for cls, group in by_class.items():
        rng.shuffle(group)
        if len(group) > args.max_per_class:
            dropped["tope_por_clase"] += len(group) - args.max_per_class
        capped += group[: args.max_per_class]

    splits = stratified_split(capped, args.val, args.test, args.seed)

    manifest_rows = []
    for split, items in splits.items():
        for i, (src, cls) in enumerate(items):
            dest = out / split / cls / f"{cls}_{i:05d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((args.resize, args.resize))
                im.save(dest, quality=92)
            manifest_rows.append({"split": split, "class": cls, "path": str(dest.relative_to(out)),
                                  "source": str(src)})

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "path", "source"])
        w.writeheader()
        w.writerows(manifest_rows)

    counts = {s: Counter(c for _, c in items) for s, items in splits.items()}
    stats = {
        "raw_counts": dict(raw_counts),
        "dropped": dict(dropped),
        "split_counts": {s: dict(c) for s, c in counts.items()},
        "total": {s: len(items) for s, items in splits.items()},
        "params": vars(args),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    lines = ["| Clase | Originales | Train | Val | Test |", "| --- | ---: | ---: | ---: | ---: |"]
    for cls in CLASSES:
        lines.append(f"| `{cls}` | {raw_counts.get(cls, 0)} | {counts['train'].get(cls, 0)} | "
                     f"{counts['val'].get(cls, 0)} | {counts['test'].get(cls, 0)} |")
    lines.append(f"| **Total** | {len(collected)} | {len(splits['train'])} | {len(splits['val'])} | "
                 f"{len(splits['test'])} |")
    table = "\n".join(lines)
    update_readme(Path("data/README.md"), table)

    missing = [c for c in CLASSES if raw_counts.get(c, 0) == 0]
    print(table)
    print(f"Descartadas: {dict(dropped)}")
    if missing:
        print(f"ATENCIÓN: clases sin imágenes: {missing}")


if __name__ == "__main__":
    main()
