"""Evalúa un modelo entrenado sobre el split de test.

Calcula: accuracy, F1 macro y ponderado, precisión/recall por clase, accuracy
por cultivo, matriz de confusión, y tiempo de inferencia en CPU (batch 1).

Uso:
  python scripts/evaluate.py --run models/efficientnet_b0

Salida en la carpeta de la corrida:
  metrics.json, classification_report.csv, confusion_matrix.png / .pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import SHORT_LABELS, crop_of  # noqa: E402
from fitodiag.model import load_checkpoint  # noqa: E402
from fitodiag.transforms import eval_transform  # noqa: E402


@torch.no_grad()
def predict_all(model, loader, device):
    preds, gts, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = logits.softmax(1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        gts.append(y)
    return torch.cat(gts).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy()


@torch.no_grad()
def cpu_latency(model, img_size: int, runs: int = 50, warmup: int = 10, threads: int = 2) -> dict:
    """Latencia de una imagen en CPU, como en el contenedor de la API."""
    torch.set_num_threads(threads)
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()), "p95_ms": float(np.percentile(times, 95)),
            "threads": threads, "runs": runs}


def plot_confusion(cm: np.ndarray, labels: list[str], out_base: Path, title: str) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    short = [SHORT_LABELS.get(c, c) for c in labels]
    fig, ax = plt.subplots(figsize=(3.5, 3.3), dpi=200)  # ancho de una columna IEEE
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels)), short, fontsize=5)
    ax.set_xlabel("Predicción", fontsize=6)
    ax.set_ylabel("Real", fontsize=6)
    ax.set_title(title, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=3.6,
                        color="white" if v > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="carpeta de la corrida, p. ej. models/efficientnet_b0")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--latency-runs", type=int, default=50)
    args = ap.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(run / "best_model.pt", device)
    classes, img_size = ckpt["classes"], ckpt["img_size"]

    ds = ImageFolder(Path(args.data) / args.split, transform=eval_transform(img_size))
    if ds.classes != classes:
        raise SystemExit(f"Las clases del split no coinciden con las del modelo:\n{ds.classes}\n{classes}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    y, yhat, probs = predict_all(model, dl, device)
    labels_idx = list(range(len(classes)))
    report = classification_report(y, yhat, labels=labels_idx, target_names=classes, output_dict=True,
                                   zero_division=0)
    pd.DataFrame(report).T.to_csv(run / "classification_report.csv", float_format="%.4f")

    crops = np.array([crop_of(c) for c in classes])
    per_crop = {}
    for crop in sorted(set(crops)):
        mask = crops[y] == crop
        if mask.any():
            per_crop[crop] = float(accuracy_score(y[mask], yhat[mask]))
    # ¿Cuántas veces el modelo se equivoca de cultivo? (error grave para el usuario)
    crop_confusion = float((crops[y] != crops[yhat]).mean())

    top3 = np.argsort(-probs, axis=1)[:, :3]
    top3_acc = float(np.mean([y[i] in top3[i] for i in range(len(y))]))

    cm = confusion_matrix(y, yhat, labels=labels_idx)
    plot_confusion(cm, classes, run / "confusion_matrix", f"{ckpt['arch']} — test")

    ckpt_path = run / "best_model.pt"
    latency = cpu_latency(model, img_size, runs=args.latency_runs)
    metrics = {
        "arch": ckpt["arch"],
        "split": args.split,
        "n_images": int(len(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "top3_accuracy": top3_acc,
        "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, yhat, average="weighted", zero_division=0)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "accuracy_per_crop": per_crop,
        "wrong_crop_rate": crop_confusion,
        "f1_per_class": {c: float(report[c]["f1-score"]) for c in classes},
        "latency_cpu": latency,
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
        "model_size_mb": ckpt_path.stat().st_size / 1e6,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "f1_per_class",
                                                                    "classes")}, indent=2))


if __name__ == "__main__":
    main()
