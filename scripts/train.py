"""Fine-tuning de EfficientNet-B0 o ResNet50 sobre el dataset unificado.

Receta (igual para las dos arquitecturas, para que la comparación sea justa):
  - Pesos iniciales de ImageNet (timm).
  - Etapa 1 (--warmup-epochs): backbone congelado, solo se entrena la cabeza.
  - Etapa 2: todo el modelo, AdamW + cosine annealing.
  - Desbalance: WeightedRandomSampler + CrossEntropy con label smoothing.
  - Early stopping sobre F1 macro de validación.

Ejemplos:
  python scripts/train.py --arch efficientnet_b0 --epochs 15
  python scripts/train.py --arch resnet50 --epochs 15
  python scripts/train.py --arch efficientnet_b0 --no-augment --no-balance --run-name effnet_baseline

Salida en models/<run-name>/: best_model.pt, history.csv, config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import CLASSES  # noqa: E402
from fitodiag.model import build_model, save_checkpoint, set_backbone_trainable  # noqa: E402
from fitodiag.transforms import eval_transform, train_transform  # noqa: E402


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir: Path, img_size: int, batch_size: int, workers: int, augment: bool, balance: bool):
    train_ds = ImageFolder(data_dir / "train", transform=train_transform(img_size, augment))
    val_ds = ImageFolder(data_dir / "val", transform=eval_transform(img_size))
    # ImageFolder ordena alfabéticamente: debe coincidir con CLASSES para que los índices sean los mismos.
    present = train_ds.classes
    if any(c not in CLASSES for c in present):
        raise SystemExit(f"Clases inesperadas en {data_dir}/train: {present}")
    if present != CLASSES:
        print(f"Aviso: faltan clases en el dataset: {sorted(set(CLASSES) - set(present))}")

    targets = np.array(train_ds.targets)
    counts = np.bincount(targets, minlength=len(present))
    if balance:
        class_w = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(torch.as_tensor(class_w[targets], dtype=torch.double),
                                        num_samples=len(targets), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=workers,
                        pin_memory=torch.cuda.is_available())
    return train_dl, val_dl, present, dict(zip(present, counts.tolist()))


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses, preds, gts = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.split(":")[0],
                                                              enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        losses.append(loss.item() * x.size(0))
        preds.append(logits.argmax(1).cpu())
        gts.append(y.cpu())
    preds, gts = torch.cat(preds).numpy(), torch.cat(gts).numpy()
    return {
        "loss": float(np.sum(losses) / len(gts)),
        "acc": float(accuracy_score(gts, preds)),
        "f1_macro": float(f1_score(gts, preds, average="macro", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["efficientnet_b0", "resnet50"], default="efficientnet_b0")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="models")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batches", type=int, default=0, help="solo para pruebas rápidas")
    ap.add_argument("--no-augment", action="store_true", help="ablación: sin data augmentation")
    ap.add_argument("--no-balance", action="store_true", help="ablación: sin sampler ponderado")
    ap.add_argument("--no-pretrained", action="store_true", help="solo para pruebas sin internet")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or args.arch
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {device} | corrida: {run_name}")

    train_dl, val_dl, classes, train_counts = make_loaders(
        Path(args.data), args.img_size, args.batch_size, args.workers,
        augment=not args.no_augment, balance=not args.no_balance)
    if args.max_batches:
        from itertools import islice

        class Limited:
            def __init__(self, dl, n):
                self.dl, self.n = dl, n

            def __iter__(self):
                return islice(iter(self.dl), self.n)
        train_dl, val_dl = Limited(train_dl, args.max_batches), Limited(val_dl, args.max_batches)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params / 1e6:.2f} M | clases: {len(classes)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    config = {**vars(args), "classes": classes, "train_counts": train_counts, "params": n_params,
              "device": device}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    history, best_f1, bad_epochs = [], -1.0, 0
    t_start = time.time()

    # Etapa 1: solo la cabeza
    set_backbone_trainable(model, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr * 3,
                            weight_decay=args.weight_decay)
    for epoch in range(1, args.warmup_epochs + 1):
        tr = run_epoch(model, train_dl, criterion, device, opt, scaler)
        va = run_epoch(model, val_dl, criterion, device)
        history.append({"epoch": epoch, "stage": "warmup", **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"[warmup {epoch}] train_loss={tr['loss']:.3f} val_acc={va['acc']:.3f} val_f1={va['f1_macro']:.3f}")

    # Etapa 2: todo el modelo
    set_backbone_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, criterion, device, opt, scaler)
        va = run_epoch(model, val_dl, criterion, device)
        sched.step()
        row = {"epoch": args.warmup_epochs + epoch, "stage": "finetune",
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(f"[ft {epoch}/{args.epochs}] train_loss={tr['loss']:.3f} train_acc={tr['acc']:.3f} "
              f"val_acc={va['acc']:.3f} val_f1={va['f1_macro']:.3f}")
        if va["f1_macro"] > best_f1:
            best_f1, bad_epochs = va["f1_macro"], 0
            save_checkpoint(out_dir / "best_model.pt", model, args.arch, classes, args.img_size,
                            {"val_acc": va["acc"], "val_f1_macro": va["f1_macro"], "epoch": row["epoch"]})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {args.patience} épocas sin mejorar el F1 macro de validación")
                break

    train_minutes = (time.time() - t_start) / 60
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    config["train_minutes"] = round(train_minutes, 1)
    config["best_val_f1_macro"] = best_f1
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Mejor F1 macro de validación: {best_f1:.4f} | {train_minutes:.1f} min | {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
