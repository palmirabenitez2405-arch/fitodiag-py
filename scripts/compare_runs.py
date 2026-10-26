"""Compara corridas y genera las tablas y figuras que usa el informe IEEE.

Uso:
  python scripts/compare_runs.py --runs models/efficientnet_b0 models/resnet50 \
      models/effnet_baseline --best models/efficientnet_b0

Escribe:
  paper/tables/results.tex       tabla principal (se incluye con \\input en el paper)
  paper/tables/per_class.tex     F1 por clase de cada modelo
  paper/tables/dataset.tex       conteos del dataset
  paper/figures/confusion_matrix.pdf   matriz del mejor modelo
  paper/figures/curves.pdf       curvas de entrenamiento (F1 macro de validación)
  results.md                     la misma tabla en Markdown para el README
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import CLASSES, SHORT_LABELS  # noqa: E402

PRETTY = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50"}
TEX_NAMES = {"eff": "EfficientNet-B0", "res": "ResNet50", "base": r"EffNet-B0 base$^{\dagger}$"}


def label_for(run: Path, cfg: dict) -> str:
    name = PRETTY.get(cfg.get("arch", ""), run.name)
    flags = []
    if cfg.get("no_augment"):
        flags.append("sin aug.")
    if cfg.get("no_balance"):
        flags.append("sin bal.")
    return f"{name} ({', '.join(flags)})" if flags else name


def macro_key(cfg: dict) -> str | None:
    """Prefijo de las macros LaTeX: \\effAcc, \\resFone, \\baseTop, etc."""
    if cfg.get("no_augment") or cfg.get("no_balance"):
        return "base" if cfg.get("arch") == "efficientnet_b0" else None
    return {"efficientnet_b0": "eff", "resnet50": "res"}.get(cfg.get("arch"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--best", required=True, help="corrida cuya matriz de confusión va al informe")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--data-stats", default="data/processed/stats.json")
    args = ap.parse_args()

    paper = Path(args.paper)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)

    rows, per_class = [], {}
    fig, ax = plt.subplots(figsize=(3.5, 2.0), dpi=200)
    macros = [r"% Generado por scripts/compare_runs.py. No editar a mano."]
    for r in args.runs:
        run = Path(r)
        m = json.loads((run / "metrics.json").read_text())
        cfg = json.loads((run / "config.json").read_text())
        label = label_for(run, cfg)
        key = macro_key(cfg)
        if key:
            pc = m.get("accuracy_per_crop", {})
            vals = {
                "Acc": f"{100 * m['accuracy']:.1f}", "Fone": f"{100 * m['f1_macro']:.1f}",
                "Top": f"{100 * m['top3_accuracy']:.1f}", "Ms": f"{m['latency_cpu']['mean_ms']:.0f}",
                "Msp": f"{m['latency_cpu']['p95_ms']:.0f}", "Params": f"{m['params_millions']:.1f}",
                "Mb": f"{m['model_size_mb']:.0f}", "Wrong": f"{100 * m['wrong_crop_rate']:.1f}",
                "Soja": f"{100 * pc.get('soja', 0):.1f}", "Mandioca": f"{100 * pc.get('mandioca', 0):.1f}",
                "Sesamo": f"{100 * pc.get('sesamo', 0):.1f}", "Min": f"{cfg.get('train_minutes', 0):.0f}",
            }
            f1s = m["f1_per_class"]
            worst = min(f1s, key=f1s.get)
            vals["WorstClass"] = SHORT_LABELS.get(worst, worst)
            vals["WorstFone"] = f"{100 * f1s[worst]:.1f}"
            macros += [rf"\newcommand{{\{key}{k}}}{{{v}}}" for k, v in vals.items()]
        rows.append({
            "Modelo": label,
            "Params (M)": m["params_millions"],
            "Acc. (%)": 100 * m["accuracy"],
            "F1 macro (%)": 100 * m["f1_macro"],
            "Top-3 (%)": 100 * m["top3_accuracy"],
            "CPU (ms)": m["latency_cpu"]["mean_ms"],
            "Entren. (min)": cfg.get("train_minutes", float("nan")),
            "_tex": TEX_NAMES.get(key, label),
        })
        per_class[label] = m["f1_per_class"]
        hist = pd.read_csv(run / "history.csv")
        ax.plot(hist["epoch"], 100 * hist["val_f1_macro"], marker="o", markersize=2, linewidth=1, label=label)

    ax.set_xlabel("Época", fontsize=6)
    ax.set_ylabel("F1 macro val. (%)", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(paper / "figures" / "curves.pdf")
    fig.savefig(paper / "figures" / "curves.png")
    plt.close(fig)

    df = pd.DataFrame(rows)
    (Path("results.md")).write_text(df.drop(columns="_tex").to_markdown(index=False, floatfmt=".1f") + "\n",
                                    encoding="utf-8")

    # Tabla principal en LaTeX (booktabs no hace falta: IEEEtran usa \hline)
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline",
             r"Modelo & Params & Acc. & F1$_{macro}$ & Top-3 & CPU \\",
             r" & (M) & (\%) & (\%) & (\%) & (ms) \\", r"\hline"]
    best_f1 = df["F1 macro (%)"].max()
    for _, row in df.iterrows():
        f1 = f"{row['F1 macro (%)']:.1f}"
        if row["F1 macro (%)"] == best_f1:
            f1 = rf"\textbf{{{f1}}}"
        lines.append(f"{row['_tex']} & {row['Params (M)']:.1f} & {row['Acc. (%)']:.1f} & {f1} & "
                     f"{row['Top-3 (%)']:.1f} & {row['CPU (ms)']:.0f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper / "tables" / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # F1 por clase
    labels = list(per_class)
    cols = "l" + "r" * len(labels)
    short_head = " & ".join(lab.split(" (")[0] if "(" not in lab else lab for lab in labels)
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\hline", rf"Clase & {short_head} \\", r"\hline"]
    for c in CLASSES:
        vals = " & ".join(f"{100 * per_class[lab].get(c, 0):.0f}" for lab in labels)
        lines.append(f"{SHORT_LABELS[c]} & {vals} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (paper / "tables" / "per_class.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Dataset
    stats_path = Path(args.data_stats)
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        sc = st["split_counts"]
        lines = [r"\begin{tabular}{lrrrr}", r"\hline", r"Cultivo & Clases & Train & Val & Test \\", r"\hline"]
        for crop, pretty in [("soja", "Soja"), ("mandioca", "Mandioca"), ("sesamo", "S\\'esamo")]:
            cls = [c for c in CLASSES if c.startswith(crop + "__")]
            tr = sum(sc["train"].get(c, 0) for c in cls)
            va = sum(sc["val"].get(c, 0) for c in cls)
            te = sum(sc["test"].get(c, 0) for c in cls)
            lines.append(f"{pretty} & {len(cls)} & {tr} & {va} & {te} \\\\")
        lines += [r"\hline",
                  rf"Total & {len(CLASSES)} & {st['total']['train']} & {st['total']['val']} & "
                  rf"{st['total']['test']} \\", r"\hline", r"\end{tabular}"]
        (paper / "tables" / "dataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        raw_total = sum(st["raw_counts"].values())
        kept = sum(st["total"].values())
        macros += [rf"\newcommand{{\dsRaw}}{{{raw_total}}}", rf"\newcommand{{\dsKept}}{{{kept}}}",
                   rf"\newcommand{{\dsTrain}}{{{st['total']['train']}}}",
                   rf"\newcommand{{\dsVal}}{{{st['total']['val']}}}",
                   rf"\newcommand{{\dsTest}}{{{st['total']['test']}}}",
                   rf"\newcommand{{\dsDup}}{{{st['dropped'].get('duplicada', 0)}}}"]

    (paper / "tables" / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    best = Path(args.best)
    shutil.copy(best / "confusion_matrix.pdf", paper / "figures" / "confusion_matrix.pdf")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"Tablas y figuras escritas en {paper}/")


if __name__ == "__main__":
    main()
