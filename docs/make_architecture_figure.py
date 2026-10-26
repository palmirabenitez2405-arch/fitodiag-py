"""Dibuja el diagrama de arquitectura (README y figura 1 del informe).

Uso: python docs/make_architecture_figure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EDGE, ACCENT, INK, QUIET, TINT = "#6b6b6b", "#2e7d32", "#1a1a1a", "#555555", "#eef3ee"

fig, ax = plt.subplots(figsize=(3.5, 1.95), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")


def box(x, y, w, h, title, sub, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="#e8f2e8" if accent else "white", ec=ACCENT if accent else EDGE,
                                lw=1.0 if accent else 0.6))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=5.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=4.2, color=QUIET)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=EDGE))


W, H = 22, 11
xs = [1, 26, 51, 76]
ya, yb = 38, 8
ax.text(1, 53, "Entrenamiento (Google Colab, GPU)", fontsize=4.6, color=QUIET)
box(xs[0], ya, W, H, "Datasets", "3 fuentes, 15 clases")
box(xs[1], ya, W, H, "prepare_data.py", "limpieza, 70/15/15")
box(xs[2], ya, W, H, "train.py", "EffNet-B0 / ResNet50")
box(xs[3], ya, W, H, "best_model.pt", "pesos + clases", accent=True)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + W, ya + H / 2, b, ya + H / 2)

ax.add_patch(FancyBboxPatch((24.5, 2.5), 75, 27, boxstyle="round,pad=0,rounding_size=1.6", fc=TINT, ec=EDGE,
                            lw=0.5))
ax.text(26, 26, "docker compose up", fontsize=4.6, color=INK, weight="bold")
box(xs[0], yb, W, H, "Celular", "foto de la hoja")
box(xs[1], yb, W, H, "Streamlit UI", ":8501")
box(xs[2], yb, W, H, "FastAPI", "POST /predict :8000")
box(xs[3], yb, W, H, "Inferencia CPU", "top-3 + manejo")
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + W, yb + H / 2, b, yb + H / 2)
arrow(xs[3] + W / 2, ya, xs[3] + W / 2, yb + H)
ax.text(xs[3] + W / 2 - 1.5, 31.5, "se carga al iniciar", fontsize=4.0, color=QUIET, ha="right")

fig.tight_layout(pad=0.1)
fig.savefig("paper/figures/arquitectura.pdf")
fig.savefig("docs/arquitectura.png")
print("ok")
