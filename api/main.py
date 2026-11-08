"""API REST de FitoDiag-PY (FastAPI).

Endpoints:
  GET  /health    estado del servicio y del modelo
  GET  /classes   lista de clases con su información de manejo
  POST /predict   recibe una imagen (multipart, campo "file") y devuelve el diagnóstico
                  ?cultivo=soja|mandioca|sesamo  limita el diagnóstico a ese cultivo
                  ?explicar=true                 agrega el mapa Grad-CAM (PNG en base64)

Documentación interactiva: http://localhost:8000/docs
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fitodiag.classes import CROPS, crop_of  # noqa: E402
from fitodiag.gradcam import logits_and_cam, overlay_png  # noqa: E402
from fitodiag.model import load_checkpoint  # noqa: E402
from fitodiag.transforms import eval_transform  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
TREATMENTS = json.loads((Path(__file__).parent / "treatments.json").read_text(encoding="utf-8"))

log = logging.getLogger("fitodiag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))

state: dict = {"model": None, "ckpt": None, "transform": None}

# Si con el filtro de cultivo menos de este porcentaje de la probabilidad original
# correspondía al cultivo elegido, la foto probablemente es de otra planta.
CROP_MISMATCH_THRESHOLD = 0.5


class Cultivo(str, Enum):
    soja = "soja"
    mandioca = "mandioca"
    sesamo = "sesamo"


def crop_mask(crop: str) -> torch.Tensor:
    """0 para las clases del cultivo elegido y -inf para las demás (se suma a los logits)."""
    classes = state["ckpt"]["classes"]
    return torch.tensor([0.0 if crop_of(c) == crop else float("-inf") for c in classes])


def load_model() -> None:
    if not MODEL_PATH.exists():
        log.warning("No se encontró el modelo en %s: /predict responderá 503", MODEL_PATH)
        return
    model, ckpt = load_checkpoint(MODEL_PATH, "cpu")
    state.update(model=model, ckpt=ckpt, transform=eval_transform(ckpt["img_size"]))
    with torch.no_grad():  # calentamiento: la primera inferencia es más lenta
        model(torch.zeros(1, 3, ckpt["img_size"], ckpt["img_size"]))
    log.info("Modelo %s cargado (%d clases)", ckpt["arch"], len(ckpt["classes"]))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="FitoDiag-PY",
    description="Diagnóstico de enfermedades foliares en soja, mandioca y sésamo a partir de una foto.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Candidate(BaseModel):
    clase: str
    nombre: str
    cultivo: str
    probabilidad: float


class Prediction(BaseModel):
    clase: str
    nombre: str
    cultivo: str
    agente: str
    severidad: str
    sintomas: str
    probabilidad: float
    confianza_baja: bool
    sugerencia: str
    top3: list[Candidate]
    tiempo_inferencia_ms: float
    modelo: str
    aviso: str
    cultivo_filtrado: str | None = None
    aviso_cultivo: str | None = None
    gradcam_png: str | None = None


def info(cls: str) -> dict:
    return TREATMENTS.get(cls, {"nombre": cls, "agente": "", "severidad": "", "sintomas": "", "sugerencia": ""})


@app.get("/health")
def health() -> dict:
    ckpt = state["ckpt"]
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model": ckpt["arch"] if ckpt else None,
        "model_path": str(MODEL_PATH),
        "n_classes": len(ckpt["classes"]) if ckpt else 0,
        "val_metrics": ckpt.get("metrics", {}) if ckpt else {},
    }


@app.get("/classes")
def classes() -> list[dict]:
    names = state["ckpt"]["classes"] if state["ckpt"] else [k for k in TREATMENTS if not k.startswith("_")]
    return [{"clase": c, "cultivo": CROPS[crop_of(c)], **info(c)} for c in names]


@app.post("/predict", response_model=Prediction)
async def predict(
    file: UploadFile = File(..., description="Foto de una hoja (JPG o PNG)"),
    cultivo: Cultivo | None = Query(None, description="Si se indica, solo se consideran las enfermedades "
                                                      "de ese cultivo"),
    explicar: bool = Query(False, description="Agrega un mapa Grad-CAM con las zonas que miró el modelo"),
) -> Prediction:
    if state["model"] is None:
        raise HTTPException(503, "El modelo no está cargado. Revisá MODEL_PATH.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"La imagen supera {MAX_UPLOAD_MB:.0f} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # respeta la rotación de la cámara del celular
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida")

    x = state["transform"](img).unsqueeze(0)
    mask = crop_mask(cultivo.value) if cultivo else None
    cam = None
    t0 = time.perf_counter()
    if explicar:
        logits, cam = logits_and_cam(state["model"], x, mask)
    else:
        with torch.no_grad():
            logits = state["model"](x)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs_all = logits.softmax(1)[0]
    aviso_cultivo = None
    if mask is not None:
        probs = (logits + mask).softmax(1)[0]
        mass = float(probs_all[mask == 0].sum())
        if mass < CROP_MISMATCH_THRESHOLD:
            aviso_cultivo = (f"La foto no parece de {CROPS[cultivo.value].lower()} "
                             f"(solo {100 * mass:.0f}% de coincidencia). Revisá el cultivo elegido.")
    else:
        probs = probs_all

    class_names = state["ckpt"]["classes"]
    top_p, top_i = probs.topk(3)
    best = class_names[top_i[0]]
    p_best = float(top_p[0])
    best_info = info(best)
    top3 = [Candidate(clase=class_names[i], nombre=info(class_names[i])["nombre"],
                      cultivo=CROPS[crop_of(class_names[i])], probabilidad=round(float(p), 4))
            for p, i in zip(top_p.tolist(), top_i.tolist())]

    low = p_best < CONFIDENCE_THRESHOLD
    gradcam_png = None
    if cam is not None:
        gradcam_png = base64.b64encode(overlay_png(img, cam, state["ckpt"]["img_size"])).decode("ascii")
    log.info("predict %s p=%.3f %.1fms cultivo=%s explicar=%s", best, p_best, elapsed_ms,
             cultivo.value if cultivo else "-", explicar)
    return Prediction(
        clase=best,
        nombre=best_info["nombre"],
        cultivo=CROPS[crop_of(best)],
        agente=best_info["agente"],
        severidad=best_info["severidad"],
        sintomas=best_info["sintomas"],
        probabilidad=round(p_best, 4),
        confianza_baja=low,
        sugerencia=("El modelo no está seguro. Sacá otra foto de cerca, con buena luz y una sola hoja, "
                    "o consultá a un técnico." if low else best_info["sugerencia"]),
        top3=top3,
        tiempo_inferencia_ms=round(elapsed_ms, 1),
        modelo=state["ckpt"]["arch"],
        aviso=TREATMENTS["_aviso"],
        cultivo_filtrado=CROPS[cultivo.value] if cultivo else None,
        aviso_cultivo=aviso_cultivo,
        gradcam_png=gradcam_png,
    )
