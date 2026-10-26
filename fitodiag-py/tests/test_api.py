"""Pruebas de la API con un modelo de pesos aleatorios (no requiere entrenar)."""

import importlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fitodiag.classes import CLASSES
from fitodiag.model import build_model, save_checkpoint


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "best_model.pt"
    save_checkpoint(path, build_model("efficientnet_b0", len(CLASSES), pretrained=False),
                    "efficientnet_b0", CLASSES, 224)
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(path))
    import api.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    mp.undo()


def jpeg_bytes(size=(640, 480), color=(40, 140, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["n_classes"] == 15


def test_classes_have_treatments(client):
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 15
    assert all(c["sugerencia"] for c in data)


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("hoja.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["clase"] in CLASSES
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert len(body["top3"]) == 3
    assert body["top3"][0]["probabilidad"] >= body["top3"][1]["probabilidad"]
    assert body["tiempo_inferencia_ms"] > 0
    assert body["sugerencia"]


def test_predict_png(client):
    buf = io.BytesIO()
    Image.new("RGBA", (300, 300), (10, 200, 10, 255)).save(buf, format="PNG")
    r = client.post("/predict", files={"file": ("hoja.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("nota.txt", b"esto no es una imagen", "text/plain")})
    assert r.status_code == 400


def test_treatments_cover_all_classes():
    import json
    from pathlib import Path
    t = json.loads((Path(__file__).resolve().parents[1] / "api" / "treatments.json").read_text())
    assert set(CLASSES) <= set(t)
