"""Interfaz de FitoDiag-PY (Streamlit).

El agricultor saca o sube la foto de una hoja y recibe la enfermedad,
la probabilidad y una sugerencia de manejo. La UI solo habla con la API.

Ejecutar: streamlit run ui/app.py   (API_URL=http://localhost:8000 por defecto)
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

SEVERITY_COLOR = {"alta": "🔴", "media": "🟠", "baja": "🟡", "ninguna": "🟢"}

st.set_page_config(page_title="FitoDiag-PY", page_icon="🌱", layout="centered")


@st.cache_data(ttl=30)
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=300)
def api_classes() -> list[dict]:
    try:
        return requests.get(f"{API_URL}/classes", timeout=5).json()
    except requests.RequestException:
        return []


def diagnose(image_bytes: bytes, filename: str, mime: str) -> dict:
    r = requests.post(f"{API_URL}/predict", files={"file": (filename, image_bytes, mime)}, timeout=30)
    if r.status_code != 200:
        is_json = r.headers.get("content-type", "").startswith("application/json")
        detail = r.json().get("detail", r.text) if is_json else r.text
        raise RuntimeError(f"Error {r.status_code}: {detail}")
    return r.json()


# --- Barra lateral ----------------------------------------------------------
with st.sidebar:
    st.header("Estado del sistema")
    health = api_health()
    if health is None:
        st.error(f"No se pudo conectar con la API en {API_URL}")
    elif not health["model_loaded"]:
        st.warning("La API está activa pero no tiene modelo cargado.")
    else:
        st.success(f"API activa · modelo {health['model']}")
        vm = health.get("val_metrics") or {}
        if "val_f1_macro" in vm:
            st.caption(f"F1 macro en validación: {100 * vm['val_f1_macro']:.1f}%")

    st.header("Qué reconoce")
    by_crop: dict[str, list[str]] = {}
    for c in api_classes():
        by_crop.setdefault(c["cultivo"], []).append(c["nombre"])
    for crop, names in by_crop.items():
        with st.expander(crop):
            st.markdown("\n".join(f"- {n}" for n in names))

    st.header("Consejos para la foto")
    st.markdown("- Una sola hoja, de cerca\n- Buena luz, sin sombra fuerte\n- Que se vean las manchas")

# --- Pantalla principal -----------------------------------------------------
st.title("🌱 FitoDiag-PY")
st.write("Diagnóstico de enfermedades en hojas de **soja**, **mandioca** y **sésamo**. "
         "Sacá o subí una foto de la hoja.")

tab_upload, tab_camera = st.tabs(["📁 Subir foto", "📷 Cámara"])
image_bytes, filename, mime = None, "foto.jpg", "image/jpeg"

with tab_upload:
    up = st.file_uploader("Foto de la hoja (en el celular también permite usar la cámara)",
                          type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        image_bytes, filename, mime = up.getvalue(), up.name, up.type or "image/jpeg"

with tab_camera:
    cam = st.camera_input("Apuntá a la hoja")
    if cam is not None:
        image_bytes, filename, mime = cam.getvalue(), "camara.jpg", "image/jpeg"

if image_bytes:
    col_img, col_res = st.columns([1, 1.3])
    with col_img:
        st.image(image_bytes, caption="Imagen analizada", width="stretch")
    with col_res:
        with st.spinner("Analizando la hoja..."):
            try:
                res = diagnose(image_bytes, filename, mime)
            except (RuntimeError, requests.RequestException) as e:
                st.error(f"No se pudo obtener el diagnóstico. {e}")
                st.stop()

        if res["confianza_baja"]:
            st.warning(f"**No estoy seguro** ({100 * res['probabilidad']:.0f}%). "
                       f"Lo más probable es: {res['nombre']}.")
        else:
            st.subheader(f"{SEVERITY_COLOR.get(res['severidad'], '')} {res['nombre']}")
        st.caption(f"Cultivo: {res['cultivo']} · Agente: {res['agente']}")
        st.progress(res["probabilidad"], text=f"Probabilidad: {100 * res['probabilidad']:.1f}%")
        st.caption(f"Tiempo de inferencia: {res['tiempo_inferencia_ms']:.0f} ms · modelo {res['modelo']}")

    st.markdown("#### Qué hacer")
    st.info(res["sugerencia"])
    if res["sintomas"] and not res["confianza_baja"]:
        st.markdown(f"**Síntomas típicos:** {res['sintomas']}")

    st.markdown("#### Otras posibilidades")
    for cand in res["top3"]:
        st.write(f"{cand['nombre']} ({cand['cultivo']})")
        st.progress(cand["probabilidad"], text=f"{100 * cand['probabilidad']:.1f}%")

    st.caption(f"⚠️ {res['aviso']}")
