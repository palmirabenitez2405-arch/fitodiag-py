"""Interfaz de FitoDiag-PY (Streamlit).

El agricultor saca o sube la foto de una hoja y recibe la enfermedad,
la probabilidad y una sugerencia de manejo. La UI solo habla con la API.

- Puede indicar el cultivo para que el modelo solo considere sus enfermedades.
- Puede ver el mapa Grad-CAM: las zonas de la hoja que más pesaron en el diagnóstico.
- Los diagnósticos de la sesión quedan en un historial descargable en CSV.

Ejecutar: streamlit run ui/app.py   (API_URL=http://localhost:8000 por defecto)
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from datetime import datetime

import requests
import streamlit as st
from PIL import Image, ImageOps

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

SEVERITY_COLOR = {"alta": "🔴", "media": "🟠", "baja": "🟡", "ninguna": "🟢"}
CROP_OPTIONS = {"No sé (detectar)": None, "Soja": "soja", "Mandioca": "mandioca", "Sésamo": "sesamo"}

st.set_page_config(page_title="FitoDiag-PY", page_icon="🌱", layout="centered")
st.session_state.setdefault("historial", [])
st.session_state.setdefault("resultados", {})  # cache: (hash de la foto, cultivo, explicar) -> respuesta


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


def diagnose(image_bytes: bytes, filename: str, mime: str, crop: str | None, explain: bool) -> dict:
    params = {"explicar": str(explain).lower()}
    if crop:
        params["cultivo"] = crop
    r = requests.post(f"{API_URL}/predict", params=params,
                      files={"file": (filename, image_bytes, mime)}, timeout=30)
    if r.status_code != 200:
        is_json = r.headers.get("content-type", "").startswith("application/json")
        detail = r.json().get("detail", r.text) if is_json else r.text
        raise RuntimeError(f"Error {r.status_code}: {detail}")
    return r.json()


def thumbnail(image_bytes: bytes, size: int = 96) -> bytes:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def history_csv(rows: list[dict]) -> bytes:
    fields = ["hora", "archivo", "cultivo_indicado", "diagnostico", "cultivo", "probabilidad_%",
              "confianza_baja", "tiempo_ms"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")  # con BOM para que Excel muestre bien los acentos


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
    for crop_name, names in by_crop.items():
        with st.expander(crop_name):
            st.markdown("\n".join(f"- {n}" for n in names))

    st.header("Consejos para la foto")
    st.markdown("- Una sola hoja, de cerca\n- Buena luz, sin sombra fuerte\n- Que se vean las manchas\n"
                "- Si sabés el cultivo, indicalo: mejora el acierto")

# --- Pantalla principal -----------------------------------------------------
st.title("🌱 FitoDiag-PY")
st.write("Diagnóstico de enfermedades en hojas de **soja**, **mandioca** y **sésamo**. "
         "Sacá o subí una foto de la hoja.")

tab_diag, tab_hist = st.tabs(["🔍 Diagnóstico", f"🕘 Historial ({len(st.session_state.historial)})"])

with tab_diag:
    col_crop, col_opt = st.columns([2, 1])
    with col_crop:
        crop_label = st.radio("¿De qué cultivo es la hoja?", list(CROP_OPTIONS), horizontal=True)
    with col_opt:
        explain = st.toggle("Mapa de calor", value=True,
                            help="Mapa Grad-CAM: en rojo, las zonas que más pesaron en el diagnóstico.")
    crop = CROP_OPTIONS[crop_label]

    src_upload, src_camera = st.tabs(["📁 Subir foto", "📷 Cámara"])
    image_bytes, filename, mime = None, "foto.jpg", "image/jpeg"
    with src_upload:
        up = st.file_uploader("Foto de la hoja (en el celular también permite usar la cámara)",
                              type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            image_bytes, filename, mime = up.getvalue(), up.name, up.type or "image/jpeg"
    with src_camera:
        cam = st.camera_input("Apuntá a la hoja")
        if cam is not None:
            image_bytes, filename, mime = cam.getvalue(), "camara.jpg", "image/jpeg"

    if image_bytes:
        # Streamlit vuelve a ejecutar el script en cada clic: se guarda la respuesta para no
        # repetir la consulta ni duplicar el historial con la misma foto y las mismas opciones.
        key = (hashlib.md5(image_bytes).hexdigest(), crop, explain)
        res = st.session_state.resultados.get(key)
        if res is None:
            with st.spinner("Analizando la hoja..."):
                try:
                    res = diagnose(image_bytes, filename, mime, crop, explain)
                except (RuntimeError, requests.RequestException) as e:
                    st.error(f"No se pudo obtener el diagnóstico. {e}")
                    st.stop()
            st.session_state.resultados[key] = res
            st.session_state.historial.insert(0, {
                "hora": datetime.now().strftime("%H:%M:%S"),
                "archivo": filename,
                "cultivo_indicado": crop_label,
                "diagnostico": res["nombre"],
                "cultivo": res["cultivo"],
                "probabilidad_%": round(100 * res["probabilidad"], 1),
                "confianza_baja": res["confianza_baja"],
                "tiempo_ms": res["tiempo_inferencia_ms"],
                "severidad": res["severidad"],
                "miniatura": thumbnail(image_bytes),
            })
            st.rerun()  # para que el contador de la pestaña Historial se actualice

        col_img, col_res = st.columns([1, 1.3])
        with col_img:
            st.image(image_bytes, caption="Imagen analizada", width="stretch")
            if res.get("gradcam_png"):
                st.image(base64.b64decode(res["gradcam_png"]), width="stretch",
                         caption="Grad-CAM: en rojo, las zonas que más pesaron")
        with col_res:
            if res.get("aviso_cultivo"):
                st.error(res["aviso_cultivo"])
            if res["confianza_baja"]:
                st.warning(f"**No estoy seguro** ({100 * res['probabilidad']:.0f}%). "
                           f"Lo más probable es: {res['nombre']}.")
            else:
                st.subheader(f"{SEVERITY_COLOR.get(res['severidad'], '')} {res['nombre']}")
            st.caption(f"Cultivo: {res['cultivo']} · Agente: {res['agente']}")
            st.progress(res["probabilidad"], text=f"Probabilidad: {100 * res['probabilidad']:.1f}%")
            filtro = f" · solo enfermedades de {res['cultivo_filtrado'].lower()}" if res.get("cultivo_filtrado") else ""
            st.caption(f"Tiempo de inferencia: {res['tiempo_inferencia_ms']:.0f} ms · "
                       f"modelo {res['modelo']}{filtro}")

        st.markdown("#### Qué hacer")
        st.info(res["sugerencia"])
        if res["sintomas"] and not res["confianza_baja"]:
            st.markdown(f"**Síntomas típicos:** {res['sintomas']}")

        st.markdown("#### Otras posibilidades")
        for cand in res["top3"]:
            st.write(f"{cand['nombre']} ({cand['cultivo']})")
            st.progress(cand["probabilidad"], text=f"{100 * cand['probabilidad']:.1f}%")

        st.caption(f"⚠️ {res['aviso']}")

with tab_hist:
    hist = st.session_state.historial
    if not hist:
        st.info("Todavía no hay diagnósticos en esta sesión.")
    else:
        n_low = sum(h["confianza_baja"] for h in hist)
        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnósticos", len(hist))
        c2.metric("Con baja confianza", n_low)
        c3.metric("Tiempo medio", f"{sum(h['tiempo_ms'] for h in hist) / len(hist):.0f} ms")

        for h in hist:
            c_img, c_txt = st.columns([1, 5])
            c_img.image(h["miniatura"], width=72)
            icon = "❔" if h["confianza_baja"] else SEVERITY_COLOR.get(h["severidad"], "")
            c_txt.markdown(f"**{icon} {h['diagnostico']}** ({h['cultivo']}) · {h['probabilidad_%']:.1f}%  \n"
                           f"{h['hora']} · {h['archivo']} · cultivo indicado: {h['cultivo_indicado']}")

        b1, b2 = st.columns(2)
        b1.download_button("⬇️ Descargar CSV", history_csv(hist), file_name="fitodiag_historial.csv",
                           mime="text/csv", width="stretch")
        if b2.button("🗑️ Borrar historial", width="stretch"):
            st.session_state.historial = []
            st.session_state.resultados = {}
            st.rerun()
