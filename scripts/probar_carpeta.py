"""Prueba la API con una carpeta de fotos etiquetadas por nombre de archivo.

Cada archivo debe llamarse <clase>_<n>.jpg, por ejemplo soja__roya_1.jpg.
Cada foto se envía dos veces: sin filtro, y con el cultivo correcto + Grad-CAM.

Uso (con la API corriendo):
  python scripts/probar_carpeta.py --fotos ruta/a/imagenes_prueba --out pruebas

Salida:
  <out>/reporte.html      una ficha por foto: foto, mapa de calor, diagnóstico y top-3 con porcentajes
  <out>/resultados.csv    una fila por foto
  <out>/gradcam/<foto>.png
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageOps

SEVERITY_ICON = {"alta": "🔴", "media": "🟠", "baja": "🟡", "ninguna": "🟢"}


def thumb_b64(data: bytes, size: int = 320) -> str:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def bars(top3: list[dict], real: str) -> str:
    out = []
    for c in top3:
        pct = 100 * c["probabilidad"]
        cls = "bar ok" if c["clase"] == real else "bar"
        out.append(f'<div class="cand"><span>{html.escape(c["nombre"])}</span><b>{pct:.1f}%</b></div>'
                   f'<div class="track"><div class="{cls}" style="width:{pct:.1f}%"></div></div>')
    return "".join(out)


def card(r: dict, names: dict[str, str]) -> str:
    p1, p2 = r["resp"], r["resp_filtro"]
    ok = r["ok"]
    head = SEVERITY_ICON.get(p1["severidad"], "") if not p1["confianza_baja"] else "❔"
    unsure = (f'<div class="warn">No estoy seguro ({100 * p1["probabilidad"]:.0f}%)</div>'
              if p1["confianza_baja"] else "")
    filtro = (f'{html.escape(p2["nombre"])} · {100 * p2["probabilidad"]:.1f}%'
              f'{" · no seguro" if p2["confianza_baja"] else ""}')
    return f"""
<article class="card {'acierto' if ok else 'error'}">
  <header><span class="badge {'b-ok' if ok else 'b-err'}">{'✔ Acierto' if ok else '✘ Error'}</span>
    <span class="file">{html.escape(r['archivo'])}</span></header>
  <div class="real">Real: <b>{html.escape(names.get(r['real'], r['real']))}</b> ({html.escape(p1['cultivo'] if r['cultivo_ok'] else r['real'].split('__')[0])})</div>
  <div class="imgs">
    <figure><img src="data:image/jpeg;base64,{r['foto_b64']}" alt="Foto"><figcaption>Foto</figcaption></figure>
    <figure><img src="data:image/png;base64,{r['cam_b64']}" alt="Mapa de calor"><figcaption>Mapa de calor</figcaption></figure>
  </div>
  {unsure}
  <div class="pred">{head} {html.escape(p1['nombre'])} <span class="pct">{100 * p1['probabilidad']:.1f}%</span></div>
  <div class="sub">Otras posibilidades</div>
  {bars(p1['top3'], r['real'])}
  <div class="filtro">Con filtro «{html.escape(p2['cultivo_filtrado'] or '')}»: {filtro}</div>
  <div class="ms">{p1['tiempo_inferencia_ms']:.0f} ms · con mapa de calor {p2['tiempo_inferencia_ms']:.0f} ms</div>
</article>"""


def write_report(path: Path, rows: list[dict], names: dict[str, str]) -> None:
    n = len(rows)
    t = lambda k: sum(r[k] for r in rows)  # noqa: E731
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["real"]].append(r)
    class_rows = "".join(
        f"<tr><td>{html.escape(names.get(c, c))}</td><td>{html.escape(c.split('__')[0].capitalize())}</td>"
        f"<td class='num'>{sum(r['ok'] for r in rs)}/{len(rs)}</td></tr>" for c, rs in by_class.items())
    stats = [("Accuracy", f"{t('ok')}/{n}", f"{100 * t('ok') / n:.0f}%"),
             ("Correcta en top-3", f"{t('top3_ok')}/{n}", f"{100 * t('top3_ok') / n:.0f}%"),
             ("Cultivo correcto", f"{t('cultivo_ok')}/{n}", f"{100 * t('cultivo_ok') / n:.0f}%"),
             ("Con filtro por cultivo", f"{t('ok_filtro')}/{n}", f"{100 * t('ok_filtro') / n:.0f}%"),
             ("No estoy seguro", str(t("baja")), "respuestas"),
             ("Tiempo medio", f"{sum(r['ms'] for r in rows) / n:.0f} ms", "por foto")]
    tiles = "".join(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>"
                    for k, v, s in stats)
    cards = "".join(card(r, names) for r in rows)
    path.write_text(f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pruebas FitoDiag-PY</title>
<style>
:root {{ --bg:#f6f7f5; --card:#fff; --fg:#1d231f; --mut:#5d6660; --line:#dfe3df; --ok:#2f8a4c; --err:#c0392b;
        --warn-bg:#fff4d6; --warn-fg:#7a5a00; --track:#e7ebe7; --bar:#8aa392; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141815; --card:#1d221e; --fg:#e7ece8; --mut:#9aa59d;
        --line:#2e352f; --ok:#52b46f; --err:#e36a5c; --warn-bg:#3a3214; --warn-fg:#f1d27a; --track:#2b322c; --bar:#6c8473; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1200px; margin:0 auto; padding:24px 16px 48px }}
h1 {{ margin:0 0 4px; font-size:26px }} .lead {{ color:var(--mut); margin:0 0 20px }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:20px }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.tile .k {{ color:var(--mut); font-size:13px }} .tile .v {{ font-size:24px; font-weight:650 }} .tile .s {{ color:var(--mut); font-size:12px }}
details {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; margin-bottom:18px }}
summary {{ cursor:pointer; font-weight:600 }}
table {{ border-collapse:collapse; width:100%; margin-top:8px }} td {{ padding:4px 6px; border-top:1px solid var(--line) }} .num {{ text-align:right; font-variant-numeric:tabular-nums }}
.filters {{ display:flex; gap:8px; margin:0 0 14px; flex-wrap:wrap }}
.filters button {{ border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:999px; padding:6px 14px; cursor:pointer; font:inherit }}
.filters button[aria-pressed=true] {{ background:var(--fg); color:var(--bg) }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; border-top:4px solid var(--ok) }}
.card.error {{ border-top-color:var(--err) }}
.card header {{ display:flex; justify-content:space-between; gap:8px; align-items:center }}
.badge {{ font-size:12px; font-weight:650; padding:2px 8px; border-radius:999px; color:#fff }} .b-ok {{ background:var(--ok) }} .b-err {{ background:var(--err) }}
.file {{ color:var(--mut); font-size:12px; overflow-wrap:anywhere }}
.real {{ margin:6px 0 8px; font-size:14px }}
.imgs {{ display:grid; grid-template-columns:1fr 1fr; gap:8px }}
figure {{ margin:0 }} figure img {{ width:100%; aspect-ratio:1; object-fit:cover; border-radius:8px; display:block }}
figcaption {{ color:var(--mut); font-size:12px; text-align:center; margin-top:2px }}
.warn {{ background:var(--warn-bg); color:var(--warn-fg); border-radius:6px; padding:4px 8px; margin-top:10px; font-size:13px; font-weight:600 }}
.pred {{ font-size:18px; font-weight:650; margin:10px 0 6px }} .pct {{ color:var(--mut); font-weight:500 }}
.sub {{ color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-top:6px }}
.cand {{ display:flex; justify-content:space-between; font-size:13px; margin-top:6px }} .cand b {{ font-variant-numeric:tabular-nums }}
.track {{ height:7px; background:var(--track); border-radius:4px; overflow:hidden }} .bar {{ height:100%; background:var(--bar) }} .bar.ok {{ background:var(--ok) }}
.filtro {{ margin-top:10px; font-size:13px }} .ms {{ color:var(--mut); font-size:12px; margin-top:4px }}
</style></head><body><main>
<h1>Pruebas de FitoDiag-PY</h1>
<p class="lead">{n} fotos enviadas a la API · {datetime.now():%d/%m/%Y %H:%M}. En verde, la barra de la clase correcta.</p>
<section class="tiles">{tiles}</section>
<details><summary>Aciertos por clase</summary><table>{class_rows}</table></details>
<div class="filters" role="group" aria-label="Filtrar fichas">
  <button data-f="todas" aria-pressed="true">Todas ({n})</button>
  <button data-f="acierto" aria-pressed="false">Aciertos ({t('ok')})</button>
  <button data-f="error" aria-pressed="false">Errores ({n - t('ok')})</button>
</div>
<section class="grid">{cards}</section>
</main>
<script>
document.querySelectorAll('.filters button').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('.filters button').forEach(x => x.setAttribute('aria-pressed', x === b));
  document.querySelectorAll('.card').forEach(c => c.hidden = b.dataset.f !== 'todas' && !c.classList.contains(b.dataset.f));
}}));
</script></body></html>""", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fotos", required=True)
    ap.add_argument("--out", default="pruebas")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "gradcam").mkdir(parents=True, exist_ok=True)
    names = {c["clase"]: c["nombre"] for c in requests.get(f"{args.api}/classes", timeout=10).json()}
    rows = []
    for f in sorted(p for p in Path(args.fotos).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        real = f.stem.rsplit("_", 1)[0]
        crop = real.split("__")[0]
        data = f.read_bytes()
        files = {"file": (f.name, data, "image/jpeg")}
        r1 = requests.post(f"{args.api}/predict", params={"explicar": "true"}, files=files, timeout=60).json()
        r2 = requests.post(f"{args.api}/predict", params={"cultivo": crop, "explicar": "true"},
                           files=files, timeout=60).json()
        cam = r1.pop("gradcam_png")
        r2.pop("gradcam_png")
        (out / "gradcam" / f"{f.stem}.png").write_bytes(base64.b64decode(cam))
        rows.append({
            "archivo": f.name, "real": real,
            "pred": r1["clase"], "prob": r1["probabilidad"], "ok": r1["clase"] == real,
            "top3_ok": real in [c["clase"] for c in r1["top3"]], "baja": r1["confianza_baja"],
            "cultivo_ok": r1["clase"].split("__")[0] == crop, "ms": r1["tiempo_inferencia_ms"],
            "pred_filtro": r2["clase"], "prob_filtro": r2["probabilidad"], "ok_filtro": r2["clase"] == real,
            "baja_filtro": r2["confianza_baja"], "aviso_cultivo": r2["aviso_cultivo"] or "",
            "ms_filtro_gradcam": r2["tiempo_inferencia_ms"],
            "resp": r1, "resp_filtro": r2, "foto_b64": thumb_b64(data), "cam_b64": cam,
        })

    csv_fields = [k for k in rows[0] if k not in {"resp", "resp_filtro", "foto_b64", "cam_b64"}]
    with open(out / "resultados.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_report(out / "reporte.html", rows, names)

    n = len(rows)
    total = lambda k: sum(r[k] for r in rows)  # noqa: E731
    print(f"Fotos: {n} | accuracy {total('ok')}/{n} | top-3 {total('top3_ok')}/{n} | "
          f"cultivo correcto {total('cultivo_ok')}/{n} | con filtro {total('ok_filtro')}/{n} | "
          f"baja confianza {total('baja')}")
    print(f"Reporte: {(out / 'reporte.html').resolve()}")


if __name__ == "__main__":
    main()
