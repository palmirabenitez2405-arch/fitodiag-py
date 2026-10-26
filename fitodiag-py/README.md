# FitoDiag-PY 🌱

Diagnóstico temprano de enfermedades foliares en **soja, mandioca y sésamo** a partir de una foto tomada con el celular.

Proyecto final de Palmira Benítez y Sebastián Villalba. El sistema es end-to-end: dataset → modelo → API REST → interfaz web → Docker.

![Arquitectura](docs/arquitectura.png)

## Qué hace

El agricultor saca una foto de una hoja. El sistema devuelve:

- la enfermedad detectada (15 clases en 3 cultivos);
- la probabilidad y las otras dos opciones más probables;
- una sugerencia de manejo;
- un aviso de "no estoy seguro" cuando la confianza es menor al 50%.

## Levantar todo con un solo comando

Requisitos: Docker Desktop (o Docker Engine + Compose v2) y el archivo `models/best_model.pt`.

```bash
docker compose up --build
```

| Servicio | URL |
| --- | --- |
| Interfaz (Streamlit) | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |

¿Todavía no hay modelo entrenado? Generen uno de prueba (pesos aleatorios, solo para verificar que todo funciona):

```bash
pip install torch torchvision timm
python scripts/make_dummy_model.py --out models/best_model.pt
```

### Usarlo desde el celular

1. Celular y laptop en la misma red wifi.
2. En el celular, abrir `http://<IP-de-la-laptop>:8501`.
3. Usar la pestaña **Subir foto**: en Android y iPhone ofrece sacar la foto con la cámara.

La pestaña **Cámara** del navegador necesita HTTPS cuando no es `localhost`. Para la demo con el celular, se puede abrir un túnel HTTPS gratis con `cloudflared tunnel --url http://localhost:8501`.

## Estructura

```
fitodiag-py/
├── api/                  API REST (FastAPI) + treatments.json
├── ui/                   Interfaz (Streamlit)
├── src/fitodiag/         Código compartido: clases, modelo, transformaciones
├── scripts/              Pipeline: descarga, preparación, entrenamiento, evaluación
├── notebooks/            Notebook para entrenar en Google Colab
├── data/README.md        Datasheet del dataset
├── models/               best_model.pt (Git LFS)
├── paper/                Informe IEEE en LaTeX
├── docker/               Dockerfiles de api y ui
├── tests/                Pruebas (pytest)
└── docker-compose.yml
```

## Pipeline completo (reproducir los resultados)

El entrenamiento se hace en Google Colab con GPU: `notebooks/entrenamiento_colab.ipynb` corre todos estos pasos.

```bash
pip install -r requirements/dev.txt

# 1. Datos (ver data/README.md)
bash scripts/download_data.sh
python scripts/prepare_data.py

# 2. Entrenamiento: dos arquitecturas + ablación sin técnicas contra el desbalance
python scripts/train.py --arch efficientnet_b0
python scripts/train.py --arch resnet50
python scripts/train.py --arch efficientnet_b0 --run-name effnet_baseline --no-augment --no-balance

# 3. Evaluación en test + tablas y figuras para el informe
for r in efficientnet_b0 resnet50 effnet_baseline; do python scripts/evaluate.py --run models/$r; done
python scripts/compare_runs.py --runs models/efficientnet_b0 models/resnet50 models/effnet_baseline \
    --best models/efficientnet_b0

# 4. Copiar el modelo elegido para la API
cp models/efficientnet_b0/best_model.pt models/best_model.pt
```

También hay atajos en el `Makefile` (`make prepare`, `make train-effnet`, `make up`, etc.).

## Modelo

| Aspecto | Elección |
| --- | --- |
| Arquitecturas | EfficientNet-B0 (4,0 M parámetros) y ResNet50 (23,5 M), preentrenadas en ImageNet (timm) |
| Fine-tuning | 2 épocas solo la cabeza + hasta 15 épocas todo el modelo; AdamW, cosine annealing |
| Desbalance | Data augmentation para fotos de campo + WeightedRandomSampler + label smoothing 0,1 |
| Selección | Early stopping sobre F1 macro de validación (paciencia 5) |
| Métricas | Accuracy, F1 macro y ponderado, top-3, accuracy por cultivo, tasa de cultivo equivocado, matriz de confusión, latencia en CPU |

## Resultados

Se completa con `results.md` después de entrenar (`compare_runs.py` lo genera).

| Modelo | Params (M) | Acc. (%) | F1 macro (%) | Top-3 (%) | CPU (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | 4,0 | — | — | — | — |
| ResNet50 | 23,5 | — | — | — | — |
| EfficientNet-B0 sin aug. ni balanceo | 4,0 | — | — | — | — |

## API

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/health` | Estado del servicio y del modelo |
| GET | `/classes` | Las 15 clases con síntomas y sugerencia |
| POST | `/predict` | Imagen (multipart, campo `file`) → diagnóstico |

```bash
curl -F "file=@hoja.jpg" http://localhost:8000/predict
```

Variables de entorno: `MODEL_PATH`, `CONFIDENCE_THRESHOLD` (0,5), `MAX_UPLOAD_MB` (10), `TORCH_THREADS` (2).

## Pruebas

```bash
pytest -q
```

Las pruebas crean un modelo de pesos aleatorios: no hace falta entrenar para correrlas.

## Flujo de trabajo en git

- Ramas: `main` (estable, lo que se presenta), `dev` (integración), `feat/<historia>`.
- Cada historia de usuario es un issue (plantilla en `.github/ISSUE_TEMPLATE`). Los commits la referencian: `feat(api): endpoint predict (#5)`.
- Prefijos: `feat`, `fix`, `docs`, `test`, `exp` (experimentos de entrenamiento), `chore`.
- El modelo se versiona con Git LFS: `git lfs install && git lfs track "*.pt"` (ya está en `.gitattributes`).

## Limitaciones

Ninguna imagen del dataset es de Paraguay. El modelo no se validó con fotos locales y puede rendir menos en el campo. Las sugerencias de manejo son orientativas y no reemplazan a un ingeniero agrónomo ni las recomendaciones del SENAVE. Ver `data/README.md`.

## Autores

Palmira Benítez · Sebastián Villalba
