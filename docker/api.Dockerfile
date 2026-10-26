# API de inferencia (CPU). Contexto de build: raíz del repo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_model.pt

WORKDIR /app

# PyTorch CPU (~200 MB en vez de ~2 GB de la versión con CUDA)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt

COPY src/ src/
COPY api/ api/

# Usuario sin privilegios
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
