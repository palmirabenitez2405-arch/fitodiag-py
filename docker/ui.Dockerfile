# Interfaz Streamlit. Contexto de build: raíz del repo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_URL=http://api:8000

WORKDIR /app
COPY requirements/ui.txt requirements/ui.txt
RUN pip install -r requirements/ui.txt
COPY ui/ ui/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false", "--server.maxUploadSize=10"]
