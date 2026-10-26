#!/usr/bin/env bash
# Descarga los tres datasets públicos en data/raw/.
# Uso: bash scripts/download_data.sh
#
# Requisitos:
#   - pip install datasets kaggle
#   - Token de Kaggle en ~/.kaggle/kaggle.json (Kaggle > Settings > Create New Token)
#   - Haber aceptado las reglas de la competencia en
#     https://www.kaggle.com/competitions/cassava-leaf-disease-classification/rules
set -euo pipefail

RAW=data/raw
mkdir -p "$RAW/soja" "$RAW/mandioca" "$RAW/sesamo"

echo "==> 1/3 Soja: MH-SoyaHealthVision (Hugging Face)"
python scripts/export_hf_soy.py --out "$RAW/soja"

echo "==> 2/3 Mandioca: Cassava Leaf Disease Classification (Kaggle 2020)"
if [ ! -f "$RAW/mandioca/train.csv" ]; then
  kaggle competitions download -c cassava-leaf-disease-classification -p "$RAW/mandioca"
  unzip -q "$RAW/mandioca/cassava-leaf-disease-classification.zip" "train.csv" "train_images/*" -d "$RAW/mandioca"
  rm "$RAW/mandioca/cassava-leaf-disease-classification.zip"
else
  echo "    ya descargado, se omite"
fi

echo "==> 3/3 Sésamo: Sesame Leaf Disease Dataset (Mendeley Data)"
if [ -z "$(ls -A "$RAW/sesamo" 2>/dev/null)" ]; then
  cat <<'EOF'
    Mendeley no permite descarga automática sin sesión. Pasos:
      1. Abrir https://data.mendeley.com/datasets/c64jt5gkzm/1
      2. "Download All" (ZIP)
      3. Descomprimir dentro de data/raw/sesamo/
    El script prepare_data.py usa SOLO las imágenes originales e ignora
    cualquier carpeta cuyo nombre contenga "augment" (evita fuga de datos).
EOF
else
  echo "    ya hay archivos en $RAW/sesamo, se omite"
fi

echo "Listo. Siguiente paso: python scripts/prepare_data.py"
