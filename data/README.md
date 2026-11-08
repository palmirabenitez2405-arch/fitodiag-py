# Datasheet del dataset FitoDiag-PY

Dataset unificado de **15 clases** de hojas de soja, mandioca y sésamo, construido a partir de tres datasets públicos. Las imágenes no se suben al repo: se regeneran con dos comandos.

```bash
bash scripts/download_data.sh          # baja los datos a data/raw/
python scripts/prepare_data.py         # limpia, unifica y divide en data/processed/
```

## Fuentes

| Cultivo | Dataset | Origen de las fotos | Licencia / términos | Cita |
| --- | --- | --- | --- | --- |
| Soja | [MH-SoyaHealthVision](https://data.mendeley.com/datasets/hkbgh5s3b7/1) (parte de hojas; descargado del espejo en [Hugging Face](https://huggingface.co/datasets/Project-AgML/MH_SoyaHealthVision_disease_classification_leaf)) | Maharashtra, India; campo | Ver la página del dataset en Mendeley Data | Shinde & Attar, 2024, doi:10.17632/hkbgh5s3b7.1 |
| Mandioca | [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | Uganda; fotos de agricultores con smartphone | Reglas de la competencia en Kaggle (se aceptan antes de descargar) | Makerere AI Lab, Kaggle 2020 |
| Sésamo | [Sesame Leaf Disease Dataset](https://data.mendeley.com/datasets/c64jt5gkzm/1) (archivo `processed.zip`, 200 MB) | Pabna, Bangladesh; smartphone en campo | Ver la página del dataset en Mendeley Data | doi:10.17632/c64jt5gkzm.1 |

Anoten en esta tabla la licencia exacta que figura en cada página al descargar: el informe debe citarla.

## Clases

| Clase | Nombre en español | Etiqueta original |
| --- | --- | --- |
| `soja__roya` | Roya asiática | Rust |
| `soja__mosaico` | Mosaico | Mosaic |
| `soja__ojo_de_rana` | Mancha ojo de rana | Frog_Leaf_Eye |
| `soja__mancha_marron` | Mancha marrón (Septoria) | Spectoria_Brown_Spot |
| `soja__oruga` | Daño por orugas | Caterpillar_Semilooper_Pest |
| `soja__sana` | Sana | Healthy |
| `mandioca__bacteriosis` | Bacteriosis (CBB) | 0 |
| `mandioca__estria_marron` | Estría marrón (CBSD) | 1 |
| `mandioca__moteado_verde` | Moteado verde (CGM) | 2 |
| `mandioca__mosaico` | Mosaico (CMD) | 3 |
| `mandioca__sana` | Sana | 4 |
| `sesamo__mancha_foliar` | Mancha foliar | Leaf Spot Disease |
| `sesamo__amarillamiento` | Amarillamiento | Yellowing Leaf Syndrome |
| `sesamo__insectos` | Daño por insectos | Insect Leaf Damage |
| `sesamo__sana` | Sana | Healthy Leaf |

## Limpieza aplicada (`scripts/prepare_data.py`)

1. Se descartan imágenes ilegibles y las de menos de 64 px de lado.
2. Se descartan duplicados exactos (MD5 del archivo).
3. Del sésamo se usan **solo las imágenes originales** (archivo `processed.zip` de Mendeley: 3.540 fotos). No se usan `Augmentation.zip` ni `original.zip` (6,7 GB, fotos en resolución original), para que una misma hoja no quede en train y en test.
4. Cada clase se limita a 1.500 imágenes (submuestreo aleatorio, semilla 42). Afecta sobre todo al mosaico de la mandioca.
5. Las imágenes se redimensionan a 512 px de lado mayor (JPEG, calidad 92).
6. División estratificada 70 / 15 / 15 (train / val / test), semilla 42.

Los conteos de cada paso quedan en `data/processed/stats.json` y la lista completa de archivos (con la ruta original de cada imagen) en `data/processed/manifest.csv`. Una copia del dataset procesado está en el Drive del equipo (`fitodiag/processed.zip`).

## Conteos

La tabla se completa sola al correr `prepare_data.py` ("Originales" = antes de limpiar y del tope por clase).

<!-- CONTEOS:INICIO -->
| Clase | Originales | Train | Val | Test |
| --- | ---: | ---: | ---: | ---: |
| `mandioca__bacteriosis` | 1.087 | 761 | 163 | 163 |
| `mandioca__estria_marron` | 2.189 | 1.050 | 225 | 225 |
| `mandioca__mosaico` | 13.158 | 1.050 | 225 | 225 |
| `mandioca__moteado_verde` | 2.386 | 1.050 | 225 | 225 |
| `mandioca__sana` | 2.577 | 1.050 | 225 | 225 |
| `sesamo__amarillamiento` | 894 | 626 | 134 | 134 |
| `sesamo__insectos` | 724 | 506 | 109 | 109 |
| `sesamo__mancha_foliar` | 587 | 411 | 88 | 88 |
| `sesamo__sana` | 1.335 | 935 | 200 | 200 |
| `soja__mancha_marron` | 268 | 188 | 40 | 40 |
| `soja__mosaico` | 707 | 491 | 105 | 105 |
| `soja__ojo_de_rana` | 169 | 119 | 25 | 25 |
| `soja__oruga` | 582 | 408 | 87 | 87 |
| `soja__roya` | 852 | 591 | 127 | 127 |
| `soja__sana` | 204 | 142 | 31 | 31 |
| **Total** | **27.719** | **9.378** | **2.009** | **2.009** |

Resultado de la limpieza (corrida del 26/10/2026 en Google Colab, semilla 42):

- Imágenes encontradas en las tres fuentes: 27.719 (soja 2.782, mandioca 21.397, sésamo 3.540).
- Descartadas por duplicado exacto: 13.
- Descartadas por el tope de 1.500 por clase: 14.310 (casi todas del mosaico de la mandioca, que tenía 13.158).
- Dataset final: **13.396 imágenes** (9.378 train / 2.009 val / 2.009 test).
- Desbalance resultante: la clase más chica (`soja__ojo_de_rana`, 169) tiene unas 9 veces menos imágenes que la más grande (1.500).
<!-- CONTEOS:FIN -->

## Limitaciones conocidas

- **Ninguna foto es de Paraguay.** Cambian el suelo, las variedades y la luz. El rendimiento con fotos locales puede ser menor que el medido en test.
- El mosaico (CMD) y la estría marrón (CBSD) de la mandioca no están reportados en América. Se mantienen porque el dataset las incluye, pero la API advierte al usuario.
- Las etiquetas de Kaggle 2020 tienen ruido conocido (algunas imágenes mal etiquetadas).
- Las fotos de soja y sésamo suelen mostrar una sola hoja; las de mandioca, la planta entera. El modelo podría aprender a reconocer el **tipo de foto** en vez del cultivo. Por eso se mide la tasa de "cultivo equivocado" en `evaluate.py`.
