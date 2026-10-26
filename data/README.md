# Datasheet del dataset FitoDiag-PY

Dataset unificado de **15 clases** de hojas de soja, mandioca y sésamo, construido a partir de tres datasets públicos. Las imágenes no se suben al repo: se regeneran con dos comandos.

```bash
bash scripts/download_data.sh          # baja los datos a data/raw/
python scripts/prepare_data.py         # limpia, unifica y divide en data/processed/
```

## Fuentes

| Cultivo | Dataset | Origen de las fotos | Licencia / términos | Cita |
| --- | --- | --- | --- | --- |
| Soja | MH-SoyaHealthVision (parte de hojas) | Maharashtra, India; campo | Ver la página del dataset en Mendeley Data | Shinde & Attar, 2024, doi:10.17632/hkbgh5s3b7.1 |
| Mandioca | Cassava Leaf Disease Classification | Uganda; fotos de agricultores con smartphone | Reglas de la competencia en Kaggle (se aceptan antes de descargar) | Makerere AI Lab, Kaggle 2020 |
| Sésamo | Sesame Leaf Disease Dataset | Pabna, Bangladesh; smartphone en campo | Ver la página del dataset en Mendeley Data | doi:10.17632/c64jt5gkzm.1 |

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
3. Del sésamo se usan **solo las imágenes originales**: las carpetas "augmented" del dataset se ignoran para que una misma hoja no quede en train y en test.
4. Cada clase se limita a 1.500 imágenes (submuestreo aleatorio, semilla 42). Afecta sobre todo al mosaico de la mandioca.
5. Las imágenes se redimensionan a 512 px de lado mayor (JPEG, calidad 92).
6. División estratificada 70 / 15 / 15 (train / val / test), semilla 42.

Los conteos de cada paso quedan en `data/processed/stats.json` y la lista completa de archivos en `data/processed/manifest.csv`.

## Conteos

La tabla se completa sola al correr `prepare_data.py` ("Originales" = antes de limpiar y del tope por clase).

<!-- CONTEOS:INICIO -->
_Pendiente: correr `python scripts/prepare_data.py`._
<!-- CONTEOS:FIN -->

## Limitaciones conocidas

- **Ninguna foto es de Paraguay.** Cambian el suelo, las variedades y la luz. El rendimiento con fotos locales puede ser menor que el medido en test.
- El mosaico (CMD) y la estría marrón (CBSD) de la mandioca no están reportados en América. Se mantienen porque el dataset las incluye, pero la API advierte al usuario.
- Las etiquetas de Kaggle 2020 tienen ruido conocido (algunas imágenes mal etiquetadas).
- Las fotos de soja y sésamo suelen mostrar una sola hoja; las de mandioca, la planta entera. El modelo podría aprender a reconocer el **tipo de foto** en vez del cultivo. Por eso se mide la tasa de "cultivo equivocado" en `evaluate.py`.
