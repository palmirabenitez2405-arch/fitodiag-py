"""Definición única de las 15 clases del sistema.

Cada clase se nombra ``cultivo__enfermedad``. Este archivo es la fuente de verdad
para el script de preparación de datos, el entrenamiento, la API y el informe.
"""

from __future__ import annotations

# Orden alfabético (el mismo que usa torchvision.ImageFolder): el índice de cada
# clase es la salida del modelo. El checkpoint guarda además su propia lista.
CLASSES: list[str] = sorted([
    # Soja — MH-SoyaHealthVision (Shinde & Attar, 2024)
    "soja__mancha_marron",
    "soja__mosaico",
    "soja__ojo_de_rana",
    "soja__oruga",
    "soja__roya",
    "soja__sana",
    # Mandioca — Cassava Leaf Disease Classification (Kaggle, 2020)
    "mandioca__moteado_verde",
    "mandioca__bacteriosis",
    "mandioca__estria_marron",
    "mandioca__mosaico",
    "mandioca__sana",
    # Sésamo — Sesame Leaf Disease Dataset (Mendeley, 2025)
    "sesamo__amarillamiento",
    "sesamo__insectos",
    "sesamo__mancha_foliar",
    "sesamo__sana",
])

CROPS = {"soja": "Soja", "mandioca": "Mandioca", "sesamo": "Sésamo"}

# Etiquetas cortas para la matriz de confusión del informe (cabe en una columna IEEE).
SHORT_LABELS: dict[str, str] = {
    "mandioca__moteado_verde": "Ma-CGM",
    "mandioca__bacteriosis": "Ma-CBB",
    "mandioca__estria_marron": "Ma-CBSD",
    "mandioca__mosaico": "Ma-CMD",
    "mandioca__sana": "Ma-sana",
    "sesamo__amarillamiento": "Se-amar",
    "sesamo__insectos": "Se-insec",
    "sesamo__mancha_foliar": "Se-manch",
    "sesamo__sana": "Se-sana",
    "soja__mancha_marron": "So-septo",
    "soja__mosaico": "So-mosai",
    "soja__ojo_de_rana": "So-frog",
    "soja__oruga": "So-oruga",
    "soja__roya": "So-roya",
    "soja__sana": "So-sana",
}


def crop_of(class_name: str) -> str:
    return class_name.split("__", 1)[0]


def disease_of(class_name: str) -> str:
    return class_name.split("__", 1)[1]


# --- Mapeo de las etiquetas originales de cada dataset a nuestras clases -----

# Soja: nombres de carpeta / etiqueta del dataset MH-SoyaHealthVision.
# Se compara por palabras clave sobre el nombre normalizado (minúsculas, sin símbolos).
SOY_KEYWORDS: list[tuple[str, str]] = [
    ("semilooper", "soja__oruga"),
    ("caterpillar", "soja__oruga"),
    ("frog", "soja__ojo_de_rana"),
    ("brownspot", "soja__mancha_marron"),
    ("septoria", "soja__mancha_marron"),
    ("spectoria", "soja__mancha_marron"),  # así está escrito en el dataset original
    ("mosaic", "soja__mosaico"),
    ("rust", "soja__roya"),
    ("healthy", "soja__sana"),
]

# Mandioca: etiquetas numéricas del train.csv de Kaggle 2020.
CASSAVA_LABELS: dict[int, str] = {
    0: "mandioca__bacteriosis",    # Cassava Bacterial Blight (CBB)
    1: "mandioca__estria_marron",  # Cassava Brown Streak Disease (CBSD)
    2: "mandioca__moteado_verde",  # Cassava Green Mottle (CGM), virus
    3: "mandioca__mosaico",        # Cassava Mosaic Disease (CMD)
    4: "mandioca__sana",           # Healthy
}

# Sésamo: nombres de carpeta del dataset de Mendeley.
SESAME_KEYWORDS: list[tuple[str, str]] = [
    ("insect", "sesamo__insectos"),
    ("yellow", "sesamo__amarillamiento"),
    ("spot", "sesamo__mancha_foliar"),
    ("healthy", "sesamo__sana"),
]


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def match_keywords(path_parts: list[str], table: list[tuple[str, str]]) -> str | None:
    """Devuelve la clase cuyo keyword aparece en alguna carpeta de la ruta (de adentro hacia afuera)."""
    for part in reversed(path_parts):
        norm = normalize(part)
        for keyword, cls in table:
            if keyword in norm:
                return cls
    return None
