"""Módulo 3 — Segmentación de la lesión.

Consigna: Procesamiento (segmentación de imágenes).
Anteproyecto: metodología 4; evaluación con el coeficiente de Dice.

Qué recibe : la imagen mejorada y la máscara `valida` (salida del módulo 1).
Qué devuelve: segmentar(img, valida, metodo) -> mascara
    bool (alto x ancho), UNA sola región conexa (la que contiene el centro, o
    la más grande si el centro cae afuera), sin huecos.

Métodos que se comparan (se elige el de mayor Dice medio en ENTRENAMIENTO):
    "otsu"            umbral de Otsu sobre el canal elegido en el exploratorio
    "crecimiento"     region growing desde el centro de la imagen
    "kmeans"          k-means de 2 grupos sobre el color
    "chan_vese"       contorno activo sin bordes (skimage.segmentation.chan_vese)
Después de cualquiera: apertura (islas), cierre (huecos del borde), relleno de
huecos y etiquetado para quedarse con una región.
"""

import numpy as np

METODOS = ("otsu", "crecimiento", "kmeans", "chan_vese")
METODO_ELEGIDO = "otsu"   # PENDIENTE (M3): reemplazar por el de mayor Dice en entrenamiento

# ------------------------------ Parámetros ------------------------------
CANAL = None              # PENDIENTE (M3): canal con más contraste lesión/piel (exploratorio de M1)
RADIO_APERTURA = None     # píxeles
RADIO_CIERRE = None       # píxeles


def segmentar(img, valida=None, metodo=METODO_ELEGIDO):
    """Máscara de la lesión. Ver la convención arriba."""
    if metodo not in METODOS:
        raise ValueError(f"metodo tiene que ser uno de {METODOS}")
    raise NotImplementedError("M3: falta programar segmentar()")


def posprocesar(mascara_cruda):
    """Apertura + cierre + relleno + región que contiene el centro."""
    raise NotImplementedError("M3: falta programar posprocesar()")


def dice(a, b):
    """Coeficiente de Dice entre dos máscaras bool: 2|A∩B| / (|A|+|B|). Ambas vacías -> 1."""
    raise NotImplementedError("M3: falta programar dice()")
