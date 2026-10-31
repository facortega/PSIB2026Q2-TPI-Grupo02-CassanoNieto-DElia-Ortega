"""Módulo 4 — Criterios A, B y C de la regla ABCD (+ circularidad).

Consigna: Extracción de características.
Anteproyecto: metodología 5; A y C se comparan con el dermatólogo (κ de Cohen).

Qué recibe : la imagen mejorada y la máscara de la lesión (módulo 3).
Qué devuelve:
    asimetria(mascara)        -> int 0, 1 o 2 (ejes asimétricos)
    borde(img, mascara)       -> int 0 a 8 (sectores con borde abrupto)
    colores(img, mascara)     -> lista de nombres de COLORES presentes (C = len, 1 a 6)
    circularidad(mascara)     -> 4πA/P², 1 para un círculo y menos para formas irregulares
    criterios(img, mascara)   -> dict con todo lo anterior (lo usa el procesamiento completo)
Además, para las figuras y la interfaz:
    fracciones_asimetria(mascara)  -> fracción no superpuesta en cada eje + ejes
    gradiente_por_sector(img, mascara) -> gradiente medio en cada uno de los 8 sectores
    mapa_de_colores(img, mascara)  -> índice del color asignado a cada píxel (-1 fuera)
"""

import numpy as np

from modulos.m0_carga_de_datos.carga_de_datos import COLORES

# ------------------------------ Parámetros ------------------------------
# PENDIENTE (M4): los umbrales se eligen con las imágenes de ENTRENAMIENTO para
# coincidir lo más posible con el dermatólogo. Documentar cómo en el cuaderno.
UMBRAL_ASIMETRIA = None     # fracción de área no superpuesta para que un eje sea asimétrico
SECTORES = 8
UMBRAL_BORDE = None         # gradiente medio a partir del cual un sector es "abrupto"
FRACCION_MIN_COLOR = None   # fracción mínima de la lesión para que un color cuente
REFERENCIAS_HSV = {c: None for c in COLORES}   # (H, S, V) de cada color, de las máscaras de PH2


def fracciones_asimetria(mascara):
    """Refleja la máscara sobre sus dos ejes principales (centroide + momentos).

    Devuelve (fraccion_eje_mayor, fraccion_eje_menor, angulo_en_grados).
    """
    raise NotImplementedError("M4: falta programar fracciones_asimetria()")


def asimetria(mascara):
    raise NotImplementedError("M4: falta programar asimetria()")


def gradiente_por_sector(img, mascara):
    """Sobel sobre V; gradiente medio sobre el contorno en cada uno de los 8 sectores."""
    raise NotImplementedError("M4: falta programar gradiente_por_sector()")


def borde(img, mascara):
    raise NotImplementedError("M4: falta programar borde()")


def mapa_de_colores(img, mascara):
    """Cada píxel de la lesión al color de referencia más cercano en HSV (H es circular)."""
    raise NotImplementedError("M4: falta programar mapa_de_colores()")


def colores(img, mascara):
    raise NotImplementedError("M4: falta programar colores()")


def circularidad(mascara):
    """4πA/P² (skimage.measure.regionprops: area y perimeter)."""
    raise NotImplementedError("M4: falta programar circularidad()")


# ----------------- Ya está hecho: no hace falta tocarlo -----------------

def criterios(img, mascara):
    cols = colores(img, mascara)
    return {
        "A": asimetria(mascara),
        "B": borde(img, mascara),
        "C": len(cols),
        "colores": cols,
        "circularidad": circularidad(mascara),
    }
