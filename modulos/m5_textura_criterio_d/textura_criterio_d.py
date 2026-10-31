"""Módulo 5 — Textura (GLCM) y criterio D adaptado.

Consigna: Extracción de características (análisis de texturas).
Anteproyecto: metodología 6.

Qué recibe : la imagen mejorada (se usa el canal V) y la máscara de la lesión.
Qué devuelve:
    atributos_glcm(gris, mascara) -> dict {atributo: valor} con los de ATRIBUTOS,
        calculados SÓLO con pares de píxeles dentro de la máscara y promediados
        en las cuatro direcciones (0°, 45°, 90°, 135°).
    elegir_atributo_d(tabla)               -> nombre del atributo que más cambia con D
    ajustar_umbrales_d(valores, d_derm)    -> umbrales (array creciente)
    estimar_d(valor, umbrales)             -> int, D estimado (cantidad de umbrales superados)
Lo de D se ajusta SOLO con las filas de entrenamiento (lo hace
herramientas/procesamiento_completo.completar_d).

Ojo con la máscara: graycomatrix no acepta máscaras. Una forma: cuantizar V en
NIVELES niveles (1..NIVELES), poner 0 fuera de la lesión, calcular la GLCM con
NIVELES+1 niveles y descartar la fila y la columna 0 antes de normalizar.
"""

import numpy as np

ATRIBUTOS = ("energia", "entropia", "contraste", "idm", "correlacion",
             "cluster_shade", "cluster_prominence")

# ------------------------------ Parámetros ------------------------------
NIVELES = 32                                   # PENDIENTE (M5): justificar (8, 16, 32, 64...)
DISTANCIA = 1                                  # píxeles
ANGULOS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def glcm(gris, mascara, distancia=DISTANCIA):
    """GLCM normalizada (NIVELES x NIVELES) promediada en ANGULOS, sólo dentro de la máscara."""
    raise NotImplementedError("M5: falta programar glcm()")


def atributos_glcm(gris, mascara):
    """Los siete atributos de ATRIBUTOS a partir de glcm(). Entropía en bits."""
    raise NotImplementedError("M5: falta programar atributos_glcm()")


def elegir_atributo_d(tabla):
    """Atributo (columna "glcm_<atributo>" de la tabla) que más cambia con D_derm.

    PENDIENTE (M5): definir "más cambia" (p. ej. |correlación de Spearman|) y
    dejarlo justificado en el cuaderno.
    """
    raise NotImplementedError("M5: falta programar elegir_atributo_d()")


def ajustar_umbrales_d(valores, d_derm):
    """Umbrales sobre el atributo que separan los valores de D (p. ej. puntos medios
    entre las medianas de cada D). Si el atributo BAJA con D, invertir el signo antes."""
    raise NotImplementedError("M5: falta programar ajustar_umbrales_d()")


def estimar_d(valor, umbrales):
    """D estimado = cantidad de umbrales que el valor supera (np.searchsorted)."""
    raise NotImplementedError("M5: falta programar estimar_d()")
