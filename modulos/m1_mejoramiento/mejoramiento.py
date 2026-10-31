"""Módulo 1 — Mejoramiento y reconstrucción.

Consigna: Pre-procesamiento (mejoramiento de imágenes; reconstrucción).
Anteproyecto: metodología 1 (análisis exploratorio) y 2 (mejoramiento); hipótesis 1.

Qué recibe : la imagen RGB float [0, 1] de un Caso.
Qué devuelve: mejorar(img) -> (img_mejorada, valida, mascara_pelos)
    img_mejorada   misma forma que la entrada, float [0, 1], sin pelos
    valida         bool (alto x ancho): False en las esquinas de la viñeta
    mascara_pelos  bool (alto x ancho): True donde se detectaron pelos

Etapas de mejorar():
  1. mascara_vineta   umbral + regiones oscuras conectadas al borde -> se excluyen
  2. quitar_pelos     cierre morfológico con elementos lineales en varias
                      orientaciones (o filtro de mediana, para comparar)
                      Los píxeles de pelo se "reconstruyen" a partir de sus vecinos.
Los métodos de quitar_pelos ("ninguno", "mediana", "cierre") son los que se
comparan en la hipótesis 1 (Dice de la segmentación en imágenes con pelos).
"""

import numpy as np

# ------------------------------ Parámetros ------------------------------
# Todos los números en un solo lugar. Si se cambia alguno, actualizar la tabla
# de parámetros del informe (sección 2) y regenerar las figuras del cuaderno.
# PENDIENTE (M1): fijar cada valor con el análisis exploratorio (cuaderno).

UMBRAL_VINETA = None        # brillo (V) por debajo del cual un píxel es "esquina oscura"
LARGO_ELEMENTO = None       # píxeles. Más largo que el ancho de un pelo, más corto que la lesión
ORIENTACIONES = 12          # cantidad de orientaciones del elemento estructurante lineal
UMBRAL_PELOS = None         # diferencia cierre - original a partir de la cual un píxel es pelo
TAMANO_MEDIANA = None       # lado de la ventana del filtro de mediana (sólo para comparar)

METODOS_PELOS = ("ninguno", "mediana", "cierre")


def mascara_vineta(img):
    """True donde la imagen es válida, False en las esquinas oscuras de la lente.

    PENDIENTE (M1): umbral sobre V + etiquetado de las regiones oscuras que
    tocan el borde de la imagen (skimage.measure.label / skimage.segmentation.clear_border).
    """
    raise NotImplementedError("M1: falta programar mascara_vineta()")


def quitar_pelos(img, metodo="cierre"):
    """Devuelve (img_sin_pelos, mascara_pelos) con el método indicado.

    "ninguno"  devuelve la imagen tal cual y una máscara vacía
    "mediana"  filtro de mediana por canal (la máscara puede quedar vacía)
    "cierre"   cierre por canal con elementos lineales en ORIENTACIONES direcciones;
               los píxeles donde el cierre sube mucho son pelo y se reemplazan
               (reconstrucción: valor del cierre o inpainting, cv2.inpaint)
    """
    if metodo not in METODOS_PELOS:
        raise ValueError(f"metodo tiene que ser uno de {METODOS_PELOS}")
    raise NotImplementedError("M1: falta programar quitar_pelos()")


def mejorar(img, metodo_pelos="cierre"):
    """La función que llaman los demás módulos. Ver la convención arriba."""
    raise NotImplementedError("M1: falta programar mejorar()")


def a_hsv(img):
    """Conversión a HSV (H, S y V en [0, 1]). Ya está hecha."""
    from skimage.color import rgb2hsv
    return rgb2hsv(img)
