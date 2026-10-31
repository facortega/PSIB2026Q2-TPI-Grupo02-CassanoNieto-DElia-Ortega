"""Módulo 2 — Compresión y descompresión.

Consigna: Pre-procesamiento (compresión y descompresión) — paso obligatorio.
Devolución del anteproyecto: la comparación completa del algoritmo sobre
imágenes comprimidas (hipótesis 3) queda como OBJETIVO DE MÁXIMA.

Por eso este módulo tiene dos niveles:
  Núcleo (obligatorio): comprimir y descomprimir con JPEG a varias calidades y
      medir cuánto se comprime (tasa de compresión) y cuánto se pierde
      (PSNR y SSIM [Wang et al., 2004]), dentro de la lesión y en toda la imagen.
  Máxima (si sobra tiempo): correr el algoritmo completo sobre las imágenes
      descomprimidas (procesar(caso, calidad_jpeg=q)) y ver cómo cambian el
      Dice, la sensibilidad y la especificidad.

Qué recibe : una imagen RGB float [0, 1].
Qué devuelve: comprimir_jpeg(img, calidad) -> (img_descomprimida, cantidad_de_bytes)
              con img_descomprimida de la MISMA forma, float [0, 1].
"""

import numpy as np

CALIDADES = (95, 75, 50, 25, 10)   # PENDIENTE (M2): revisar el rango con las primeras pruebas


def comprimir_jpeg(img, calidad):
    """Comprime en memoria con JPEG y descomprime.

    PENDIENTE (M2): PIL.Image.save sobre un io.BytesIO con quality=calidad;
    pasar a uint8 antes y volver a float [0, 1] después.
    """
    raise NotImplementedError("M2: falta programar comprimir_jpeg()")


def tasa_de_compresion(img, cantidad_de_bytes):
    """Bytes de la imagen sin comprimir (8 bits por canal) / bytes del JPEG."""
    raise NotImplementedError("M2: falta programar tasa_de_compresion()")


def psnr(original, reconstruida, mascara=None):
    """PSNR en dB (señal máxima = 1). Con `mascara`, sólo en esos píxeles. Idénticas -> inf."""
    raise NotImplementedError("M2: falta programar psnr()")


def ssim(original, reconstruida):
    """SSIM medio entre las dos imágenes (skimage.metrics.structural_similarity)."""
    raise NotImplementedError("M2: falta programar ssim()")
