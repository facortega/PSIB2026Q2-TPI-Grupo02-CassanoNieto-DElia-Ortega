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

import io                                             # AGREGADO (M2): archivo JPEG en memoria

import numpy as np
from PIL import Image                                 # AGREGADO (M2): codificador JPEG
from skimage.metrics import structural_similarity     # AGREGADO (M2)

# AGREGADO (M2): calidades de JPEG (escala de 1 a 100 de la librería: no es un porcentaje
# de tamaño). 95 es casi sin pérdida visible; 10 ya muestra bloques de 8x8 claramente.
CALIDADES = (95, 75, 50, 25, 10)


def comprimir_jpeg(img, calidad):
    """Comprime en memoria con JPEG y descomprime.

    AGREGADO (M2): la imagen se pasa a uint8 (8 bits por canal, como el .bmp original),
    se codifica en un buffer en memoria (no se escribe ningún archivo), se mide su
    tamaño en bytes y se decodifica de vuelta a float [0, 1].
    PIL usa por defecto submuestreo de croma 4:2:0 (el color se guarda a la mitad de
    resolución en cada eje), que es lo estándar en JPEG.
    """
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(u8).save(buffer, format="JPEG", quality=int(calidad))
    cantidad_de_bytes = buffer.tell()
    buffer.seek(0)
    reconstruida = np.asarray(Image.open(buffer).convert("RGB")).astype(float) / 255
    return reconstruida, cantidad_de_bytes


def tasa_de_compresion(img, cantidad_de_bytes):
    """Bytes de la imagen sin comprimir (8 bits por canal) / bytes del JPEG.

    AGREGADO (M2): alto x ancho x 3 canales x 1 byte. Una tasa de 20 significa que el
    JPEG ocupa 20 veces menos que la imagen sin comprimir.
    """
    return img.shape[0] * img.shape[1] * 3 / cantidad_de_bytes


def psnr(original, reconstruida, mascara=None):
    """PSNR en dB (señal máxima = 1). Con `mascara`, sólo en esos píxeles. Idénticas -> inf.

    AGREGADO (M2): PSNR = 10 log10(1 / MSE), con el MSE promediado sobre los tres canales.
    """
    error = (original - reconstruida) ** 2
    if mascara is not None:
        error = error[mascara]
    mse = float(np.mean(error))
    return np.inf if mse == 0 else 10 * np.log10(1 / mse)


def ssim(original, reconstruida):
    """SSIM medio entre las dos imágenes (skimage.metrics.structural_similarity).

    AGREGADO (M2): se calcula por canal y se promedia (channel_axis=2), con rango 1.
    """
    return float(structural_similarity(original, reconstruida, channel_axis=2, data_range=1.0))
