"""Módulo 1 — Mejoramiento y reconstrucción.

Consigna: Pre-procesamiento (mejoramiento de imágenes; reconstrucción).
Anteproyecto: metodología 1 (análisis exploratorio) y 2 (mejoramiento); hipótesis 1.

Qué recibe : la imagen RGB float [0, 1] de un Caso.
Qué devuelve: mejorar(img) -> (img_mejorada, valida, mascara_pelos)
    img_mejorada   misma forma que la entrada, float [0, 1], sin pelos
    valida         bool (alto x ancho): False en las esquinas de la viñeta
    mascara_pelos  bool (alto x ancho): True donde se detectaron pelos

Etapas de mejorar():
  1. mascara_vineta   perfil radial del brillo -> radio donde empieza la viñeta
  2. quitar_pelos     cierre morfológico con elementos lineales en varias
                      orientaciones (o filtro de mediana, para comparar)
                      Los píxeles de pelo se "reconstruyen" a partir de sus vecinos.
Los métodos de quitar_pelos ("ninguno", "mediana", "cierre") son los que se
comparan en la hipótesis 1 (Dice de la segmentación en imágenes con pelos).
"""

import cv2                                   # AGREGADO (M1): morfología rápida e inpainting
import numpy as np
from scipy import ndimage as ndi             # AGREGADO (M1)
from skimage import color, measure, morphology   # AGREGADO (M1)

# ------------------------------ Parámetros ------------------------------
# Todos los números en un solo lugar. Si se cambia alguno, actualizar la tabla
# de parámetros del informe (sección 2) y regenerar las figuras del cuaderno.
# AGREGADO (M1): valores elegidos con el exploratorio sobre ENTRENAMIENTO (ver cuaderno).

# Viñeta. DECIDIDO: no se usa un umbral fijo sobre V porque el brillo de las esquinas cambia
# mucho entre imágenes (V de las esquinas entre 0,13 y 0,77): un umbral que capture las esquinas de todas
# captura también partes oscuras de casi todas las lesiones. Además, en 50 imágenes la
# lesión toca el borde, así que "regiones oscuras conectadas al borde" se llevaría lesión.
# En su lugar se usa que la viñeta es CIRCULAR: se calcula la mediana de V en anillos
# alrededor del centro y la viñeta empieza donde esa mediana cae por debajo de
# UMBRAL_VINETA veces el anillo más brillante. La mediana del anillo no se deja
# engañar por una lesión que ocupa sólo una parte del anillo.
UMBRAL_VINETA = 0.6         # fracción del brillo del anillo más brillante
ANCHO_ANILLO = 4            # píxeles
RADIO_MINIMO = 0.80         # la viñeta nunca empieza antes de esta fracción de la semidiagonal
                            # (evita que una lesión enorme se tome como viñeta: IMD031, IMD065...)

# Pelos (cierre con elementos lineales = "black-hat" lineal, idea de DullRazor)
LARGO_ELEMENTO = 21         # píxeles. Más largo que el ancho de un pelo (3-8 px), más corto que la lesión
ORIENTACIONES = 12          # cada 15°
UMBRAL_PELOS = 0.10         # cierre - original (en gris [0, 1]) a partir del cual un píxel es candidato
LARGO_MIN_PELO = 60         # píxeles de esqueleto: un pelo es largo...
ANCHO_MAX_PELO = 8          # ...fino (área / largo)...
EJE_MIN_PELO = 80           # ...y alargado: eje mayor de la región >= 80 px. AGREGADO (M1): sin esta
                            # condición, la red de pigmento enmarañada pasa el filtro (tiene esqueleto
                            # largo, pero es compacta): en IMD200, sin pelos, se marcaba el 14 % de la
                            # lesión como pelo; con ella, el 1 %, y las imágenes con pelos no cambian.
RADIO_INPAINT = 5           # píxeles de vecindad para reconstruir (cv2.inpaint, método de Telea)
TAMANO_MEDIANA = 9          # lado de la ventana del filtro de mediana (sólo para comparar)

METODOS_PELOS = ("ninguno", "mediana", "cierre")


# ----------------------------- AGREGADO (M1) -----------------------------

def _distancia_al_centro(forma):
    alto, ancho = forma
    f, c = np.mgrid[0:alto, 0:ancho]
    return np.hypot(f - (alto - 1) / 2, c - (ancho - 1) / 2)


def radio_vineta(img):
    """Radio (px) donde empieza la viñeta, a partir del perfil radial de V. inf si no hay viñeta."""
    v = a_hsv(img)[..., 2]
    r = _distancia_al_centro(v.shape)
    anillo = (r // ANCHO_ANILLO).astype(int)
    mediana = ndi.median(v, labels=anillo + 1, index=np.arange(1, anillo.max() + 2))
    radios = np.arange(len(mediana)) * ANCHO_ANILLO
    oscuro = (mediana < UMBRAL_VINETA * np.max(mediana)) & (radios >= RADIO_MINIMO * r.max())
    return float(radios[np.argmax(oscuro)]) if oscuro.any() else np.inf


def _elementos_lineales(largo, cantidad):
    """Segmentos de `largo` px en `cantidad` orientaciones entre 0° y 180°."""
    elementos = []
    centro = largo // 2
    for angulo in np.linspace(0, np.pi, cantidad, endpoint=False):
        se = np.zeros((largo, largo), np.uint8)
        dx, dy = centro * np.cos(angulo), centro * np.sin(angulo)
        cv2.line(se, (round(centro - dx), round(centro - dy)), (round(centro + dx), round(centro + dy)), 1, 1)
        elementos.append(se)
    return elementos


def detectar_pelos(img):
    """Máscara de pelos: black-hat lineal + filtro de forma (largo, fino y alargado)."""
    gris = (color.rgb2gray(img) * 255).astype(np.uint8)
    cierre = np.max([cv2.morphologyEx(gris, cv2.MORPH_CLOSE, se)
                     for se in _elementos_lineales(LARGO_ELEMENTO, ORIENTACIONES)], axis=0)
    candidatos = (cierre.astype(float) - gris) / 255 > UMBRAL_PELOS
    etiquetas = measure.label(candidatos)
    pelos = np.zeros_like(candidatos)
    for region in measure.regionprops(etiquetas):
        largo = morphology.skeletonize(region.image).sum()
        if (largo >= LARGO_MIN_PELO and region.area / max(largo, 1) <= ANCHO_MAX_PELO
                and region.axis_major_length >= EJE_MIN_PELO):
            pelos[etiquetas == region.label] = True
    return ndi.binary_dilation(pelos, iterations=1)    # incluye el borde difuso del pelo


# ---------------------------------------------------------------------------

def mascara_vineta(img):
    """True donde la imagen es válida, False en las esquinas oscuras de la lente.

    AGREGADO (M1): círculo de radio radio_vineta(img) (ver Parámetros).
    """
    return _distancia_al_centro(img.shape[:2]) < radio_vineta(img)


def quitar_pelos(img, metodo="cierre"):
    """Devuelve (img_sin_pelos, mascara_pelos) con el método indicado.

    "ninguno"  devuelve la imagen tal cual y una máscara vacía
    "mediana"  filtro de mediana por canal (la máscara queda vacía: no detecta, suaviza todo)
    "cierre"   detecta los pelos con el cierre lineal (detectar_pelos) y reconstruye
               esos píxeles con inpainting (cv2.inpaint, Telea) a partir de sus vecinos
    """
    if metodo not in METODOS_PELOS:
        raise ValueError(f"metodo tiene que ser uno de {METODOS_PELOS}")
    vacia = np.zeros(img.shape[:2], bool)
    if metodo == "ninguno":
        return img.copy(), vacia
    if metodo == "mediana":
        filtrada = np.stack([ndi.median_filter(img[..., k], size=TAMANO_MEDIANA) for k in range(3)], axis=-1)
        return filtrada, vacia
    pelos = detectar_pelos(img)
    u8 = (img * 255).round().astype(np.uint8)
    reconstruida = cv2.inpaint(u8, pelos.astype(np.uint8), RADIO_INPAINT, cv2.INPAINT_TELEA)
    salida = img.copy()
    salida[pelos] = reconstruida[pelos].astype(float) / 255   # sólo cambian los píxeles de pelo
    return salida, pelos


def mejorar(img, metodo_pelos="cierre"):
    """La función que llaman los demás módulos. Ver la convención arriba."""
    valida = mascara_vineta(img)
    sin_pelos, pelos = quitar_pelos(img, metodo_pelos)
    return sin_pelos, valida, pelos & valida


def a_hsv(img):
    """Conversión a HSV (H, S y V en [0, 1]). Ya está hecha."""
    from skimage.color import rgb2hsv
    return rgb2hsv(img)
