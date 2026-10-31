"""Lesiones dermatoscópicas inventadas, con todo lo "verdadero" conocido.

Ya está hecho: no hace falta tocarlo. Sirve para probar cada módulo sin
esperar a PH2 ni al módulo anterior, y es lo que usa verificacion.py.

    img, info = generar_lesion(semilla=0)
    img                     imagen RGB float en [0, 1], forma (alto, ancho, 3)
    info["mascara"]         máscara verdadera de la lesión (bool)
    info["sin_pelos"]       la misma imagen sin pelos (para medir cuánto se recupera)
    info["mascara_pelos"]   dónde están los pelos (bool)
    info["mascara_vineta"]  True donde la imagen es válida (fuera de las esquinas negras)
    info["centro"]          (fila, columna) del centro de la lesión

Ojo: es una caricatura. Sirve para comprobar que el código hace lo que dice
(formas, tipos, que un círculo dé simétrico...), no para ajustar umbrales.
Los umbrales se eligen con las imágenes de entrenamiento de PH2.
"""

import numpy as np
from scipy import ndimage as ndi

ALTO, ANCHO = 280, 384   # la mitad de PH2 (560 x 768), para que las pruebas sean rápidas

# Colores aproximados en RGB [0, 1] (sólo para dibujar; NO son las referencias del criterio C)
PIEL = (0.86, 0.66, 0.58)
COLORES_DIBUJO = {
    "blanco": (0.92, 0.90, 0.90),
    "rojo": (0.75, 0.30, 0.30),
    "marron_claro": (0.62, 0.42, 0.30),
    "marron_oscuro": (0.36, 0.22, 0.15),
    "azul_grisaceo": (0.40, 0.45, 0.55),
    "negro": (0.10, 0.08, 0.08),
}


def _radio_irregular(theta, irregularidad, rng):
    """Factor que deforma el contorno: 1 + suma de armónicos al azar."""
    if irregularidad <= 0:
        return np.ones_like(theta)
    factor = np.ones_like(theta)
    for k in range(2, 9):
        amp = irregularidad * rng.uniform(0.3, 1.0) / k
        factor += amp * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    return factor


def mascara_forma(alto=ALTO, ancho=ANCHO, centro=None, ejes=(70, 50), angulo=0.0,
                  irregularidad=0.0, lobulo=0.0, semilla=0):
    """Máscara bool de una lesión.

    ejes          semiejes (filas, columnas) de la elipse base, en píxeles
    angulo        rotación en grados
    irregularidad 0 = elipse perfecta; 0,3 = borde bastante irregular
    lobulo        0 = nada; 0,5 = agrega un lóbulo de un solo lado (asimétrica)
    """
    rng = np.random.default_rng(semilla)
    if centro is None:
        centro = (alto / 2, ancho / 2)
    f, c = np.mgrid[0:alto, 0:ancho].astype(float)
    df, dc = f - centro[0], c - centro[1]
    a = np.deg2rad(angulo)
    u = dc * np.cos(a) + df * np.sin(a)      # eje "columnas" rotado
    v = -dc * np.sin(a) + df * np.cos(a)     # eje "filas" rotado
    theta = np.arctan2(v, u)
    r_norm = np.sqrt((u / ejes[1]) ** 2 + (v / ejes[0]) ** 2)
    mascara = r_norm <= _radio_irregular(theta, irregularidad, rng)
    if lobulo > 0:
        cf = centro[0] + 0.2 * ejes[0]
        cc = centro[1] + 0.9 * ejes[1]
        r = lobulo * min(ejes)
        mascara |= (f - cf) ** 2 + (c - cc) ** 2 <= r ** 2
    return mascara


def _dibujar_pelos(alto, ancho, cantidad, rng, grosor=2):
    """Curvas oscuras y finas que cruzan la imagen."""
    pelos = np.zeros((alto, ancho), bool)
    for _ in range(cantidad):
        f0, c0 = rng.uniform(0, alto), rng.uniform(0, ancho)
        ang = rng.uniform(0, np.pi)
        curv = rng.uniform(-0.004, 0.004)
        largo = rng.uniform(0.5, 1.2) * max(alto, ancho)
        s = np.linspace(-largo / 2, largo / 2, int(2 * largo))
        ff = f0 + s * np.sin(ang) + curv * s ** 2
        cc = c0 + s * np.cos(ang)
        ok = (ff >= 0) & (ff < alto) & (cc >= 0) & (cc < ancho)
        pelos[ff[ok].astype(int), cc[ok].astype(int)] = True
    if grosor > 1:
        pelos = ndi.binary_dilation(pelos, iterations=grosor - 1)
    return pelos


def mascara_vineta(alto=ALTO, ancho=ANCHO, radio=None):
    """True dentro del círculo de la lente, False en las esquinas oscuras."""
    if radio is None:
        radio = 0.42 * np.hypot(alto, ancho)   # corta las esquinas, como la lente
    f, c = np.mgrid[0:alto, 0:ancho]
    return (f - alto / 2) ** 2 + (c - ancho / 2) ** 2 <= radio ** 2


def generar_lesion(alto=ALTO, ancho=ANCHO, semilla=0, ejes=(70, 50), angulo=0.0,
                   irregularidad=0.15, lobulo=0.0, color_centro="marron_oscuro",
                   color_periferia="marron_claro", textura=0.0, pelos=0,
                   vineta=True, desenfoque_borde=1.5, ruido=0.01):
    """Imagen RGB float [0, 1] de una lesión inventada y sus "verdades" (ver docstring del módulo).

    textura  amplitud de un patrón en red dentro de la lesión (0 = lisa)
    pelos    cantidad de pelos
    """
    rng = np.random.default_rng(semilla)
    mascara = mascara_forma(alto, ancho, ejes=ejes, angulo=angulo,
                            irregularidad=irregularidad, lobulo=lobulo, semilla=semilla)

    # Distancia normalizada al borde: 0 en el borde, 1 en el punto más interno
    dist = ndi.distance_transform_edt(mascara)
    prof = dist / max(dist.max(), 1)

    piel = np.array(PIEL)[None, None, :] * (1 + 0.03 * rng.standard_normal((alto, ancho, 1)))
    c_centro = np.array(COLORES_DIBUJO[color_centro])
    c_perif = np.array(COLORES_DIBUJO[color_periferia])
    mezcla = np.clip(prof * 1.6, 0, 1)[..., None]
    lesion = c_perif * (1 - mezcla) + c_centro * mezcla

    if textura > 0:
        f, c = np.mgrid[0:alto, 0:ancho]
        red = (np.sin(f / 3.0) * np.sin(c / 3.0)) > 0.3
        lesion = lesion * (1 - textura * red[..., None])

    # Borde suave: la mezcla piel/lesión se desenfoca un poco
    alfa = ndi.gaussian_filter(mascara.astype(float), desenfoque_borde) if desenfoque_borde > 0 \
        else mascara.astype(float)
    img = piel * (1 - alfa[..., None]) + lesion * alfa[..., None]
    img = img + ruido * rng.standard_normal(img.shape)
    img = np.clip(img, 0, 1)

    sin_pelos = img.copy()
    m_pelos = _dibujar_pelos(alto, ancho, pelos, rng) if pelos > 0 else np.zeros((alto, ancho), bool)
    img[m_pelos] = np.array([0.12, 0.09, 0.08]) + 0.02 * rng.standard_normal((m_pelos.sum(), 3))

    m_vineta = mascara_vineta(alto, ancho) if vineta else np.ones((alto, ancho), bool)
    img[~m_vineta] *= 0.08
    sin_pelos[~m_vineta] *= 0.08

    info = {
        "mascara": mascara,
        "sin_pelos": np.clip(sin_pelos, 0, 1),
        "mascara_pelos": m_pelos & m_vineta,
        "mascara_vineta": m_vineta,
        "centro": (alto / 2, ancho / 2),
    }
    return np.clip(img, 0, 1), info
