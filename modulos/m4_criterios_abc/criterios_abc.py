"""Módulo 4 — Criterios A, B y C de la regla ABCD (+ circularidad).

Qué recibe : la imagen mejorada y la máscara de la lesión (del módulo 3).
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

Los umbrales y las referencias de color se eligieron con las imágenes de entrenamiento
en el cuaderno (criterios_abc.ipynb), midiendo sobre la máscara de M3.
"""

import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, io, measure

from modulos.m0_carga_de_datos.carga_de_datos import CARPETA_DATOS, CARPETA_IMAGENES, COLORES
from modulos.m1_mejoramiento import mejoramiento

# ------------------------------ Parámetros ------------------------------
# Elegidos con las imágenes de ENTRENAMIENTO (ver el cuaderno).

# A: un eje es asimétrico si la fracción de la lesión que no se superpone con su reflejo
# supera este valor. Es el que da el κ más alto con el dermatólogo.
UMBRAL_ASIMETRIA = 0.11

# B
SECTORES = 8
MEDIDAS_BORDE = ("crudo", "normalizado")
MEDIDA_BORDE = "crudo"          # "crudo" = gradiente; "normalizado" = gradiente / salto de brillo
UMBRAL_BORDE = 0.0513           # en las unidades de MEDIDA_BORDE; el que mejor separa comunes de sospechosas
SIGMA_BORDE = 1.0               # px. Suavizado antes de Sobel, para que el ruido de la piel no haga picos
VECINDAD_BORDE = 5              # px. El pico de gradiente se busca hasta esta distancia del contorno
ANILLO_SALTO = (5, 20)          # px. Distancias al contorno, adentro y afuera, donde se mide el brillo
                                # de la lesión y de la piel para calcular el salto
SALTO_MINIMO = 0.02             # evita dividir por casi cero si lesión y piel brillan casi igual

# C
FRACCION_MIN_COLOR = 0.05       # fracción mínima de la lesión para que un color cuente
ANILLO_PIEL = (10, 30)          # px. La piel de cada imagen sale del anillo a esta distancia de la lesión
ESCALA_RELATIVA = 0.5           # brillo en que queda la piel después de dividir por ella (gris medio)
PARECIDO_A_PIEL = -2            # valor del mapa de colores para los píxeles que se parecen más a la piel
# (H, S, V) típico de cada color en la imagen relativa a la piel: mediana de las zonas de color
# de PH2 (y de nuestras zonas rojas) en entrenamiento. Se copian del cuaderno, sección 2.
REFERENCIAS_HSV = {"blanco": (0.63, 0.152, 0.541), "rojo": (0.976, 0.212, 0.492), "marron_claro": (0.083, 0.249, 0.452),
                   "marron_oscuro": (0.06, 0.242, 0.3), "azul_grisaceo": (0.965, 0.069, 0.285),
                   "negro": (0.038, 0.065, 0.254)}

# Zonas rojas marcadas a mano por nosotros (PH2 no trae máscaras de rojo). Ver marcar_rojo.py
CARPETA_ANOTACIONES_ROJO = CARPETA_DATOS / "anotaciones_rojo"


# ------------------------------- Asimetría -------------------------------

def _ejes(mascara):
    """Centroide (fila, columna) y dirección (dc, df) del eje mayor.

    Los ejes principales salen de los momentos de segundo orden: son las direcciones de
    la matriz de covarianza de las coordenadas de los píxeles. El eje mayor es la de
    mayor varianza (por donde la lesión es más larga).
    """
    f, c = np.nonzero(mascara)
    centro = np.array([f.mean(), c.mean()])
    valores, vectores = np.linalg.eigh(np.cov(np.vstack([c - centro[1], f - centro[0]])))
    dc, df = vectores[:, np.argmax(valores)]
    return centro, dc, df


def _reflejar(mascara, centro, dc, df):
    """La máscara reflejada sobre la recta que pasa por `centro` con dirección (dc, df)."""
    filas, columnas = np.indices(mascara.shape)
    y, x = filas - centro[0], columnas - centro[1]
    a_lo_largo = x * dc + y * df
    # Reflejo de un punto p sobre la recta de dirección u: 2 (p·u) u - p. Como reflejar dos
    # veces deja todo igual, el píxel p de la máscara reflejada vale lo que vale la máscara en su reflejo.
    yr, xr = 2 * a_lo_largo * df - y, 2 * a_lo_largo * dc - x
    return ndi.map_coordinates(mascara.astype(np.uint8), [yr + centro[0], xr + centro[1]],
                               order=0, cval=0).astype(bool)


def reflejo(mascara, eje="mayor"):
    """La máscara reflejada sobre su eje "mayor" o "menor" (para las figuras)."""
    mascara = np.asarray(mascara, bool)
    centro, dc, df = _ejes(mascara)
    return _reflejar(mascara, centro, dc, df) if eje == "mayor" else _reflejar(mascara, centro, -df, dc)


def fracciones_asimetria(mascara):
    """Refleja la máscara sobre sus dos ejes principales (centroide + momentos).

    Devuelve (fraccion_eje_mayor, fraccion_eje_menor, angulo_en_grados): la fracción de la
    lesión que queda afuera de su reflejo en cada eje (0 = simétrica) y el ángulo del eje
    mayor medido desde el eje de las columnas (como lo dibuja la interfaz).
    Si la lesión toca el borde de la foto, lo que se refleja afuera de la imagen cuenta como
    no superpuesto: se mide la forma que se ve.
    """
    mascara = np.asarray(mascara, bool)
    centro, dc, df = _ejes(mascara)
    fracciones = [float(np.sum(mascara & ~_reflejar(mascara, centro, ux, uy)) / mascara.sum())
                  for ux, uy in ((dc, df), (-df, dc))]               # eje mayor y su perpendicular
    return fracciones[0], fracciones[1], float(np.degrees(np.arctan2(df, dc)))


def contar_ejes_asimetricos(f_mayor, f_menor, umbral=None):
    """A = cantidad de ejes cuya fracción supera el umbral (anda con números o con arrays)."""
    umbral = UMBRAL_ASIMETRIA if umbral is None else umbral
    return (np.asarray(f_mayor) > umbral).astype(int) + (np.asarray(f_menor) > umbral).astype(int)


def asimetria(mascara, umbral=None):
    f_mayor, f_menor, _ = fracciones_asimetria(mascara)
    return int(contar_ejes_asimetricos(f_mayor, f_menor, umbral))


# --------------------------------- Borde ---------------------------------

def _sector(filas, columnas, centro, dc, df):
    """Sector (0 a SECTORES-1) de cada punto según su ángulo alrededor del centroide.

    El ángulo se cuenta desde el eje mayor, así el resultado no depende de cómo esté girada la foto.
    """
    angulo = np.arctan2(filas - centro[0], columnas - centro[1]) - np.arctan2(df, dc)
    return (np.mod(angulo, 2 * np.pi) // (2 * np.pi / SECTORES)).astype(int)


def gradiente_por_sector(img, mascara, medida=None):
    """Sobel sobre V; gradiente medio sobre el contorno en cada uno de los 8 sectores.

    En cada punto del contorno se toma el gradiente más alto a menos de VECINDAD_BORDE px,
    por si el contorno de M3 quedó un poco corrido del borde real. Con medida="normalizado"
    ese gradiente se divide por el salto de brillo entre la piel y la lesión del sector: así
    se mide qué tan rápido cambia el brillo y no cuánto (una lesión clara con borde nítido
    tiene poco gradiente sólo porque contrasta poco).

    Los puntos del contorno pegados al marco de la foto o a la viñeta no se cuentan: ahí la
    lesión no termina, la cortó la foto. Un sector sin puntos queda en NaN.
    """
    medida = MEDIDA_BORDE if medida is None else medida
    if medida not in MEDIDAS_BORDE:
        raise ValueError(f"medida tiene que ser una de {MEDIDAS_BORDE}")
    mascara = np.asarray(mascara, bool)
    v = mejoramiento.a_hsv(img)[..., 2]
    valida = mejoramiento.mascara_vineta(img)
    pico = ndi.maximum_filter(filters.sobel(filters.gaussian(v, SIGMA_BORDE)), size=2 * VECINDAD_BORDE + 1)

    f, c, sector = contorno_por_sector(mascara, valida)
    suma = np.bincount(sector, weights=pico[f, c], minlength=SECTORES)
    cantidad = np.bincount(sector, minlength=SECTORES)
    with np.errstate(invalid="ignore", divide="ignore"):
        gradiente = suma / cantidad                                   # NaN si el sector no tiene contorno
    if medida == "crudo":
        return gradiente
    return gradiente / _salto_por_sector(v, mascara, valida, *_ejes(mascara))


def contorno_por_sector(mascara, valida):
    """(filas, columnas, sector) de los puntos del contorno que no tocan el marco ni la viñeta."""
    lejos_del_marco = ndi.binary_erosion(valida, iterations=VECINDAD_BORDE, border_value=0)
    f, c = np.nonzero(mascara & ~ndi.binary_erosion(mascara) & lejos_del_marco)
    return f, c, _sector(f, c, *_ejes(mascara))


def _salto_por_sector(v, mascara, valida, centro, dc, df):
    """Brillo de la piel menos brillo de la lesión cerca del contorno, en cada sector (medianas)."""
    def mediana_por_sector(anillo):
        f, c = np.nonzero(anillo)
        sector = _sector(f, c, centro, dc, df)
        return np.array([np.median(v[f, c][sector == k]) if np.any(sector == k) else np.nan
                         for k in range(SECTORES)])

    adentro = ndi.distance_transform_edt(mascara)
    afuera = ndi.distance_transform_edt(~mascara)
    d0, d1 = ANILLO_SALTO
    lesion = mediana_por_sector((adentro > d0) & (adentro <= d1))
    piel = mediana_por_sector((afuera > d0) & (afuera <= d1) & valida)
    return np.maximum(piel - lesion, SALTO_MINIMO)          # NaN si al sector le falta piel o lesión


def contar_sectores_abruptos(gradientes, umbral=None):
    """B = cantidad de sectores por encima del umbral (gradientes: 8 valores, o una fila por imagen)."""
    umbral = UMBRAL_BORDE if umbral is None else umbral
    return np.sum(np.nan_to_num(np.asarray(gradientes, float), nan=-np.inf) > umbral, axis=-1)


def borde(img, mascara, medida=None, umbral=None):
    return int(contar_sectores_abruptos(gradiente_por_sector(img, mascara, medida), umbral))


# --------------------------------- Color ---------------------------------

def a_cono(hsv):
    """HSV -> punto de un cono: el matiz es un ángulo, el radio es S·V y la altura es V.

    Así el matiz se mide como ángulo (0 y 1 son el mismo rojo) y los colores muy oscuros
    o grises, donde H no significa nada, quedan juntos cerca del eje del cono.
    """
    hsv = np.asarray(hsv, float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    radio = s * v
    return np.stack([radio * np.cos(2 * np.pi * h), radio * np.sin(2 * np.pi * h), v], axis=-1)


def desde_cono(punto):
    """La inversa de a_cono, para guardar las referencias en HSV (más fácil de leer)."""
    x, y, v = punto
    radio = np.hypot(x, y)
    h = np.mod(np.arctan2(y, x) / (2 * np.pi), 1)
    return float(h), float(radio / v) if v > 0 else 0.0, float(v)


def referencia_de_color(pixeles_por_imagen):
    """Color típico a partir de las zonas de varias imágenes (lista de arrays de píxeles HSV).

    Primero la mediana de cada imagen y después la mediana de esas medianas, en el cono:
    cada imagen vota una vez, así una zona enorme no pesa más que las otras. Devuelve (H, S, V).
    """
    por_imagen = [np.median(a_cono(p), axis=0) for p in pixeles_por_imagen]
    return tuple(round(x, 3) for x in desde_cono(np.median(por_imagen, axis=0)))


def piel_de_la_imagen(img, mascara, valida):
    """RGB típico de la piel de esta imagen: mediana del anillo alrededor de la lesión."""
    afuera = ndi.distance_transform_edt(~mascara)
    anillo = (afuera > ANILLO_PIEL[0]) & (afuera <= ANILLO_PIEL[1]) & valida
    if anillo.sum() < 100:                 # lesión enorme: toda la piel que se vea
        anillo = ~mascara & valida
    if anillo.sum() < 100:                 # no se ve piel: no se corrige
        return np.ones(3)
    return np.median(img[anillo], axis=0)


def hsv_relativo(img, mascara, valida=None):
    """HSV de la imagen después de dividir cada canal por el de la piel de la misma foto.

    El mismo color se ve distinto según la luz y el tono de piel de cada foto; dividiendo
    por la piel se mide cuánto se aparta de ella, como hace el ojo. La piel queda en gris
    medio (ESCALA_RELATIVA), así entra hasta el doble de brillo que la piel.
    """
    valida = mejoramiento.mascara_vineta(img) if valida is None else valida
    piel = piel_de_la_imagen(img, np.asarray(mascara, bool), valida)
    return mejoramiento.a_hsv(np.clip(img / np.maximum(piel, 1e-3) * ESCALA_RELATIVA, 0, 1))


def mapa_de_colores(img, mascara, referencias=None):
    """Cada píxel de la lesión al color de referencia más cercano en HSV (H es circular).

    Trabaja sobre la imagen relativa a la piel (hsv_relativo). Además de los seis colores
    se compara con la piel misma (gris medio): el píxel que se parece más a la piel vale
    PARECIDO_A_PIEL y no cuenta para ningún color. Fuera de la lesión vale -1. Los colores
    sin referencia (None) no se asignan nunca.
    """
    referencias = REFERENCIAS_HSV if referencias is None else referencias
    mascara = np.asarray(mascara, bool)
    hsv = hsv_relativo(img, mascara)
    indices = [k for k, nombre in enumerate(COLORES) if referencias.get(nombre) is not None]
    puntos = [a_cono(referencias[COLORES[k]]) for k in indices]
    indices.append(PARECIDO_A_PIEL)
    puntos.append(a_cono((0, 0, ESCALA_RELATIVA)))

    mapa = np.full(mascara.shape, -1, int)
    if not mascara.any():
        return mapa
    distancias = np.linalg.norm(a_cono(hsv[mascara])[:, None, :] - np.array(puntos)[None], axis=-1)
    mapa[mascara] = np.array(indices)[np.argmin(distancias, axis=1)]
    return mapa


def fracciones_de_color(img, mascara, referencias=None):
    """Fracción de la lesión asignada a cada color (y a "piel"), como dict."""
    mapa = mapa_de_colores(img, mascara, referencias)
    total = max(int(np.sum(mapa != -1)), 1)
    fracciones = {nombre: float(np.sum(mapa == k) / total) for k, nombre in enumerate(COLORES)}
    fracciones["piel"] = float(np.sum(mapa == PARECIDO_A_PIEL) / total)
    return fracciones


def colores_presentes(fracciones, fraccion_min=None):
    """Colores que superan la fracción mínima. Si ninguno llega, el más abundante (C >= 1)."""
    fraccion_min = FRACCION_MIN_COLOR if fraccion_min is None else fraccion_min
    presentes = [c for c in COLORES if fracciones[c] >= fraccion_min]
    return presentes or [max(COLORES, key=lambda c: fracciones[c])]


def colores(img, mascara, referencias=None, fraccion_min=None):
    return colores_presentes(fracciones_de_color(img, mascara, referencias), fraccion_min)


# ----------------------------- Circularidad -----------------------------

def circularidad(mascara):
    """4πA/P² (skimage.measure.regionprops: area y perimeter)."""
    region = measure.regionprops(np.asarray(mascara, np.uint8))[0]
    return float(4 * np.pi * region.area / region.perimeter ** 2)


# ------------------- Zonas de color de referencia (cuaderno) -------------------

def zonas_de_color(nombre):
    """Máscaras de color de una imagen: las de PH2 (<nombre>_roi) y nuestras zonas rojas.

    Devuelve {color: máscara bool}. En PH2 el número de "Label" es el código de color
    (1 blanco ... 6 negro); los archivos sin número (p. ej. "LabelStreaks") no son colores.
    """
    zonas = {}
    for archivo in sorted((CARPETA_IMAGENES / nombre / f"{nombre}_roi").glob("*.bmp")):
        codigo = re.search(r"Label(\d)\.bmp$", archivo.name)
        if codigo:
            color = COLORES[int(codigo.group(1)) - 1]
            mascara = _leer_mascara(archivo)
            zonas[color] = zonas[color] | mascara if color in zonas else mascara
    rojo = CARPETA_ANOTACIONES_ROJO / f"{nombre}_rojo.png"
    if rojo.exists():
        mascara = _leer_mascara(rojo)
        if mascara.any():
            zonas["rojo"] = mascara
    return zonas


def _leer_mascara(ruta):
    m = io.imread(Path(ruta))
    return (m[..., 0] if m.ndim == 3 else m) > 0


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
