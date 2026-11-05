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
import pandas as pd                          # AGREGADO (M5)
from scipy import stats                      # AGREGADO (M5): correlación de Spearman
from skimage.feature import graycomatrix     # AGREGADO (M5)

ATRIBUTOS = ("energia", "entropia", "contraste", "idm", "correlacion",
             "cluster_shade", "cluster_prominence")

# ------------------------------ Parámetros ------------------------------
# AGREGADO (M5): elegidos con ENTRENAMIENTO y la máscara de M3 (ver cuaderno).
# DECIDIDO: NIVELES y DISTANCIA casi no cambian cuánto sigue cada atributo a D: la correlación
# de Spearman de "correlacion" queda entre 0,37 y 0,40 con 8, 16, 32 o 64 niveles y con
# distancia 1, 2 o 4. Se deja 32 porque conserva detalle sin dejar la matriz casi vacía
# (64 niveles son 4096 celdas), y distancia 1 porque es la más simple y no pierde nada.
NIVELES = 32
DISTANCIA = 1                                  # píxeles
ANGULOS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def glcm(gris, mascara, distancia=DISTANCIA, niveles=NIVELES):
    """Calcula la GLCM usando sólo pares de píxeles que caen dentro de la máscara.

    V se cuantiza con un rango fijo [0, 1], no con el rango de cada lesión, para que
    los valores se puedan comparar entre imágenes. Cada dirección se normaliza por
    separado y después se promedian las de ANGULOS. La matriz es simétrica: cuenta
    cada par en los dos sentidos.

    Entra: gris      array float en [0, 1] (alto x ancho)
           mascara   array bool (alto x ancho)
           distancia píxeles entre los dos elementos de cada par
           niveles   cantidad de niveles de gris
    Sale : array float (niveles x niveles) que suma 1. Todo ceros si la máscara no
           contiene ningún par de píxeles.
    """
    cuantizada = np.minimum((np.clip(gris, 0, 1) * niveles).astype(np.uint8), niveles - 1) + 1
    cuantizada[~mascara] = 0
    cuentas = graycomatrix(cuantizada, [distancia], ANGULOS, levels=niveles + 1, symmetric=True)
    cuentas = cuentas[1:, 1:, 0, :].astype(float)   # la fila y la columna 0 son los pares con el exterior
    pares = cuentas.sum(axis=(0, 1))
    if not pares.any():
        return np.zeros((niveles, niveles))
    return (cuentas[..., pares > 0] / pares[pares > 0]).mean(axis=2)


def atributos_glcm(gris, mascara, distancia=DISTANCIA, niveles=NIVELES):
    """Calcula los siete atributos de ATRIBUTOS a partir de glcm(). Entropía en bits.

    En una zona uniforme la correlación es 0/0: se devuelve 1, como hace skimage.

    Entra: gris, mascara, distancia, niveles   igual que en glcm()
    Sale : dict {atributo: float} con las claves de ATRIBUTOS. Todos NaN si la máscara
           no contiene ningún par de píxeles.
    """
    p = glcm(gris, mascara, distancia, niveles)
    if not p.any():
        return {k: np.nan for k in ATRIBUTOS}
    i, j = np.indices(p.shape)
    media_i, media_j = (i * p).sum(), (j * p).sum()
    desvio_i = np.sqrt((((i - media_i) ** 2) * p).sum())
    desvio_j = np.sqrt((((j - media_j) ** 2) * p).sum())
    cov = ((i - media_i) * (j - media_j) * p).sum()
    suma = i + j - media_i - media_j
    no_nulos = p[p > 0]
    return {
        "energia": float((p ** 2).sum()),
        "entropia": float(-(no_nulos * np.log2(no_nulos)).sum()),
        "contraste": float((((i - j) ** 2) * p).sum()),
        "idm": float((p / (1 + (i - j) ** 2)).sum()),
        "correlacion": float(cov / (desvio_i * desvio_j)) if desvio_i * desvio_j > 0 else 1.0,
        "cluster_shade": float(((suma ** 3) * p).sum()),
        "cluster_prominence": float(((suma ** 4) * p).sum()),
    }


def correlaciones_con_d(tabla):
    """AGREGADO (M5): mide cuánto cambia cada atributo con D (correlación de Spearman).

    Entra: tabla   DataFrame con columnas "glcm_<atributo>" y "D_derm"
    Sale : Series {columna: rho}, ordenada de mayor a menor rho; sin los atributos constantes
    """
    columnas = [c for c in tabla.columns if c.startswith("glcm_")]
    rho = {c: stats.spearmanr(tabla[c], tabla["D_derm"], nan_policy="omit").statistic for c in columnas}
    return pd.Series(rho, dtype=float).dropna().sort_values(ascending=False)


def elegir_atributo_d(tabla):
    """Elige el atributo que más crece con D_derm (columna "glcm_<atributo>" de la tabla).

    DECIDIDO: "más crece" = mayor correlación de Spearman, porque sólo importa el orden y
    no hace falta suponer una relación lineal. Se usa rho y no |rho| porque D estimado es la
    cantidad de umbrales superados: un atributo que baja con D daría el D al revés, y
    completar_d no le pasa el signo a estimar_d. No se pierde nada: en entrenamiento los dos
    que más cambian (correlacion, 0,39; cluster_prominence, 0,38) ya crecen con D.

    Entra: tabla   DataFrame con columnas "glcm_<atributo>" y "D_derm" (sólo ENTRENAMIENTO)
    Sale : str, nombre de la columna elegida
    """
    return correlaciones_con_d(tabla).index[0]


def ajustar_umbrales_d(valores, d_derm):
    """Calcula los umbrales que separan los valores de D sobre el atributo elegido.

    DECIDIDO: cuantiles, de modo que el D estimado se reparta como el del dermatólogo. Los
    puntos medios entre medianas daban peor resultado (validación cruzada en entrenamiento:
    acierto 21 % contra 30 %, error medio 1,42 contra 1,04) porque hay muy pocas imágenes
    con D alto (4 con D = 5) y sus medianas no crecen con D: se estimaba D = 5 en unas 27
    imágenes en vez de 4, y eso infla el TDS.

    Entra: valores  array float con el atributo de cada imagen de ENTRENAMIENTO
           d_derm   array int con el D del dermatólogo de cada imagen
    Sale : array float estrictamente creciente, con max(d_derm) umbrales. Si un valor de D
           no aparece, sus dos umbrales quedan pegados y ese D no se estima nunca.
    """
    valores, d_derm = np.asarray(valores, float), np.asarray(d_derm)
    conocidos = np.isfinite(valores)
    valores, d_derm = valores[conocidos], d_derm[conocidos]
    acumulado = np.cumsum(np.bincount(d_derm))[:-1] / len(d_derm)
    umbrales = np.quantile(valores, acumulado)
    for k in range(1, len(umbrales)):
        umbrales[k] = max(umbrales[k], np.nextafter(umbrales[k - 1], np.inf))
    return umbrales


def estimar_d(valor, umbrales):
    """Estima D como la cantidad de umbrales que el valor supera (np.searchsorted).

    Entra: valor     float, atributo de la imagen
           umbrales  array creciente, salida de ajustar_umbrales_d()
    Sale : int, D estimado entre 0 y len(umbrales)
    """
    return int(np.searchsorted(umbrales, valor, side="left"))
