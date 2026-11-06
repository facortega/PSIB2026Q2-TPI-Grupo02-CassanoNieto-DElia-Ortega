"""Módulo 5 — Textura (GLCM) y criterio D adaptado.

Consigna: Extracción de características (análisis de texturas).
Anteproyecto: metodología 6.

Qué recibe : la imagen mejorada y la máscara de la lesión (la textura usa el canal V; regresión y velo, el color).
Qué devuelve:
    atributos_glcm(gris, mascara) -> dict {atributo: valor} con los de ATRIBUTOS,
        calculados SÓLO con pares de píxeles dentro de la máscara y promediados
        en las cuatro direcciones (0°, 45°, 90°, 135°).

    Criterio D, por estructuras:
    medidas_estructuras(img, mascara)      -> dict {estructura: valor}, una medida pensada
                                              para cada estructura de PH2
    umbrales_estructuras(tabla)            -> dict {estructura: umbral}, por Youden
    detectar_estructuras(medidas, umbrales)-> dict {estructura: bool}
    D estimado = cantidad de estructuras detectadas (0 a 5), igual que cuenta el dermatólogo.
Lo de D se ajusta SOLO con las filas de entrenamiento (lo hace
herramientas/procesamiento_completo.completar_d).

Ojo con la máscara: graycomatrix no acepta máscaras. Una forma: cuantizar V en
NIVELES niveles (1..NIVELES), poner 0 fuera de la lesión, calcular la GLCM con
NIVELES+1 niveles y descartar la fila y la columna 0 antes de normalizar.
"""

import numpy as np
from scipy import ndimage as ndi             # AGREGADO (M5): distancia al borde
from skimage import measure, morphology      # AGREGADO (M5): detectores de estructuras
from skimage.feature import hessian_matrix, hessian_matrix_eigvals   # AGREGADO (M5): líneas de las estrías
from skimage.feature import graycomatrix     # AGREGADO (M5)

from modulos.m0_carga_de_datos.carga_de_datos import ESTRUCTURAS   # AGREGADO (M5)
from modulos.m1_mejoramiento import mejoramiento                    # AGREGADO (M5)

ATRIBUTOS = ("energia", "entropia", "contraste", "idm", "correlacion",
             "cluster_shade", "cluster_prominence")

# ------------------------------ Parámetros ------------------------------
# AGREGADO (M5): 32 niveles conservan detalle sin que la matriz de 32 x 32 quede casi vacía (64 serían
# 4096 celdas), y distancia 1 compara píxeles vecinos, la escala de las líneas finas de la red de pigmento.
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


# ------------------- D por estructuras (AGREGADO, M5) -------------------
# Cada estructura de PH2 tiene una medida pensada para ella. Las variantes se eligieron mirando
# sólo ENTRENAMIENTO (ver cuaderno). Los umbrales también salen de entrenamiento.
RADIO_GLOBULO = 6          # px. El cierre rellena manchas oscuras más angostas que este disco
RADIO_LINEA = 2            # px. La apertura borra lo más fino que esto (líneas de la red)
UMBRAL_GLOBULO = 0.05      # diferencia de brillo mínima para contar una mancha
AREA_GLOBULO = (8, 300)    # px. Tamaño de una mancha que cuenta como punto o glóbulo
ANCHO_BORDE = 15           # px a cada lado del borde: anillo donde se buscan las estrías
SIGMA_LINEA = 1.5          # px. Escala del filtro de líneas (las estrías son finas)
RADIAL_MIN = 0.8           # |coseno| mínimo entre la línea y la dirección radial (unos 37°)
PERCENTIL_LINEA = 90       # una línea es marcada si supera este percentil de la propia lesión
PIEL_DESDE, PIEL_HASTA = 30, 60   # px afuera del borde donde se toma la piel de referencia


def _medida_red(v, mascara):
    """Red atípica: cuánto varía el contraste local dentro de la lesión.

    Una red regular da un contraste parecido en toda la lesión; una atípica, con líneas de
    distinto grosor y zonas sin red, lo hace variar. Usa el mapa de contraste de M6.

    Entra: v        array float [0, 1] (alto x ancho), brillo de la imagen mejorada
           mascara  array bool, la lesión
    Sale : float, desvío sobre media del mapa de contraste (NaN si el mapa queda vacío)
    """
    from modulos.m6_zonas_y_mapas import zonas_y_mapas   # acá y no arriba: M6 importa M5
    mapa = zonas_y_mapas.mapa_caracteristica(v, mascara, "contraste")
    valores = mapa[mascara & np.isfinite(mapa)]
    return float(np.std(valores) / np.mean(valores)) if valores.size and np.mean(valores) > 0 else np.nan


def _medida_globulos(v, mascara):
    """Puntos y glóbulos: cuántas manchas oscuras y compactas hay por área de lesión.

    El cierre con un disco rellena lo oscuro más angosto que el disco, y la resta con la imagen
    (black-hat, como en los pelos de M1) deja esas manchas. Después una apertura borra lo más fino,
    que son las líneas de la red, y quedan las manchas redondeadas.

    Entra: v        array float [0, 1], brillo de la imagen mejorada
           mascara  array bool, la lesión
    Sale : float, manchas cada 10 000 px de lesión
    """
    negro = morphology.closing(v, morphology.disk(RADIO_GLOBULO)) - v
    manchas = morphology.opening(negro, morphology.disk(RADIO_LINEA)) > UMBRAL_GLOBULO
    regiones = measure.regionprops(measure.label(manchas & mascara))
    cantidad = sum(AREA_GLOBULO[0] <= r.area <= AREA_GLOBULO[1] for r in regiones)
    return float(cantidad / mascara.sum() * 1e4)


def _medida_estrias(v, mascara, valida):
    """Estrías: qué parte del borde tiene líneas oscuras que apuntan hacia afuera.

    Las estrías son líneas oscuras y finas que salen radialmente de la periferia (Sadeghi et al.,
    2013, las detectan como crestas y valles orientados). Con la matriz Hessiana se marcan las
    líneas oscuras (valles: curvatura fuerte a lo ancho y débil a lo largo) y se calcula su
    dirección; después se compara con la dirección radial, que sale de la distancia al borde.

    Entra: v        array float [0, 1], brillo de la imagen mejorada
           mascara  array bool, la lesión
           valida   array bool, fuera de la viñeta
    Sale : float entre 0 y 1, fracción del anillo con líneas marcadas y radiales
    """
    hrr, hrc, hcc = hessian_matrix(v, sigma=SIGMA_LINEA, order="rc", use_gaussian_derivatives=True)
    mayor, menor = hessian_matrix_eigvals((hrr, hrc, hcc))
    linea = np.where((mayor > 0) & (np.abs(menor) < mayor), mayor - np.abs(menor), 0)
    # La línea corre perpendicular a la dirección de máxima curvatura
    angulo = 0.5 * np.arctan2(2 * hrc, hrr - hcc) + np.pi / 2
    distancia = np.where(mascara, ndi.distance_transform_edt(mascara), -ndi.distance_transform_edt(~mascara))
    gf, gc = np.gradient(ndi.gaussian_filter(distancia, 3))
    radial = np.abs(np.cos(angulo) * gf + np.sin(angulo) * gc) / (np.hypot(gf, gc) + 1e-9)
    anillo = (np.abs(distancia) <= ANCHO_BORDE) & valida
    marcada = linea > np.percentile(linea[mascara], PERCENTIL_LINEA)
    return float((marcada & (radial > RADIAL_MIN))[anillo].mean()) if anillo.any() else np.nan


def _medida_regresion(hsv, mascara, valida):
    """Regresión: qué parte de la lesión es más clara y menos saturada que la piel.

    Las zonas de regresión son blancas, como una cicatriz. En el canal azul se ven casi como
    piel, así que M3 suele dejarlas afuera de la máscara; por eso se mira toda la envolvente
    convexa de la lesión y no sólo la máscara.

    Entra: hsv      array float (alto x ancho x 3), imagen mejorada en HSV
           mascara  array bool, la lesión
           valida   array bool, fuera de la viñeta
    Sale : float entre 0 y 1
    """
    s, v = hsv[..., 1], hsv[..., 2]
    afuera = ndi.distance_transform_edt(~mascara)
    piel = (afuera > PIEL_DESDE) & (afuera <= PIEL_HASTA) & valida
    if not piel.any():
        return np.nan
    envolvente = morphology.convex_hull_image(mascara) & valida
    blanco = (v > np.median(v[piel])) & (s < np.median(s[piel]))
    return float(blanco[envolvente].mean())


def _medida_velo(img, mascara):
    """Velo azul-blanquecino: cuánto domina el azul sobre el rojo en la parte más azulada.

    La piel y los marrones tienen más rojo que azul; el velo es la única estructura azulada.

    Entra: img      imagen mejorada RGB float [0, 1]
           mascara  array bool, la lesión
    Sale : float, percentil 90 de B - R dentro de la lesión
    """
    return float(np.percentile((img[..., 2] - img[..., 0])[mascara], 90))


def medidas_estructuras(img, mascara):
    """Calcula una medida por cada estructura de PH2.

    Entra: img      imagen mejorada RGB float [0, 1] (alto x ancho x 3)
           mascara  array bool (alto x ancho), la lesión
    Sale : dict {estructura: float} con las claves de ESTRUCTURAS. Un valor más alto
           indica más probabilidad de que la estructura esté.
    """
    hsv = mejoramiento.a_hsv(img)
    v = hsv[..., 2]
    valida = mejoramiento.mascara_vineta(img)
    return {"red_pigmento": _medida_red(v, mascara),
            "puntos_globulos": _medida_globulos(v, mascara),
            "estrias": _medida_estrias(v, mascara, valida),
            "regresion": _medida_regresion(hsv, mascara, valida),
            "velo": _medida_velo(img, mascara)}


def umbrales_estructuras(tabla):
    """Elige el umbral de cada medida con el índice de Youden.

    Entra: tabla  DataFrame con columnas "estr_<estructura>" (medida) y "derm_<estructura>"
                  (True si el dermatólogo la marcó). Sólo filas de ENTRENAMIENTO.
    Sale : dict {estructura: umbral}
    """
    from modulos.m7_clasificacion_resultados.clasificacion_resultados import umbral_youden
    umbrales = {}
    for e in ESTRUCTURAS:
        conocidas = tabla[f"estr_{e}"].notna()
        umbrales[e] = umbral_youden(tabla.loc[conocidas, f"estr_{e}"], tabla.loc[conocidas, f"derm_{e}"].astype(bool))
    return umbrales


def detectar_estructuras(medidas, umbrales):
    """Decide qué estructuras están presentes.

    Entra: medidas   dict {estructura: valor}, salida de medidas_estructuras()
           umbrales  dict {estructura: umbral}, salida de umbrales_estructuras()
    Sale : dict {estructura: bool}. Una medida NaN cuenta como ausente.
    Para D: sum(detectar_estructuras(...).values()).
    """
    return {e: bool(np.isfinite(medidas[e]) and medidas[e] > umbrales[e]) for e in ESTRUCTURAS}
