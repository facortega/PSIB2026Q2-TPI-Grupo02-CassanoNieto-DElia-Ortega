"""Módulo 3 — Segmentación de la lesión.

Recibe la imagen mejorada y la máscara `valida` de M1. segmentar(img, valida, metodo)
devuelve una máscara bool con una sola región y sin huecos.

Pasos: preparar (canal B suavizado) -> método -> posprocesar.
Cualquier parámetro se puede cambiar en la llamada sin tocar este archivo, por ejemplo
segmentar(img, valida, "crecimiento", sigma=1, tolerancia=0.4).
"""

import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from skimage import color, filters, segmentation, transform
from sklearn.cluster import KMeans

METODOS = ("otsu", "crecimiento", "watershed", "kmeans", "chan_vese")
# Casi el mismo Dice que watershed, pero más simple, más rápido y con menos fallas (cuaderno, sección 3)
METODO_ELEGIDO = "crecimiento"

# ------------------------------ Parámetros ------------------------------
# Ajustados con la grilla del cuaderno sobre las imágenes de entrenamiento.

CANAL = 2                   # canal de RGB sobre el que se segmenta (2 = azul)
SIGMA = 4.0                 # px. Desvío del suavizado gaussiano antes de segmentar
RADIO_APERTURA = 3          # px. Borra islas y puentes más finos que el disco
RADIO_CIERRE = 10           # px. Rellena mellas del borde; uno más grande taparía entrantes
                            # del contorno que después mide M4

# Semilla y niveles de referencia (los usan crecimiento y watershed)
VENTANA_SEMILLA = 0.25      # lado del recuadro central donde se busca la semilla (fracción de la imagen)
PERCENTIL_LESION = 5        # brillo típico de la lesión: percentil 5 del recuadro central
PERCENTIL_PIEL = 90         # brillo típico de la piel: percentil 90 de toda la zona válida

# Crecimiento de regiones
TOLERANCIA = 0.5            # dónde cae el límite, entre el nivel de la lesión (0) y el de la piel (1)

# Watershed: marcadores como fracción del camino entre el nivel de la lesión (0) y el de la piel (1)
MARCA_LESION = 0.4          # más oscuro que esto -> seguro lesión
MARCA_PIEL = 0.6            # más claro que esto y conectado con el borde -> seguro piel

# k-means
GRUPOS = 2                  # el grupo más claro es piel y el resto, lesión
MUESTRA_KMEANS = 20000      # píxeles al azar para calcular los centros (con todos tarda mucho más)
SEMILLA_KMEANS = 0          # semilla fija para que dé siempre lo mismo

# Chan-Vese
MU = 0.1                    # cuánto se penaliza el largo del contorno (más alto = más suave)
ITERACIONES_CV = 200
ESCALA_CV = 0.25            # se resuelve en la imagen reducida: a tamaño completo es muy lento

# Comparación de métodos (cuaderno)
EMPATE = 0.001              # diferencias de Dice menores a esto cuentan como empate


# ----------------------------- AGREGADO (M3) -----------------------------

def preparar(img, sigma=None):
    """Toma el canal B y lo suaviza con un gaussiano de desvío `sigma` px."""
    sigma = SIGMA if sigma is None else sigma
    canal = img[..., CANAL].astype(float)
    return ndi.gaussian_filter(canal, sigma) if sigma > 0 else canal


def _disco(radio):
    """Elemento estructurante circular de radio `radio` px (para cv2)."""
    lado = 2 * int(radio) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))


def semilla_y_niveles(canal, valida):
    """Devuelve (semilla, nivel_lesion, nivel_piel).

    La semilla es el píxel más oscuro del recuadro central. Los niveles son percentiles
    y no el mínimo y el máximo, así un reflejo o un glóbulo suelto no los mueve.
    """
    alto, ancho = canal.shape
    f0, f1 = int(alto * (0.5 - VENTANA_SEMILLA / 2)), int(alto * (0.5 + VENTANA_SEMILLA / 2))
    c0, c1 = int(ancho * (0.5 - VENTANA_SEMILLA / 2)), int(ancho * (0.5 + VENTANA_SEMILLA / 2))
    recuadro = np.where(valida[f0:f1, c0:c1], canal[f0:f1, c0:c1], np.inf)
    f, c = np.unravel_index(np.argmin(recuadro), recuadro.shape)
    nivel_lesion = np.percentile(recuadro[np.isfinite(recuadro)], PERCENTIL_LESION)
    nivel_piel = np.percentile(canal[valida], PERCENTIL_PIEL)
    return (f0 + f, c0 + c), nivel_lesion, nivel_piel


# --- Los métodos. Cada uno devuelve una máscara "cruda" (sin posprocesar). ---

def _otsu(canal, valida):
    """Umbral de Otsu calculado sólo con los píxeles válidos; lo más oscuro es lesión."""
    umbral = filters.threshold_otsu(canal[valida])
    return (canal < umbral) & valida


def _crecimiento(canal, valida, tolerancia=None):
    """Crecimiento de regiones desde la semilla con un límite de brillo fijo.

    Agregar vecinos de a uno hasta que no entre ninguno da lo mismo que quedarse con la
    componente conexa de "píxeles bajo el límite" que contiene a la semilla, así que se
    hace de una vez con ndi.label. El límite no se recalcula a medida que la región crece:
    si lo hiciera, al sumar la periferia clara se iría aflojando y se escaparía a la piel.
    """
    tolerancia = TOLERANCIA if tolerancia is None else tolerancia
    semilla, nivel_lesion, nivel_piel = semilla_y_niveles(canal, valida)
    limite = nivel_lesion + tolerancia * max(nivel_piel - nivel_lesion, 1e-3)
    etiquetas, _ = ndi.label((canal <= limite) & valida, structure=np.ones((3, 3)))
    if etiquetas[semilla] == 0:                 # no pasa: la semilla es el píxel más oscuro
        return np.zeros(canal.shape, bool)
    return etiquetas == etiquetas[semilla]


def _watershed(canal, valida, marca_lesion=None, marca_piel=None):
    """Watershed con marcadores sobre el gradiente de Sobel.

    Los marcadores eligen qué es seguro para cada lado y la inundación decide el resto:
    la frontera termina sobre el borde más marcado entre lesión y piel.
    """
    marca_lesion = MARCA_LESION if marca_lesion is None else marca_lesion
    marca_piel = MARCA_PIEL if marca_piel is None else marca_piel
    semilla, nivel_lesion, nivel_piel = semilla_y_niveles(canal, valida)
    contraste = max(nivel_piel - nivel_lesion, 1e-3)

    oscuro = (canal <= nivel_lesion + marca_lesion * contraste) & valida
    claro, _ = ndi.label((canal >= nivel_lesion + marca_piel * contraste) & valida)
    borde_valido = valida & ~ndi.binary_erosion(valida)        # anillo exterior de la zona válida
    claro_externo = np.isin(claro, np.unique(claro[borde_valido & (claro > 0)]))
    marcadores = np.zeros(canal.shape, np.int32)
    marcadores[claro_externo | ~valida] = 1                                          # piel
    marcadores[oscuro] = 2                                                           # lesión
    marcadores[semilla] = 2                  # la semilla siempre es oscura, pero por las dudas
    etiquetas = segmentation.watershed(filters.sobel(canal), marcadores)
    return (etiquetas == 2) & valida


def _kmeans(img, valida, sigma=None, grupos=None):
    """k-means sobre el color en Lab (los tres canales suavizados, no sólo B).

    Los centros se calculan con una muestra de píxeles válidos y después cada píxel va
    al centro más cercano. El grupo de mayor L (el más claro) es la piel.
    """
    sigma = SIGMA if sigma is None else sigma
    grupos = GRUPOS if grupos is None else int(grupos)
    suave = np.stack([ndi.gaussian_filter(img[..., k], sigma) if sigma > 0 else img[..., k]
                      for k in range(3)], axis=-1)
    lab = color.rgb2lab(np.clip(suave, 0, 1))
    puntos = lab[valida]
    rng = np.random.default_rng(SEMILLA_KMEANS)
    muestra = puntos[rng.choice(len(puntos), min(MUESTRA_KMEANS, len(puntos)), replace=False)]
    centros = KMeans(n_clusters=grupos, n_init=4, random_state=SEMILLA_KMEANS).fit(muestra).cluster_centers_

    # Cada píxel al centro más cercano (distancia euclídea en Lab)
    distancias = ((lab[..., None, :] - centros[None, None]) ** 2).sum(axis=-1)
    grupo = np.argmin(distancias, axis=-1)
    piel = np.argmax(centros[:, 0])                       # el de mayor L es el más claro
    return (grupo != piel) & valida


def _chan_vese(canal, valida, mu=None):
    """Chan-Vese sobre el canal reducido, arrancando desde el contorno de Otsu.

    Fuera de `valida` se rellena con el nivel de la piel para que las esquinas negras no
    cuenten como lesión. Al final la máscara se vuelve a llevar al tamaño original.
    """
    mu = MU if mu is None else mu
    _, _, nivel_piel = semilla_y_niveles(canal, valida)
    relleno = np.where(valida, canal, nivel_piel)
    chico = transform.rescale(relleno, ESCALA_CV, anti_aliasing=True)
    inicial = np.where(chico < filters.threshold_otsu(chico), 1.0, -1.0)
    fase = segmentation.chan_vese(chico, mu=mu, max_num_iter=ITERACIONES_CV,
                                  init_level_set=inicial)
    # Chan-Vese devuelve las dos zonas sin decir cuál es cuál: la lesión es la más oscura
    if chico[fase].mean() > chico[~fase].mean():
        fase = ~fase
    grande = transform.resize(fase.astype(float), canal.shape, order=1) > 0.5
    return grande & valida


_FUNCIONES = {"otsu": _otsu, "crecimiento": _crecimiento,
              "watershed": _watershed, "chan_vese": _chan_vese}


def mascara_cruda(img, valida=None, metodo=None, sigma=None, **parametros):
    """Salida del método antes de posprocesar (el cuaderno la reusa para probar varios radios)."""
    metodo = METODO_ELEGIDO if metodo is None else metodo
    if metodo not in METODOS:
        raise ValueError(f"metodo tiene que ser uno de {METODOS}")
    if valida is None:
        valida = np.ones(img.shape[:2], bool)
    if metodo == "kmeans":           # el único que trabaja con los tres canales de color
        return _kmeans(img, valida, sigma, **parametros)
    return _FUNCIONES[metodo](preparar(img, sigma), valida, **parametros)


# ---------------------------------------------------------------------------

def segmentar(img, valida=None, metodo=None, radio_apertura=None, radio_cierre=None, **parametros):
    """Máscara final de la lesión: el método y después el posprocesado.

    `parametros` (opcionales) reemplazan a los de arriba: sigma, tolerancia,
    marca_lesion, marca_piel, grupos, mu.
    """
    cruda = mascara_cruda(img, valida, metodo, **parametros)
    return posprocesar(cruda, radio_apertura, radio_cierre)


def posprocesar(mascara_cruda, radio_apertura=None, radio_cierre=None):
    """Apertura, cierre, región del centro (o la más grande) y relleno de huecos.

    Si la apertura borra todo, se sigue sin ella para no devolver una máscara vacía.
    """
    radio_apertura = RADIO_APERTURA if radio_apertura is None else radio_apertura
    radio_cierre = RADIO_CIERRE if radio_cierre is None else radio_cierre
    m = mascara_cruda.astype(np.uint8)
    if radio_apertura > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disco(radio_apertura))
        if not m.any():
            m = mascara_cruda.astype(np.uint8)
    if radio_cierre > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disco(radio_cierre))

    etiquetas, cantidad = ndi.label(m)
    if cantidad == 0:
        return np.zeros(mascara_cruda.shape, bool)
    alto, ancho = m.shape
    elegida = etiquetas[alto // 2, ancho // 2]
    if elegida == 0:
        areas = np.bincount(etiquetas.ravel())
        areas[0] = 0
        elegida = int(np.argmax(areas))
    return ndi.binary_fill_holes(etiquetas == elegida)


def dice(a, b):
    """Coeficiente de Dice entre dos máscaras bool. Si las dos están vacías, da 1."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / total)


# ----------------------- Para comparar métodos (cuaderno) -----------------------

def describir(parametros):
    """Texto corto de una combinación de parámetros, p. ej. "tolerancia=0.5" ("-" si no hay)."""
    return ", ".join(f"{k}={v}" for k, v in parametros.items()) or "-"


def evaluar_imagen(img, valida, referencia, grilla, sigmas, radios):
    """Dice de cada combinación de la grilla en una imagen, como lista de filas (dicts).

    `grilla` es {metodo: [parámetros, ...]}; se prueban todos los `sigmas` y todos los
    pares de `radios`. La salida cruda se calcula una sola vez y se reusa para cada par.
    """
    filas = []
    for metodo, opciones in grilla.items():
        for sigma in sigmas:
            for parametros in opciones:
                inicio = time.perf_counter()
                cruda = mascara_cruda(img, valida, metodo, sigma=sigma, **parametros)
                segundos = time.perf_counter() - inicio
                for radio_apertura in radios:
                    for radio_cierre in radios:
                        mascara = posprocesar(cruda, radio_apertura, radio_cierre)
                        filas.append({"metodo": metodo, "sigma": sigma, "parametros": describir(parametros),
                                      "radio_apertura": radio_apertura, "radio_cierre": radio_cierre,
                                      "dice": dice(mascara, referencia), "segundos": segundos})
    return filas


def prueba_de_los_signos(dice_a, dice_b):
    """Compara dos métodos imagen por imagen. Devuelve (gana_a, gana_b, empates, p).

    Cuenta en cuántas imágenes gana cada uno (sin contar empates). Si fueran igual de
    buenos, cada imagen sería como tirar una moneda; `p` es la probabilidad de un reparto
    tan desparejo como el observado o más. Con p < 0,05 la diferencia no parece casualidad.
    """
    diferencia = np.asarray(dice_a, float) - np.asarray(dice_b, float)
    gana_a = int((diferencia > EMPATE).sum())
    gana_b = int((diferencia < -EMPATE).sum())
    empates = len(diferencia) - gana_a - gana_b
    p = stats.binomtest(gana_a, gana_a + gana_b, 0.5).pvalue if gana_a + gana_b > 0 else 1.0
    return gana_a, gana_b, empates, float(p)
