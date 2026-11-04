"""Módulo 6 — Zonas de la lesión y mapas de características.

Sale de la DEVOLUCIÓN del anteproyecto:
  "hacer mapas de características y observar las interfases entre piel y
   lunares, además de zonas dentro del lunar: centro y periferia, para mejorar
   la caracterización y vincularla con el TDS intentando aumentar las opciones
   y términos de la fórmula".

Qué recibe : la imagen mejorada y la máscara de la lesión.
Qué devuelve:
    zonas(mascara) -> dict de máscaras bool:
        "centro"             dentro de la lesión, lejos del borde
        "periferia"          dentro de la lesión, cerca del borde
                             (centro y periferia no se pisan y juntas son toda la lesión)
        "interfase_interna"  anillo de ANCHO_INTERFASE px por DENTRO del borde
        "interfase_externa"  anillo de ANCHO_INTERFASE px por FUERA del borde (piel)
    atributos_por_zona(img, mascara) -> dict {zona: {atributo: valor}}
        (color medio en HSV y atributos GLCM del módulo 5 en cada zona)
    contrastes_entre_zonas(por_zona) -> dict plano con las diferencias que van a
        la tabla y se prueban como términos extra del TDS:
        "periferia_vs_centro_X" (periferia - centro) e "interfase_X" (piel - lesión en el
        borde), para X en H, S, V, contraste y entropía.
    mapa_caracteristica(gris, mascara, atributo) -> array float (alto x ancho):
        el atributo GLCM calculado en ventanas de VENTANA px (NaN fuera de la lesión).
        Es lo que se muestra en el informe y en la interfaz para ver DÓNDE está la
        información (contexto frente a las redes neuronales, que no lo muestran).
"""

import numpy as np
from scipy import ndimage as ndi                                 # AGREGADO (M6)

from modulos.m1_mejoramiento import mejoramiento                 # AGREGADO (M6)
from modulos.m5_textura_criterio_d import textura_criterio_d     # AGREGADO (M6)

ZONAS = ("centro", "periferia", "interfase_interna", "interfase_externa")

# ------------------------------ Parámetros ------------------------------
# AGREGADO (M6): elegidos con ENTRENAMIENTO y la máscara de referencia (ver cuaderno).
# DECIDIDO: con 0,5 el centro es el 29 % de la lesión (mediana). Se probó 0,3 y 0,7: el AUC del
# mejor contraste es 0,80, 0,78 y 0,77, diferencias chicas para 140 imágenes, así que no hay
# motivo para apartarse del valor que marca la mitad de la distancia al borde.
FRACCION_CENTRO = 0.5     # el centro es donde la distancia al borde supera esta fracción de la máxima
# DECIDIDO: en píxeles fijos y no relativo al tamaño: el ancho de la transición lesión-piel no
# depende del área (Spearman 0,07). La piel se alcanza a 10 px del borde en la mediana y a 16 en el
# percentil 75, y 15 px no se come la lesión (la distancia máxima al borde es de al menos 60 px).
ANCHO_INTERFASE = 15      # píxeles
# DECIDIDO: una ventana de 15 px tiene menos píxeles que celdas la GLCM de 32 x 32, y una de 51 px
# ya aplana el mapa (variación relativa 0,08 contra 0,10). Con PASO 8 tarda unos 0,2 s por imagen.
VENTANA = 31              # píxeles, lado de la ventana del mapa de características
PASO = 8                  # píxeles entre ventanas (para que no tarde una eternidad)
FRACCION_MIN_VENTANA = 0.25   # AGREGADO (M6): fracción mínima de la ventana dentro de la lesión

# AGREGADO (M6): qué se compara entre zonas.
COLOR_HSV = ("H", "S", "V")
TEXTURAS_CONTRASTE = ("contraste", "entropia")


def zonas(mascara):
    """Divide la lesión y su entorno en cuatro zonas con la transformada de distancia.

    El borde de la imagen no cuenta como borde de la lesión: si la lesión toca el marco,
    esos píxeles no son periferia por estar pegados al marco.

    Entra: mascara   array bool (alto x ancho), la lesión
    Sale : dict {zona: array bool (alto x ancho)} con las claves de ZONAS
    """
    adentro = ndi.distance_transform_edt(mascara)
    afuera = ndi.distance_transform_edt(~mascara)
    centro = adentro > FRACCION_CENTRO * adentro.max()
    return {
        "centro": centro,
        "periferia": mascara & ~centro,
        "interfase_interna": mascara & (adentro <= ANCHO_INTERFASE),
        "interfase_externa": ~mascara & (afuera <= ANCHO_INTERFASE),
    }


def _color_medio(hsv, zona):
    """AGREGADO (M6): calcula el color medio de una zona en HSV.

    H es un ángulo (los rojos están en 0 y también en 1), así que su media es circular.

    Entra: hsv    array float (alto x ancho x 3), con H, S y V en [0, 1]
           zona   array bool (alto x ancho), con al menos un píxel
    Sale : dict con "H" (en [0, 1)), "S" y "V"
    """
    angulo = 2 * np.pi * hsv[..., 0][zona]
    h = np.arctan2(np.sin(angulo).mean(), np.cos(angulo).mean()) / (2 * np.pi) % 1
    return {"H": float(h), "S": float(hsv[..., 1][zona].mean()), "V": float(hsv[..., 2][zona].mean())}


def atributos_por_zona(img, mascara):
    """Mide el color medio HSV y los atributos GLCM (M5) de V en cada zona.

    La interfase externa no usa las esquinas de la viñeta (mejoramiento.mascara_vineta):
    son negro de la lente, no piel, y falsearían la comparación lesión-piel.

    Entra: img       imagen mejorada RGB float [0, 1] (alto x ancho x 3)
           mascara   array bool (alto x ancho), la lesión
    Sale : dict {zona: {atributo: float}} con las claves de ZONAS. Cada zona tiene H, S, V y
           los de textura_criterio_d.ATRIBUTOS; todo NaN si la zona queda vacía.
    """
    hsv = mejoramiento.a_hsv(img)
    por_zona = {}
    z = zonas(mascara)
    z["interfase_externa"] = z["interfase_externa"] & mejoramiento.mascara_vineta(img)
    for nombre in ZONAS:
        color = _color_medio(hsv, z[nombre]) if z[nombre].any() else {k: np.nan for k in COLOR_HSV}
        por_zona[nombre] = {**color, **textura_criterio_d.atributos_glcm(hsv[..., 2], z[nombre])}
    return por_zona


def _diferencia(a, b, atributo):
    """AGREGADO (M6): resta un atributo entre dos zonas (H se resta como ángulo).

    Entra: a, b       dict {atributo: valor} de dos zonas
           atributo   str, clave de los dict
    Sale : float a - b; para H, entre -0,5 y 0,5
    """
    resta = a[atributo] - b[atributo]
    return float((resta + 0.5) % 1 - 0.5) if atributo == "H" else float(resta)


def contrastes_entre_zonas(por_zona):
    """Calcula las diferencias entre zonas que van a la tabla y al TDS extendido.

    "periferia_vs_centro_X" = periferia - centro: cuánto cambia la lesión de adentro hacia afuera.
    "interfase_X" = externa - interna (piel - lesión): cuánto se distingue la lesión del entorno.

    DECIDIDO: las restas van en este sentido porque tds_extendido SUMA cada término, así que
    conviene que un valor alto signifique "más sospechosa". En ENTRENAMIENTO, los que más
    separan son periferia_vs_centro_S (AUC 0,78) e interfase_S (AUC 0,68). V de la periferia
    contra el centro no aporta (AUC 0,51).

    Entra: por_zona   dict {zona: {atributo: valor}}, salida de atributos_por_zona()
    Sale : dict plano {nombre: float}. NaN si alguna de las dos zonas estaba vacía.
    """
    salida = {}
    for atributo in (*COLOR_HSV, *TEXTURAS_CONTRASTE):
        salida[f"periferia_vs_centro_{atributo}"] = _diferencia(por_zona["periferia"], por_zona["centro"], atributo)
        salida[f"interfase_{atributo}"] = _diferencia(por_zona["interfase_externa"], por_zona["interfase_interna"], atributo)
    return salida


def mapa_caracteristica(gris, mascara, atributo="contraste"):
    """Calcula un atributo GLCM en ventanas para ver en qué parte de la lesión está la textura.

    Divide la imagen en bloques de PASO x PASO. El valor de cada bloque sale de una ventana
    de VENTANA x VENTANA centrada en él, usando sólo píxeles de la lesión. Las ventanas con
    menos de FRACCION_MIN_VENTANA de lesión quedan en NaN porque tienen muy pocos pares.

    Entra: gris      array float en [0, 1] (alto x ancho), en general el canal V
           mascara   array bool (alto x ancho), la lesión
           atributo  str, uno de textura_criterio_d.ATRIBUTOS
    Sale : array float (alto x ancho), con NaN fuera de la lesión
    """
    if atributo not in textura_criterio_d.ATRIBUTOS:
        raise ValueError(f"atributo tiene que ser uno de {textura_criterio_d.ATRIBUTOS}")
    mapa = np.full(mascara.shape, np.nan)
    minimo = FRACCION_MIN_VENTANA * VENTANA ** 2
    filas, columnas = np.nonzero(mascara)
    for f0 in range(filas.min() // PASO * PASO, filas.max() + 1, PASO):
        for c0 in range(columnas.min() // PASO * PASO, columnas.max() + 1, PASO):
            bloque = mascara[f0:f0 + PASO, c0:c0 + PASO]
            if not bloque.any():
                continue
            arriba, izquierda = max(f0 + PASO // 2 - VENTANA // 2, 0), max(c0 + PASO // 2 - VENTANA // 2, 0)
            abajo, derecha = f0 + PASO // 2 + VENTANA // 2 + 1, c0 + PASO // 2 + VENTANA // 2 + 1
            ventana = mascara[arriba:abajo, izquierda:derecha]
            if ventana.sum() < minimo:
                continue
            valor = textura_criterio_d.atributos_glcm(gris[arriba:abajo, izquierda:derecha], ventana)[atributo]
            mapa[f0:f0 + PASO, c0:c0 + PASO][bloque] = valor
    return mapa
