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
        la tabla y se prueban como términos extra del TDS, p. ej.
        "centro_vs_periferia_V", "interfase_V" (lesión vs. piel en el borde)...
    mapa_caracteristica(gris, mascara, atributo) -> array float (alto x ancho):
        el atributo GLCM calculado en ventanas de VENTANA px (NaN fuera de la lesión).
        Es lo que se muestra en el informe y en la interfaz para ver DÓNDE está la
        información (contexto frente a las redes neuronales, que no lo muestran).
"""

import numpy as np

ZONAS = ("centro", "periferia", "interfase_interna", "interfase_externa")

# ------------------------------ Parámetros ------------------------------
FRACCION_CENTRO = 0.5     # el centro es donde la distancia al borde supera esta fracción de la máxima
ANCHO_INTERFASE = None    # píxeles. PENDIENTE (M6): justificar (relativo al tamaño de la lesión?)
VENTANA = None            # píxeles, lado de la ventana del mapa de características
PASO = None               # píxeles entre ventanas (para que no tarde una eternidad)


def zonas(mascara):
    """Usar la transformada de distancia (scipy.ndimage.distance_transform_edt)."""
    raise NotImplementedError("M6: falta programar zonas()")


def atributos_por_zona(img, mascara):
    raise NotImplementedError("M6: falta programar atributos_por_zona()")


def contrastes_entre_zonas(por_zona):
    """PENDIENTE (M6): decidir qué diferencias tienen sentido clínico (ver marco teórico)."""
    raise NotImplementedError("M6: falta programar contrastes_entre_zonas()")


def mapa_caracteristica(gris, mascara, atributo="contraste"):
    raise NotImplementedError("M6: falta programar mapa_caracteristica()")
