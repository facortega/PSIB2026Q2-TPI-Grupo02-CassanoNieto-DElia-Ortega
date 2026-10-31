"""Módulo 7 — TDS, clasificación, evaluación y resultados finales.

Consigna: Resultados/diagnóstico (análisis estadísticos, comparaciones, significancia).
Anteproyecto: metodología 7 y 9; hipótesis 2.
Devolución: "aumentar las opciones y términos de la fórmula" -> TDS extendido.

Qué recibe : la tabla de características (una fila por imagen, ver
             herramientas/procesamiento_completo.tabla_de_caracteristicas), con
             la columna "conjunto" (entrenamiento / prueba).
Qué hace   :
    tds(A, B, C, D)                      TDS = 1,3 A + 0,1 B + 0,5 C + 0,5 D
    tds_extendido(tabla, terminos, ref)  TDS + suma de los términos extra estandarizados:
                                         (x - media) / desvío, con media y desvío de
                                         `ref` (las filas de ENTRENAMIENTO). Peso 1 cada uno.
    umbral_youden(puntajes, etiquetas)   umbral que maximiza sensibilidad + especificidad
                                         CONVENCIÓN: sospechosa si puntaje > umbral
    metricas(predicho, real)             dict con sensibilidad, especificidad, vpp, exactitud
    auc(puntajes, etiquetas)             área bajo la curva ROC
    kappa_cohen(a, b)                    acuerdo descontando el azar (A y C vs. dermatólogo)
    generar(tabla)                       figuras y tablas finales (lo llama main.py)
"""

import numpy as np

from modulos.herramientas import graficos

PESOS_TDS = {"A": 1.3, "B": 0.1, "C": 0.5, "D": 0.5}
UMBRAL_SOSPECHOSA = 4.75     # umbrales clínicos de la regla [Nachbar et al., 1994]
UMBRAL_MELANOMA = 5.45

# PENDIENTE (M7): qué términos se agregan al TDS (los elige el grupo con
# ENTRENAMIENTO: textura de M5, contrastes entre zonas de M6, circularidad...).
TERMINOS_EXTRA = []


def tds(A, B, C, D):
    """Funciona con números o con arrays/columnas."""
    raise NotImplementedError("M7: falta programar tds()")


def tds_extendido(tabla, terminos, ref):
    raise NotImplementedError("M7: falta programar tds_extendido()")


def umbral_youden(puntajes, etiquetas):
    raise NotImplementedError("M7: falta programar umbral_youden()")


def metricas(predicho, real):
    raise NotImplementedError("M7: falta programar metricas()")


def auc(puntajes, etiquetas):
    raise NotImplementedError("M7: falta programar auc()")


def kappa_cohen(a, b):
    raise NotImplementedError("M7: falta programar kappa_cohen()")


def generar(tabla):
    """Lo llama main.py al final, con la tabla de todas las imágenes.

    PENDIENTE (M7):
      - [ ] Elegir umbrales con ENTRENAMIENTO y evaluar en PRUEBA.
      - [ ] Tabla: TDS clínico (4,75), TDS con umbral de Youden, TDS extendido
            -> sensibilidad, especificidad y AUC en prueba.
      - [ ] κ de A y de C contra el dermatólogo; Dice medio de la segmentación.
      - [ ] Figuras: curvas ROC, distribución del TDS por grupo
            (graficos.guardar_figura) y tablas en informe/tablas/.
    """
    raise NotImplementedError("M7: falta programar generar()")
