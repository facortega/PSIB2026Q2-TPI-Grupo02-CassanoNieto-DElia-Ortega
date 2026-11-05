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
    intervalo_auc(...)                   AUC con intervalo de confianza del 95 % (bootstrap)
    diferencia_auc(...)                  diferencia entre dos AUC con su intervalo (bootstrap)
    generar(tabla)                       figuras y tablas finales (lo llama main.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from modulos.herramientas import graficos

PESOS_TDS = {"A": 1.3, "B": 0.1, "C": 0.5, "D": 0.5}
UMBRAL_SOSPECHOSA = 4.75     # umbrales clínicos de la regla [Nachbar et al., 1994]
UMBRAL_MELANOMA = 5.45

# Se eligieron en M6 sólo con ENTRENAMIENTO, antes de mirar prueba: son los dos contrastes
# que mejor separan sin correlacionarse entre sí (ver el cuaderno de M6).
TERMINOS_EXTRA = ["zona_periferia_vs_centro_S", "zona_interfase_S"]
REPETICIONES_BOOTSTRAP = 2000
SEMILLA_BOOTSTRAP = 0


def tds(A, B, C, D):
    """Calcula el puntaje dermatoscópico total de la regla ABCD.

    Entra: A, B, C, D   números o arrays/columnas del mismo largo
    Sale : número o array con 1,3 A + 0,1 B + 0,5 C + 0,5 D
    """
    return PESOS_TDS["A"] * A + PESOS_TDS["B"] * B + PESOS_TDS["C"] * C + PESOS_TDS["D"] * D


def tds_extendido(tabla, terminos, ref):
    """Suma al TDS cada término, llevado a la misma escala con la media y el desvío de `ref`.

    Entra: tabla      DataFrame con A, B, C, D y las columnas de `terminos`
           terminos   lista de nombres de columna (peso 1 cada una)
           ref        DataFrame con las filas de ENTRENAMIENTO, de donde salen media y desvío
    Sale : array con un puntaje por fila de `tabla`
    """
    puntaje = np.asarray(tds(tabla["A"], tabla["B"], tabla["C"], tabla["D"]), float)
    for termino in terminos:
        puntaje = puntaje + (tabla[termino].to_numpy(float) - ref[termino].mean()) / ref[termino].std()
    return puntaje


def umbral_youden(puntajes, etiquetas):
    """Busca el umbral que maximiza sensibilidad + especificidad (índice de Youden).

    Prueba los puntos medios entre valores consecutivos. Una lesión es sospechosa si puntaje > umbral.

    Entra: puntajes   array de números
           etiquetas  array bool (True = sospechosa)
    Sale : float con el umbral
    """
    puntajes, etiquetas = np.asarray(puntajes, float), np.asarray(etiquetas, bool)
    valores = np.unique(puntajes)
    candidatos = np.r_[valores[0] - 1, (valores[:-1] + valores[1:]) / 2]
    youden = [(puntajes[etiquetas] > u).mean() + (puntajes[~etiquetas] <= u).mean() for u in candidatos]
    return float(candidatos[int(np.argmax(youden))])


def metricas(predicho, real):
    """Compara una clasificación con la verdad.

    Entra: predicho, real   arrays bool del mismo largo (True = sospechosa)
    Sale : dict con sensibilidad, especificidad, vpp (valor predictivo positivo) y exactitud
    """
    predicho, real = np.asarray(predicho, bool), np.asarray(real, bool)
    return {"sensibilidad": predicho[real].mean(), "especificidad": (~predicho[~real]).mean(),
            "vpp": real[predicho].mean() if predicho.any() else np.nan, "exactitud": (predicho == real).mean()}


def auc(puntajes, etiquetas):
    """Área bajo la curva ROC: probabilidad de que una sospechosa puntúe más que una común.

    Se calcula con la suma de rangos de Mann-Whitney; los empates cuentan la mitad.

    Entra: puntajes   array de números
           etiquetas  array bool (True = sospechosa)
    Sale : float entre 0 y 1 (0,5 = azar)
    """
    puntajes, etiquetas = np.asarray(puntajes, float), np.asarray(etiquetas, bool)
    n_pos, n_neg = etiquetas.sum(), (~etiquetas).sum()
    return float((rankdata(puntajes)[etiquetas].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def kappa_cohen(a, b):
    """Acuerdo entre dos evaluaciones descontando el que se daría por azar.

    Entra: a, b   arrays o listas del mismo largo con categorías
    Sale : float (1 = acuerdo perfecto, 0 = el esperable por azar)
    """
    a, b = np.asarray(a), np.asarray(b)
    categorias = np.union1d(a, b)
    observado = (a == b).mean()
    esperado = sum((a == c).mean() * (b == c).mean() for c in categorias)
    return float((observado - esperado) / (1 - esperado))


def _muestras_bootstrap(etiquetas, repeticiones, semilla):
    """Sortea con reposición dentro de cada grupo, para conservar sospechosas y comunes.

    Entra: etiquetas    array bool
           repeticiones cantidad de muestras
           semilla      entero, para que dé siempre lo mismo
    Sale : generador de arrays de índices
    """
    rng = np.random.default_rng(semilla)
    positivos, negativos = np.flatnonzero(etiquetas), np.flatnonzero(~etiquetas)
    for _ in range(repeticiones):
        yield np.r_[rng.choice(positivos, len(positivos)), rng.choice(negativos, len(negativos))]


def intervalo_auc(puntajes, etiquetas, repeticiones=REPETICIONES_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP):
    """AUC con su intervalo de confianza del 95 % por bootstrap de percentiles.

    Entra: puntajes, etiquetas   como en auc()
           repeticiones, semilla opcionales
    Sale : tupla (auc, límite inferior, límite superior)
    """
    puntajes, etiquetas = np.asarray(puntajes, float), np.asarray(etiquetas, bool)
    valores = [auc(puntajes[i], etiquetas[i]) for i in _muestras_bootstrap(etiquetas, repeticiones, semilla)]
    return (auc(puntajes, etiquetas), *np.percentile(valores, [2.5, 97.5]))


def diferencia_auc(puntajes_a, puntajes_b, etiquetas, repeticiones=REPETICIONES_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP):
    """Diferencia AUC(a) - AUC(b) sobre las mismas imágenes, con su intervalo del 95 %.

    Si el intervalo no incluye al 0, la diferencia es significativa al 5 %.

    Entra: puntajes_a, puntajes_b  arrays del mismo largo
           etiquetas               array bool
           repeticiones, semilla   opcionales
    Sale : tupla (diferencia, límite inferior, límite superior)
    """
    a, b, etiquetas = np.asarray(puntajes_a, float), np.asarray(puntajes_b, float), np.asarray(etiquetas, bool)
    valores = [auc(a[i], etiquetas[i]) - auc(b[i], etiquetas[i]) for i in _muestras_bootstrap(etiquetas, repeticiones, semilla)]
    return (auc(a, etiquetas) - auc(b, etiquetas), *np.percentile(valores, [2.5, 97.5]))


def _fila_clasificador(nombre, puntaje_prueba, umbral, sospechosa):
    """Arma la fila de la tabla final de un clasificador evaluado en prueba.

    Entra: nombre          texto
           puntaje_prueba  array con el puntaje de las imágenes de prueba
           umbral          float elegido antes de mirar prueba
           sospechosa      array bool con la verdad de prueba
    Sale : dict con nombre, umbral, sensibilidad, especificidad y AUC con su intervalo
    """
    met = metricas(puntaje_prueba > umbral, sospechosa)
    valor, inferior, superior = intervalo_auc(puntaje_prueba, sospechosa)
    return {"clasificador": nombre, "umbral": umbral, "sensibilidad": met["sensibilidad"], "especificidad": met["especificidad"],
            "AUC": valor, "AUC inf.": inferior, "AUC sup.": superior}


def generar(tabla):
    """Genera la tabla, las figuras y el acuerdo finales. Lo llama main.py cuando M0 a M6 están listos.

    Los umbrales se eligen con entrenamiento y todo se mide en prueba.

    Entra: tabla  DataFrame de pc.completar_d, con A, B, C, D, A_derm, C_derm, dice, sospechosa,
                  conjunto y las columnas de TERMINOS_EXTRA
    Sale : nada; guarda m7_clasificacion y m7_acuerdo en informe/tablas/,
           y m7_roc y m7_tds_grupos en informe/figuras/
    """
    entrenamiento, prueba = tabla[tabla["conjunto"] == "entrenamiento"], tabla[tabla["conjunto"] == "prueba"]
    real_e, real_p = entrenamiento["sospechosa"].to_numpy(bool), prueba["sospechosa"].to_numpy(bool)
    tds_e = np.asarray(tds(entrenamiento["A"], entrenamiento["B"], entrenamiento["C"], entrenamiento["D"]), float)
    tds_p = np.asarray(tds(prueba["A"], prueba["B"], prueba["C"], prueba["D"]), float)
    ext_e = tds_extendido(entrenamiento, TERMINOS_EXTRA, entrenamiento)
    ext_p = tds_extendido(prueba, TERMINOS_EXTRA, entrenamiento)

    filas = [_fila_clasificador("TDS, umbral clínico", tds_p, UMBRAL_SOSPECHOSA, real_p),
             _fila_clasificador("TDS, umbral de Youden", tds_p, umbral_youden(tds_e, real_e), real_p),
             _fila_clasificador("TDS extendido, umbral de Youden", ext_p, umbral_youden(ext_e, real_e), real_p)]
    graficos.guardar_tabla(pd.DataFrame(filas), "m7_clasificacion", decimales=2)
    diferencia, inferior, superior = diferencia_auc(ext_p, tds_p, real_p)
    print(f"  AUC extendido - TDS en prueba: {diferencia:.3f} (IC 95 %: {inferior:.3f} a {superior:.3f})")

    acuerdo = pd.DataFrame([{"medida": "kappa de A", "valor": kappa_cohen(prueba["A"], prueba["A_derm"])},
                            {"medida": "kappa de C", "valor": kappa_cohen(prueba["C"], prueba["C_derm"])},
                            {"medida": "Dice medio", "valor": prueba["dice"].mean()}])
    graficos.guardar_tabla(acuerdo, "m7_acuerdo", decimales=2)

    fig, ax = plt.subplots()
    for nombre, puntaje in (("TDS", tds_p), ("TDS extendido", ext_p)):
        umbrales = np.r_[np.inf, np.sort(np.unique(puntaje))[::-1], -np.inf]
        ax.plot([(puntaje[~real_p] > u).mean() for u in umbrales], [(puntaje[real_p] > u).mean() for u in umbrales],
                label=f"{nombre} (AUC {auc(puntaje, real_p):.2f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad"); ax.legend()
    graficos.guardar_figura(fig, "m7_roc")

    fig, ax = plt.subplots()
    cajas = ax.boxplot([tds_p[~real_p], tds_p[real_p]], tick_labels=["común", "sospechosa"], patch_artist=True, flierprops={"markersize": 2})
    cajas["boxes"][0].set_facecolor(graficos.COLOR_GRUPO["comun"]); cajas["boxes"][1].set_facecolor(graficos.COLOR_GRUPO["sospechosa"])
    ax.axhline(UMBRAL_SOSPECHOSA, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("TDS en prueba")
    graficos.guardar_figura(fig, "m7_tds_grupos")
