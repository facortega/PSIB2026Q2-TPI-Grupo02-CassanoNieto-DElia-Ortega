"""Junta los módulos 1 a 6 en orden sobre una imagen y arma la tabla de características.

Ya está terminado: no hace falta tocarlo para programar los módulos.
Con `hasta=` se corre sólo hasta la etapa que ya esté programada, por ejemplo
procesar(caso, hasta="segmentacion").

    caso = carga_de_datos.cargar_caso("IMD002")
    res = procesar(caso)                         # todo, con los métodos elegidos
    res = procesar(caso, metodo_pelos="mediana") # hipótesis 1
    res = procesar(caso, calidad_jpeg=25)        # objetivo de máxima (compresión)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modulos.m0_carga_de_datos import carga_de_datos
from modulos.m0_carga_de_datos.carga_de_datos import Caso
from modulos.m1_mejoramiento import mejoramiento
from modulos.m2_compresion import compresion
from modulos.m3_segmentacion import segmentacion
from modulos.m4_criterios_abc import criterios_abc
from modulos.m5_textura_criterio_d import textura_criterio_d
from modulos.m6_zonas_y_mapas import zonas_y_mapas

ETAPAS = ("mejoramiento", "segmentacion", "criterios_abc", "textura", "zonas")


@dataclass
class Resultado:
    """Todo lo que se obtiene de una imagen. Se usa así: res.mascara, res.criterios["A"]..."""
    caso: Caso
    img_entrada: np.ndarray = None       # la original, o la descomprimida si calidad_jpeg
    img_mejorada: np.ndarray = None
    valida: np.ndarray = None
    mascara_pelos: np.ndarray = None
    mascara: np.ndarray = None
    dice: float = np.nan
    criterios: dict = field(default_factory=dict)
    textura: dict = field(default_factory=dict)
    zonas: dict = field(default_factory=dict)       # contrastes entre zonas (dict plano)


def procesar(caso, hasta="zonas", metodo_pelos="cierre", metodo_seg=None, calidad_jpeg=None):
    if hasta not in ETAPAS:
        raise ValueError(f"'hasta' tiene que ser una de estas: {ETAPAS}")
    ultima = ETAPAS.index(hasta)
    res = Resultado(caso=caso)

    img = caso.imagen
    if calidad_jpeg is not None:
        img, _ = compresion.comprimir_jpeg(img, calidad_jpeg)
    res.img_entrada = img

    res.img_mejorada, res.valida, res.mascara_pelos = mejoramiento.mejorar(img, metodo_pelos)
    if ultima < 1:
        return res
    metodo_seg = metodo_seg or segmentacion.METODO_ELEGIDO
    res.mascara = segmentacion.segmentar(res.img_mejorada, res.valida, metodo_seg)
    res.dice = segmentacion.dice(res.mascara, caso.mascara_ref)
    if ultima < 2:
        return res
    res.criterios = criterios_abc.criterios(res.img_mejorada, res.mascara)
    if ultima < 3:
        return res
    v = mejoramiento.a_hsv(res.img_mejorada)[..., 2]
    res.textura = textura_criterio_d.atributos_glcm(v, res.mascara)
    if ultima < 4:
        return res
    por_zona = zonas_y_mapas.atributos_por_zona(res.img_mejorada, res.mascara)
    res.zonas = zonas_y_mapas.contrastes_entre_zonas(por_zona)
    return res


def procesar_todos(conjunto="todos", hasta="zonas", **kwargs):
    """Procesa todas las imágenes de un conjunto ("entrenamiento", "prueba" o "todos")."""
    tabla = carga_de_datos.leer_tabla_ph2()
    salida = []
    for nombre in carga_de_datos.nombres(conjunto):
        caso = carga_de_datos.cargar_caso(nombre, tabla)
        salida.append(procesar(caso, hasta=hasta, **kwargs))
    return salida


def tabla_de_caracteristicas(resultados):
    """Una fila por imagen con todo lo medido y lo anotado por el dermatólogo.

    Las imágenes (que ocupan mucho) no se guardan: sólo los números.
    """
    particion = carga_de_datos.leer_particion().set_index("nombre")["conjunto"]
    filas = []
    for res in resultados:
        c = res.caso
        fila = {
            "nombre": c.nombre,
            "conjunto": particion.get(c.nombre, ""),
            "diagnostico": c.diagnostico,
            "sospechosa": c.sospechosa,
            "dice": res.dice,
            "A_derm": c.asimetria,
            "C_derm": c.c_dermatologo,
            "D_derm": c.d_dermatologo,
            "colores_derm": " ".join(c.colores),
        }
        for k, v in res.criterios.items():
            fila[k] = " ".join(v) if k == "colores" else v
        fila.update({f"glcm_{k}": v for k, v in res.textura.items()})
        fila.update({f"zona_{k}": v for k, v in res.zonas.items()})
        filas.append(fila)
    return pd.DataFrame(filas)


def completar_d(tabla):
    """Agrega la columna "D" (criterio D estimado) ajustando SOLO con entrenamiento."""
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    atributo = textura_criterio_d.elegir_atributo_d(entrenamiento)
    umbrales = textura_criterio_d.ajustar_umbrales_d(entrenamiento[atributo].to_numpy(),
                                                     entrenamiento["D_derm"].to_numpy())
    tabla = tabla.copy()
    tabla["D"] = [textura_criterio_d.estimar_d(v, umbrales) for v in tabla[atributo]]
    tabla.attrs["atributo_d"], tabla.attrs["umbrales_d"] = atributo, [float(u) for u in umbrales]
    return tabla
