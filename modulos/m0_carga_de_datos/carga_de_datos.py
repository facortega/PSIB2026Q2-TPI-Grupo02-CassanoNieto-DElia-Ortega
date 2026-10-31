"""Módulo 0 — Carga de datos (base PH2).

Consigna: Contextualización (características de las imágenes y del dataset).
Devolución del anteproyecto: acotar el alcance y la cantidad de imágenes.

Qué tiene que devolver este módulo: un `Caso` por imagen, con
    imagen       RGB float en [0, 1], forma (alto, ancho, 3)
    mascara_ref  máscara de la lesión dibujada a mano (bool, alto x ancho)
    diagnostico  0 = nevo común, 1 = nevo atípico, 2 = melanoma
    asimetria    evaluación del dermatólogo: 0, 1 o 2 ejes
    colores      lista de colores presentes (nombres de COLORES)
    estructuras  dict {estructura: True/False} con las de ESTRUCTURAS
El resto de los módulos sólo usa eso.

La base NO se sube al repositorio (ver datos/LEEME.md): cada integrante la
descarga y la deja en datos/PH2Dataset/.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_DATOS = Path(__file__).resolve().parents[2] / "datos"   # carpeta datos/ del repo
CARPETA_PH2 = CARPETA_DATOS / "PH2Dataset"
CARPETA_IMAGENES = CARPETA_PH2 / "PH2 Dataset images"          # VERIFICAR con la descarga
TABLA_PH2 = CARPETA_PH2 / "PH2_dataset.txt"                    # VERIFICAR (también viene .xlsx)
PARTICION = CARPETA_DATOS / "particion.csv"                    # se genera con armar_particion()

# Vocabulario común a todo el repo. El orden de COLORES es el de los códigos 1 a 6 de PH2
# (VERIFICAR en la documentación de la base).
COLORES = ("blanco", "rojo", "marron_claro", "marron_oscuro", "azul_grisaceo", "negro")
ESTRUCTURAS = ("red_pigmento", "puntos_globulos", "estrias", "regresion", "velo")
DIAGNOSTICOS = {0: "nevo común", 1: "nevo atípico", 2: "melanoma"}

# ------------------------------ Parámetros ------------------------------
# PENDIENTE (M0): decidir y justificar en el informe (devolución: "acotar la
# cantidad de imágenes"). Por ejemplo: todas las de 0 y 2 y una parte de las de 1,
# o sólo las que tienen anotados todos los criterios.
FRACCION_PRUEBA = 0.30   # fracción de imágenes que se reservan para la evaluación final
SEMILLA = 0              # para que la partición sea siempre la misma


class FaltanDatos(Exception):
    """La base PH2 todavía no está en datos/PH2Dataset/."""


@dataclass
class Caso:
    """Una imagen de PH2 con sus anotaciones. Se usa así: caso.imagen, caso.diagnostico..."""
    nombre: str                          # p. ej. "IMD002"
    imagen: np.ndarray                   # RGB float [0, 1]
    mascara_ref: np.ndarray              # bool
    diagnostico: int                     # 0, 1 o 2 (ver DIAGNOSTICOS)
    asimetria: int = -1                  # 0, 1 o 2 según el dermatólogo (-1 = sin dato)
    colores: list = field(default_factory=list)
    estructuras: dict = field(default_factory=dict)

    @property
    def sospechosa(self):
        """Etiqueta binaria del trabajo: True si NO es un nevo común.

        PENDIENTE (M0): confirmar con el grupo que esta es la definición
        (anteproyecto: "lesiones sospechosas" vs. "lunares comunes").
        """
        return self.diagnostico > 0

    @property
    def c_dermatologo(self):
        """Criterio C del dermatólogo: cantidad de colores (1 a 6)."""
        return len(self.colores)

    @property
    def d_dermatologo(self):
        """Criterio D adaptado: cantidad de estructuras presentes (0 a 5)."""
        return int(sum(bool(v) for v in self.estructuras.values()))


def hay_datos():
    return TABLA_PH2.exists() and CARPETA_IMAGENES.exists()


def leer_tabla_ph2():
    """Tabla de anotaciones de PH2, una fila por imagen.

    Columnas que tiene que tener (los módulos y verificacion.py las usan):
        nombre, diagnostico, asimetria, colores, red_pigmento, puntos_globulos,
        estrias, regresion, velo
    con `colores` como texto con los nombres separados por espacios
    (p. ej. "marron_claro marron_oscuro") y las estructuras en True/False.

    PENDIENTE (M0):
      - [ ] Abrir PH2_dataset.txt (o el .xlsx) y ver cómo están codificadas las
            columnas: separadores, filas de encabezado, códigos de diagnóstico,
            de asimetría, de colores y de cada estructura (A/P, T/AT...).
      - [ ] Decidir qué cuenta como "presente" en cada estructura y anotarlo
            en el cuaderno (lo usa el criterio D adaptado).
      - [ ] Devolver la tabla con las columnas de arriba.
    """
    if not hay_datos():
        raise FaltanDatos("todavía no está la base en datos/PH2Dataset/ (ver datos/LEEME.md)")
    raise NotImplementedError("M0: falta programar leer_tabla_ph2()")


def seleccionar_subconjunto(tabla):
    """Filtra la tabla según el alcance acotado del trabajo.

    PENDIENTE (M0): aplicar el criterio decidido en Parámetros. Mientras no
    se decida, devolver la tabla completa está bien.
    """
    raise NotImplementedError("M0: falta programar seleccionar_subconjunto()")


def cargar_caso(nombre, tabla=None):
    """Lee la imagen, la máscara y las anotaciones de una imagen de PH2.

    PENDIENTE (M0):
      - [ ] Ubicar la imagen dermatoscópica y la máscara de la lesión dentro de
            CARPETA_IMAGENES/<nombre>/ (ver nombres de subcarpetas en la descarga).
      - [ ] Imagen en float [0, 1] (skimage.util.img_as_float) y máscara en bool.
      - [ ] Completar las anotaciones con la fila de `tabla` (si es None, leerla).
    """
    raise NotImplementedError("M0: falta programar cargar_caso()")


def armar_particion(tabla):
    """Divide las imágenes en entrenamiento y prueba y guarda datos/particion.csv.

    La partición tiene que ser estratificada por diagnóstico (misma proporción
    de cada clase en ambos conjuntos) y reproducible (SEMILLA). Se genera UNA
    vez y se sube al repo, así todos usan la misma.
    Columnas de particion.csv: nombre, diagnostico, conjunto ("entrenamiento"/"prueba").

    PENDIENTE (M0): programarla (sklearn.model_selection.train_test_split con stratify).
    """
    raise NotImplementedError("M0: falta programar armar_particion()")


# ----------------- Ya está hecho: no hace falta tocarlo -----------------

def leer_particion():
    if not PARTICION.exists():
        raise FaltanDatos("falta datos/particion.csv: correr armar_particion() en el cuaderno de M0")
    return pd.read_csv(PARTICION, dtype={"nombre": str})


def nombres(conjunto="todos"):
    """Nombres de las imágenes de un conjunto: "entrenamiento", "prueba" o "todos"."""
    part = leer_particion()
    if conjunto != "todos":
        part = part[part["conjunto"] == conjunto]
    return list(part["nombre"])


def cargar_todos(conjunto="todos"):
    """Carga todos los casos de un conjunto (puede tardar: son imágenes grandes)."""
    tabla = leer_tabla_ph2()
    return [cargar_caso(n, tabla) for n in nombres(conjunto)]
