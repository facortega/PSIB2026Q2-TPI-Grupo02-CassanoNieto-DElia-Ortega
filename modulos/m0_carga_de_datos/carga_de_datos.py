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
from skimage import io                                    # AGREGADO (M0): leer los .bmp
from skimage.util import img_as_float                     # AGREGADO (M0): uint8 -> float [0, 1]
from sklearn.model_selection import train_test_split      # AGREGADO (M0): partición estratificada

CARPETA_DATOS = Path(__file__).resolve().parents[2] / "datos"   # carpeta datos/ del repo
CARPETA_PH2 = CARPETA_DATOS / "PH2Dataset"
CARPETA_IMAGENES = CARPETA_PH2 / "PH2 Dataset images"          # VERIFICADO con la descarga
TABLA_PH2 = CARPETA_PH2 / "PH2_dataset.txt"                    # VERIFICADO (también viene .xlsx)
PARTICION = CARPETA_DATOS / "particion.csv"                    # se genera con armar_particion()

# Vocabulario común a todo el repo. El orden de COLORES es el de los códigos 1 a 6 de PH2
# (VERIFICADO con las leyendas al final de PH2_dataset.txt: 1 White, 2 Red, 3 Light-Brown,
# 4 Dark-Brown, 5 Blue-Gray, 6 Black).
COLORES = ("blanco", "rojo", "marron_claro", "marron_oscuro", "azul_grisaceo", "negro")
ESTRUCTURAS = ("red_pigmento", "puntos_globulos", "estrias", "regresion", "velo")
DIAGNOSTICOS = {0: "nevo común", 1: "nevo atípico", 2: "melanoma"}

# ------------------------------ Parámetros ------------------------------
# DECIDIDO (M0): se usan las 200 imágenes de PH2 (ver DIAGNOSTICOS_INCLUIDOS).
# El recorte que pidió la devolución se hace en los objetivos (la comparación con
# JPEG queda como objetivo de máxima), no en la cantidad de imágenes.
FRACCION_PRUEBA = 0.30   # fracción de imágenes que se reservan para la evaluación final
SEMILLA = 0              # para que la partición sea siempre la misma

# AGREGADO (M0): diagnósticos que entran al trabajo.
# DECIDIDO: (0, 1, 2) = las 200 imágenes. Si se cambia, hay que regenerar particion.csv.
DIAGNOSTICOS_INCLUIDOS = (0, 1, 2)

# AGREGADO (M0): códigos de PH2_dataset.txt (leyendas al final del archivo).
#   Estructuras: A = Absent, T = Typical, AT = Atypical, P = Present.
#   DECIDIDO para el criterio D adaptado:
#     - red de pigmento: cuenta SÓLO si es atípica ("AT"). La regla ABCD original cuenta
#       su presencia, pero en PH2 las 200 imágenes tienen red (T o AT), así que eso no
#       distingue nada; la red atípica es la que se asocia al melanoma.
#       Resultado: 0 nevos comunes, 77/80 atípicos y 39/40 melanomas con red atípica.
#     - el resto de las estructuras: todo lo que NO es "A" cuenta como presente.
CODIGO_AUSENTE = "A"
CODIGO_ATIPICA = "AT"
COLUMNAS_TXT = ("asimetria", *ESTRUCTURAS)   # orden de las columnas separadas por "|" en el .txt


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

        DECIDIDO (M0): sospechosa = nevo atípico o melanoma (120 imágenes);
        no sospechosa = nevo común (80). Es la pregunta del anteproyecto:
        distinguir lesiones sospechosas de lunares comunes.
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

    Hecho (M0):
      - [x] Abrir PH2_dataset.txt (o el .xlsx) y ver cómo están codificadas las
            columnas: separadores, filas de encabezado, códigos de diagnóstico,
            de asimetría, de colores y de cada estructura (A/P, T/AT...).
      - [x] Decidir qué cuenta como "presente" en cada estructura y anotarlo
            en el cuaderno (lo usa el criterio D adaptado).
      - [x] Devolver la tabla con las columnas de arriba.
    """
    if not hay_datos():
        raise FaltanDatos("todavía no está la base en datos/PH2Dataset/ (ver datos/LEEME.md)")

    # AGREGADO (M0) ------------------------------------------------------------
    # Formato de cada fila (separadores "||" entre bloques y "|" dentro del bloque de criterios):
    # || IMD003 || <histológico> || 0 || 0 | T | A | A | A | A || 3  4 ||
    #     nombre    (casi vacío)   clín.  asim. + 5 estructuras     colores
    # El archivo está en latin-1 (por el "PH²"), tiene un encabezado y al final las leyendas:
    # sólo se toman las filas que empiezan con "|| IMD".
    filas = []
    with open(TABLA_PH2, encoding="latin-1") as f:
        for linea in f:
            if not linea.strip().startswith("|| IMD"):
                continue
            bloques = [b.strip() for b in linea.strip().strip("|").split("||")]
            nombre, _histologico, clinico, criterios, codigos_color = bloques
            valores = [v.strip() for v in criterios.split("|")]
            fila = {"nombre": nombre, "diagnostico": int(clinico)}
            for columna, valor in zip(COLUMNAS_TXT, valores):
                if columna == "asimetria":
                    fila[columna] = int(valor)
                elif columna == "red_pigmento":                  # DECIDIDO: sólo la red atípica
                    fila[columna] = valor == CODIGO_ATIPICA
                else:
                    fila[columna] = valor != CODIGO_AUSENTE
            # Colores: códigos 1-6 separados por espacios -> nombres de COLORES
            fila["colores"] = " ".join(COLORES[int(c) - 1] for c in codigos_color.split())
            filas.append(fila)

    tabla = pd.DataFrame(filas)
    if len(tabla) != 200:   # PH2 tiene 200 imágenes: si no, el formato cambió
        raise ValueError(f"se leyeron {len(tabla)} filas de PH2_dataset.txt (tendrían que ser 200)")
    return tabla
    # --------------------------------------------------------------------------


def seleccionar_subconjunto(tabla):
    """Filtra la tabla según el alcance acotado del trabajo.

    DECIDIDO (M0): las 200 imágenes (DIAGNOSTICOS_INCLUIDOS = (0, 1, 2)).
    """
    # AGREGADO (M0): filtra por DIAGNOSTICOS_INCLUIDOS (ver Parámetros)
    return tabla[tabla["diagnostico"].isin(DIAGNOSTICOS_INCLUIDOS)].reset_index(drop=True)


def cargar_caso(nombre, tabla=None):
    """Lee la imagen, la máscara y las anotaciones de una imagen de PH2.

    Hecho (M0):
      - [x] Ubicar la imagen dermatoscópica y la máscara de la lesión dentro de
            CARPETA_IMAGENES/<nombre>/ (ver nombres de subcarpetas en la descarga).
      - [x] Imagen en float [0, 1] (skimage.util.img_as_float) y máscara en bool.
      - [x] Completar las anotaciones con la fila de `tabla` (si es None, leerla).
    """
    # AGREGADO (M0) ------------------------------------------------------------
    # Estructura de la descarga (verificada):
    #   <nombre>/<nombre>_Dermoscopic_Image/<nombre>.bmp   imagen RGB uint8
    #   <nombre>/<nombre>_lesion/<nombre>_lesion.bmp       máscara (ya viene en bool)
    # Ojo: no todas las imágenes miden 768 x 560 (hay 24 tamaños distintos; p. ej. IMD002
    # es 765 x 572). Ningún módulo debe asumir un tamaño fijo.
    if tabla is None:
        tabla = leer_tabla_ph2()
    carpeta = CARPETA_IMAGENES / nombre
    imagen = img_as_float(io.imread(carpeta / f"{nombre}_Dermoscopic_Image" / f"{nombre}.bmp"))
    imagen = imagen[..., :3]                                   # por si algún .bmp trae canal alfa
    mascara = io.imread(carpeta / f"{nombre}_lesion" / f"{nombre}_lesion.bmp")
    if mascara.ndim == 3:                                      # por si alguna máscara viene en RGB
        mascara = mascara[..., 0]
    mascara = mascara > 0

    fila = tabla.loc[tabla["nombre"] == nombre]
    if fila.empty:
        raise KeyError(f"{nombre} no está en la tabla de PH2")
    fila = fila.iloc[0]
    return Caso(
        nombre=nombre,
        imagen=imagen,
        mascara_ref=mascara,
        diagnostico=int(fila["diagnostico"]),
        asimetria=int(fila["asimetria"]),
        colores=fila["colores"].split(),
        estructuras={e: bool(fila[e]) for e in ESTRUCTURAS},
    )
    # --------------------------------------------------------------------------


def armar_particion(tabla):
    """Divide las imágenes en entrenamiento y prueba y guarda datos/particion.csv.

    La partición tiene que ser estratificada por diagnóstico (misma proporción
    de cada clase en ambos conjuntos) y reproducible (SEMILLA). Se genera UNA
    vez y se sube al repo, así todos usan la misma.
    Columnas de particion.csv: nombre, diagnostico, conjunto ("entrenamiento"/"prueba").

    Hecho (M0): sklearn.model_selection.train_test_split con stratify.
    """
    # AGREGADO (M0): estratificada por diagnóstico y reproducible (SEMILLA)
    entrenamiento, prueba = train_test_split(
        tabla[["nombre", "diagnostico"]],
        test_size=FRACCION_PRUEBA,
        stratify=tabla["diagnostico"],
        random_state=SEMILLA,
    )
    particion = pd.concat([entrenamiento.assign(conjunto="entrenamiento"),
                           prueba.assign(conjunto="prueba")])
    particion = particion.sort_values("nombre").reset_index(drop=True)
    particion.to_csv(PARTICION, index=False)
    return particion


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
