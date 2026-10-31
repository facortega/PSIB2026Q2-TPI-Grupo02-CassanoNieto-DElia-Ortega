"""Estilo común para que todas las figuras del informe se vean iguales.

Reglas para TODAS las figuras:
  - grupos: nevo común en azul, sospechosa (atípico + melanoma) en naranja;
    si se separan los tres diagnósticos, el melanoma va en rojo oscuro.
  - contornos: referencia de PH2 en blanco discontinuo, segmentación automática
    en amarillo continuo. Así nada depende sólo del color.
  - imágenes sin ejes y con el nombre de la imagen de PH2 en el título.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_FIGURAS = RAIZ / "informe" / "figuras"
CARPETA_TABLAS = RAIZ / "informe" / "tablas"

COLOR_GRUPO = {"comun": "#2a78d6", "sospechosa": "#eb6834"}
COLOR_DIAGNOSTICO = {0: "#2a78d6", 1: "#eb6834", 2: "#8c1d18"}
CONTORNO_REFERENCIA = {"colors": "white", "linestyles": "--", "linewidths": 1.2}
CONTORNO_AUTOMATICO = {"colors": "#ffd400", "linestyles": "-", "linewidths": 1.2}


def aplicar_estilo():
    """Llamarla una vez al principio de cada cuaderno."""
    cm = 1 / 2.54
    mpl.rcParams.update({
        "figure.figsize": (16 * cm, 7 * cm),   # 16 cm = ancho del texto en el informe
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,                      # las imágenes van como mapa de bits dentro del PDF
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "image.interpolation": "nearest",
    })


def mostrar(ax, img, titulo="", **kwargs):
    """imshow sin ejes. Para imágenes de un canal usa escala de grises por defecto."""
    if img.ndim == 2:
        kwargs.setdefault("cmap", "gray")
    ax.imshow(img, **kwargs)
    ax.set_title(titulo)
    ax.set_axis_off()
    ax.grid(False)


def contorno(ax, mascara, referencia=False):
    """Dibuja el contorno de una máscara sobre una imagen ya mostrada."""
    estilo = CONTORNO_REFERENCIA if referencia else CONTORNO_AUTOMATICO
    ax.contour(mascara.astype(float), levels=[0.5], **estilo)


def superponer(ax, img, mascara, color=(1.0, 0.85, 0.0), alfa=0.4):
    """Muestra la imagen con la máscara pintada encima (semitransparente)."""
    capa = np.zeros((*mascara.shape, 4))
    capa[mascara] = (*color, alfa)
    ax.imshow(img)
    ax.imshow(capa)
    ax.set_axis_off()


def guardar_figura(fig, nombre):
    """Guarda la figura en informe/figuras/<nombre>.pdf, lista para el informe."""
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_FIGURAS / f"{nombre}.pdf"
    fig.savefig(ruta)
    return ruta


def guardar_tabla(df, nombre, decimales=2):
    """Guarda una tabla en informe/tablas/<nombre>.csv y <nombre>.tex (booktabs, coma decimal)."""
    CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(CARPETA_TABLAS / f"{nombre}.csv", index=False)
    tex = df.to_latex(index=False, float_format=lambda x: f"{x:.{decimales}f}".replace(".", "{,}"))
    (CARPETA_TABLAS / f"{nombre}.tex").write_text(tex, encoding="utf-8")
    return CARPETA_TABLAS / f"{nombre}.tex"
