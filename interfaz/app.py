"""Interfaz gráfica (opcional según la consigna, comprometida en el anteproyecto).

Uso, desde la carpeta del repo:

    python interfaz/app.py

1. Elegir una imagen de PH2 (conjunto de prueba) o una imagen propia.
2. Procesar: corre el algoritmo en segundo plano y muestra el avance en el registro. Usa los
   métodos que elegimos en cada módulo (el cierre para los pelos en M1, el crecimiento de
   regiones en M3), los mismos que usa main.py: el usuario no elige métodos.
3. Ver los resultados: A, B, C, D, el TDS, el TDS extendido y la clasificación, comparados
   con PH2; los ejes de simetría, el borde por sector y el mapa de colores; las zonas y los
   mapas de textura; los atributos GLCM y los contrastes entre zonas.

Si una etapa falla, el registro lo dice y la interfaz sigue con lo que se pueda calcular; si
falla la segmentación, la imagen no se analiza (no se reemplaza por la máscara de PH2).
La interfaz NO tiene lógica propia: sólo llama a las funciones de modulos/ (así lo que se
ve es exactamente lo que se evalúa).

Los colores siguen el tema claro u oscuro del sistema; el botón de arriba a la derecha cambia
de uno a otro.

"""

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")   # Los gráficos se insertan en la ventana con FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulos.herramientas import graficos, procesamiento_completo  # noqa: E402
from modulos.m0_carga_de_datos import carga_de_datos  # noqa: E402
from modulos.m1_mejoramiento import mejoramiento  # noqa: E402
from modulos.m3_segmentacion import segmentacion  # noqa: E402
from modulos.m4_criterios_abc import criterios_abc  # noqa: E402
from modulos.m5_textura_criterio_d import textura_criterio_d  # noqa: E402
from modulos.m6_zonas_y_mapas import zonas_y_mapas  # noqa: E402
from modulos.m7_clasificacion_resultados import clasificacion_resultados as m7  # noqa: E402

TABLA = Path(__file__).resolve().parents[1] / "resultados" / "caracteristicas.csv"
LADO_MAXIMO = 768   # píxeles: las imágenes propias se achican al tamaño de PH2
TIPOS_IMAGEN = [("Imágenes", "*.png *.jpg *.jpeg *.bmp *.PNG *.JPG *.JPEG *.BMP"), ("Todos los archivos", "*.*")]
TOTAL_PASOS = 6
METODO_PELOS = "cierre"   # el elegido en M1; es el mismo que usa procesamiento_completo.procesar
NOMBRES_METODO = {"ninguno": "ninguno", "mediana": "un filtro de mediana", "cierre": "un cierre morfológico",
                  "otsu": "el umbral de Otsu", "crecimiento": "crecimiento de regiones", "watershed": "watershed",
                  "kmeans": "k-means", "chan_vese": "Chan-Vese"}

NOMBRES_GLCM = {"energia": "Energía", "entropia": "Entropía", "contraste": "Contraste", "idm": "Homogeneidad (IDM)",
                "correlacion": "Correlación", "cluster_shade": "Cluster shade", "cluster_prominence": "Cluster prominence"}
NOMBRES_CONTRASTE = {"H": "H (tono)", "S": "S (saturación)", "V": "V (brillo)", "contraste": "Contraste", "entropia": "Entropía"}
NOMBRES_COLOR = {"blanco": "blanco", "rojo": "rojo", "marron_claro": "marrón claro", "marron_oscuro": "marrón oscuro",
                 "azul_grisaceo": "azul grisáceo", "negro": "negro"}
NOMBRES_ESTRUCTURA = {"red_pigmento": "red atípica", "puntos_globulos": "puntos y glóbulos", "estrias": "estrías",
                      "regresion": "regresión", "velo": "velo"}
NOMBRES_TERMINO = {"zona_periferia_vs_centro_S": "la saturación periferia − centro",
                   "zona_interfase_S": "la saturación piel − lesión"}
# Con qué se pinta cada color en el mapa de M4 (los mismos del cuaderno de M4)
PINTURA = {"blanco": "#ffffff", "rojo": "#d62728", "marron_claro": "#c08a4b", "marron_oscuro": "#6b3a1e",
           "azul_grisaceo": "#6a86a8", "negro": "#111111"}
PINTURA_PIEL = "#999999"   # píxeles de la lesión que se parecen más a la piel que a cualquier color

TEMA = None   # None: el del sistema operativo; "claro" u "oscuro" para forzarlo

# Colores de la interfaz en cada tema: los mismos de la aplicación de pupilometría
PALETAS = {
    "claro": dict(fondo="#f3f2ee", superficie="#fbfbfa", panel="#e6e5df", texto="#1f1f1e",
                  texto2="#52514e", tenue="#a5a39b", borde="#d0cec6", acento="#2a78d6",
                  acento_activo="#2368bd", seleccion="#cfe0f6", grilla="#e4e3dd",
                  ok="#1a7f4b", aviso="#a86b00", error="#c0392b",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "oscuro": dict(fondo="#1a1a19", superficie="#232322", panel="#2e2e2c", texto="#f0efec",
                   texto2="#c3c2b7", tenue="#77766f", borde="#41413d", acento="#3987e5",
                   acento_activo="#4a95ec", seleccion="#28456b", grilla="#33332f",
                   ok="#4cc38a", aviso="#e0a33a", error="#ff6b5e",
                   series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
PALETA = PALETAS["claro"]   # La define la ventana principal al arrancar


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def a_foto(img, ancho_max, alto_max):
    """Convierte una imagen en una foto que Tk puede mostrar, sin deformarla.

    Entra: img       imagen RGB float [0, 1]
           ancho_max, alto_max   tamaño máximo en píxeles
    Sale : ImageTk.PhotoImage que entra en ese tamaño
    """
    alto, ancho = img.shape[:2]
    escala = min(ancho_max / ancho, alto_max / alto)
    tam = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
    return ImageTk.PhotoImage(Image.fromarray((img * 255).round().astype(np.uint8)).resize(tam, Image.LANCZOS))


def numero(valor, decimales=2):
    """Da formato a un número, con coma decimal.

    Entra: valor      número, o None o NaN si no se pudo calcular
           decimales  int
    Sale : str; "—" si no hay valor
    """
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "—"
    return f"{valor:.{decimales}f}".replace(".", ",")


def sistema_en_modo_oscuro():
    """Averigua si macOS o GNOME usan el tema oscuro.

    Entra: nada
    Sale : bool; False si no se puede averiguar
    """
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                               capture_output=True, text=True, timeout=2)
            return r.stdout.strip().lower() == "dark"
        r = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                           capture_output=True, text=True, timeout=2)
        return "dark" in r.stdout
    except Exception:   # Sin la clave, el comando o el escritorio: se usa el tema claro
        return False


def ajuste_entrenamiento():
    """Busca lo que se ajusta con las imágenes de entrenamiento: D y los umbrales de M7.

    Entra: nada; lee resultados/caracteristicas.csv (lo genera python main.py)
    Sale : dict con umbrales_estructuras (el umbral de cada estructura de D), entrenamiento
           (las filas de entrenamiento, con D) y los umbrales de Youden del TDS y del TDS
           extendido. None si la tabla todavía no existe.
    """
    if not TABLA.exists():
        return None
    tabla = procesamiento_completo.completar_d(pd.read_csv(TABLA, dtype={"nombre": str}))
    entrenamiento = tabla[tabla["conjunto"] == "entrenamiento"]
    sospechosa = entrenamiento["sospechosa"].to_numpy(bool)
    tds = m7.tds(entrenamiento["A"], entrenamiento["B"], entrenamiento["C"], entrenamiento["D"])
    extendido = m7.tds_extendido(entrenamiento, m7.TERMINOS_EXTRA, entrenamiento)
    return dict(umbrales_estructuras=tabla.attrs["umbrales_estructuras"], entrenamiento=entrenamiento,
                umbral_tds=m7.umbral_youden(tds, sospechosa), umbral_extendido=m7.umbral_youden(extendido, sospechosa))


def clasificar(tds):
    """Traduce el TDS a la clasificación de la regla, con el color que le corresponde.

    Entra: tds   float
    Sale : tupla (texto, clave del color en la paleta)
    """
    if tds > m7.UMBRAL_MELANOMA:
        return f"Probable melanoma (TDS mayor que {numero(m7.UMBRAL_MELANOMA)})", "error"
    if tds > m7.UMBRAL_SOSPECHOSA:
        return f"Sospechosa (TDS mayor que {numero(m7.UMBRAL_SOSPECHOSA)})", "aviso"
    return f"Benigna (TDS hasta {numero(m7.UMBRAL_SOSPECHOSA)})", "ok"


class Grafico(ttk.Frame):
    """Una figura de matplotlib dentro de la ventana, con su barra de herramientas (zoom, guardar)."""

    def __init__(self, padre):
        """Crea la figura y su barra.

        Entra: padre   widget que la contiene
        Sale : nada
        """
        super().__init__(padre)
        self.figura = Figure(layout="constrained")
        self.lienzo = FigureCanvasTkAgg(self.figura, master=self)
        self.barra = None
        self.rehacer_barra()
        self.lienzo.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar()

    def rehacer_barra(self):
        """Crea la barra de herramientas con los colores del tema actual.

        Al cambiar de tema se rehace: matplotlib elige los íconos según el fondo, y Tk no
        recolorea bien los botones de una barra ya creada.

        Entra: nada
        Sale : nada
        """
        vieja = self.barra
        if vieja is not None:
            for conexion in (vieja._id_press, vieja._id_release, vieja._id_drag):
                self.lienzo.mpl_disconnect(conexion)
            if self.lienzo.widgetlock.isowner(vieja):   # si quedó activo el zoom o el desplazamiento
                self.lienzo.widgetlock.release(vieja)
            vieja.destroy()
        self.barra = NavigationToolbar2Tk(self.lienzo, self, pack_toolbar=False)
        self.barra.update()
        widget = self.lienzo.get_tk_widget()
        self.barra.pack(side="bottom", fill="x", **({"before": widget} if widget.winfo_manager() else {}))

    def limpiar(self):
        """Borra la figura y le pone el fondo del tema actual.

        Entra: nada
        Sale : nada
        """
        self.figura.clear()
        self.figura.set_facecolor(PALETA["superficie"])
        self.lienzo.get_tk_widget().configure(background=PALETA["superficie"])

    def mensaje(self, texto):
        """Deja la figura vacía, con un mensaje en el centro.

        Entra: texto   str
        Sale : nada
        """
        self.limpiar()
        self.figura.text(0.5, 0.5, texto, ha="center", va="center", color=PALETA["texto2"], fontsize=11)
        self.lienzo.draw_idle()


# ------------------------------------------------------------
# Ventana principal
# ------------------------------------------------------------

class Aplicacion(tk.Tk):

    def __init__(self):
        """Arma la ventana y carga la primera imagen disponible.

        Entra: nada
        Sale : nada
        """
        super().__init__()
        self.title("Lunares sospechosos · Regla ABCD")
        self.geometry(f"{min(1400, self.winfo_screenwidth() - 40)}x{min(860, self.winfo_screenheight() - 90)}")
        self.minsize(1050, 700)
        self._estilos()

        self.img = self.ref = self.caso = None
        self.titulo = ""
        self.res = None                # Resultados del último procesamiento
        self.cola = queue.Queue()      # Lo que manda el trabajo en segundo plano a la ventana
        self.version = 0               # Para descartar resultados de una imagen anterior
        self.procesando = False
        self.fotos = {}                # Tk borra las fotos si nadie guarda una referencia

        self._armar_ventana()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(50, self._revisar_cola)
        if self.nombres_ph2:
            self.elegir_ph2()

    # ---------------- Aspecto ----------------

    def _estilos(self):
        """Prepara la fuente y el tema inicial.

        "clam" es el único tema de ttk que respeta todos los colores en todos los sistemas; los
        nativos mezclan colores del sistema con los propios y en modo oscuro quedan textos ilegibles.

        Entra: nada
        Sale : nada
        """
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=12 if sys.platform == "darwin" else 10)
        self.familia = base.actual("family")
        self.estilo = ttk.Style(self)
        self.estilo.theme_use("clam")
        self._aplicar_tema(TEMA or ("oscuro" if sistema_en_modo_oscuro() else "claro"))

    def _aplicar_tema(self, nombre):
        """Aplica los colores de un tema a los estilos de ttk y a los widgets clásicos de Tk.

        Entra: nombre   "claro" u "oscuro"
        Sale : nada
        """
        global PALETA
        self.tema = nombre
        PALETA = p = PALETAS[nombre]
        for opcion, color in {"*Background": p["fondo"], "*Foreground": p["texto"],
                              "*activeBackground": p["panel"], "*activeForeground": p["texto"],
                              "*selectBackground": p["seleccion"], "*selectForeground": p["texto"],
                              "*selectColor": p["panel"], "*highlightBackground": p["fondo"],
                              "*highlightColor": p["acento"], "*insertBackground": p["texto"],
                              "*troughColor": p["grilla"], "*disabledForeground": p["tenue"],
                              "*TCombobox*Listbox.background": p["superficie"],
                              "*TCombobox*Listbox.foreground": p["texto"],
                              "*TCombobox*Listbox.selectBackground": p["seleccion"],
                              "*TCombobox*Listbox.selectForeground": p["texto"]}.items():
            self.option_add(opcion, color)
        self.configure(background=p["fondo"])

        estilo, familia = self.estilo, self.familia
        plano = lambda color: dict(lightcolor=color, darkcolor=color)   # Sin el relieve 3D de clam
        activo = lambda normal, activo_: [("disabled", normal), ("pressed", activo_), ("active", activo_)]

        estilo.configure(".", background=p["fondo"], foreground=p["texto"], fieldbackground=p["superficie"],
                         bordercolor=p["borde"], troughcolor=p["grilla"], selectbackground=p["seleccion"],
                         selectforeground=p["texto"], insertcolor=p["texto"], arrowcolor=p["texto2"],
                         focuscolor=p["acento"], **plano(p["fondo"]))
        estilo.map(".", foreground=[("disabled", p["tenue"])])

        estilo.configure("Titulo.TLabel", font=(familia, 20, "bold"))
        estilo.configure("Sub.TLabel", foreground=p["texto2"])
        estilo.configure("Dato.TLabel", font=(familia, 12, "bold"))
        estilo.configure("Flecha.TLabel", font=(familia, 16), foreground=p["texto2"])
        for clave in ("ok", "aviso", "error"):
            estilo.configure(f"Clase{clave}.TLabel", font=(familia, 15, "bold"), foreground=p[clave])
        estilo.configure("ClaseSin.TLabel", font=(familia, 12), foreground=p["texto2"])
        estilo.configure("TLabelframe", bordercolor=p["borde"], **plano(p["fondo"]))
        estilo.configure("TLabelframe.Label", foreground=p["texto"], font=(familia, 12, "bold"))

        estilo.configure("TButton", background=p["panel"], bordercolor=p["borde"], padding=(10, 5), **plano(p["panel"]))
        estilo.map("TButton", background=activo(p["fondo"], p["borde"]), lightcolor=activo(p["fondo"], p["borde"]),
                   darkcolor=activo(p["fondo"], p["borde"]), foreground=[("disabled", p["tenue"])])
        estilo.configure("Accion.TButton", font=(familia, 13, "bold"), padding=(10, 8), background=p["acento"],
                         foreground="#ffffff", bordercolor=p["acento"], **plano(p["acento"]))
        estilo.map("Accion.TButton", background=activo(p["panel"], p["acento_activo"]),
                   lightcolor=activo(p["panel"], p["acento_activo"]), darkcolor=activo(p["panel"], p["acento_activo"]),
                   bordercolor=[("disabled", p["borde"])], foreground=[("disabled", p["tenue"])])

        estilo.configure("TCombobox", background=p["panel"], bordercolor=p["borde"], **plano(p["superficie"]))
        estilo.map("TCombobox", fieldbackground=[("disabled", p["fondo"]), ("readonly", p["superficie"])],
                   foreground=[("disabled", p["tenue"]), ("readonly", p["texto"])],
                   selectbackground=[("readonly", p["superficie"])], selectforeground=[("readonly", p["texto"])])

        estilo.configure("TNotebook", bordercolor=p["borde"], tabmargins=(0, 0, 0, 0), **plano(p["fondo"]))
        estilo.configure("TNotebook.Tab", background=p["panel"], foreground=p["texto2"], bordercolor=p["borde"],
                         padding=(12, 5), **plano(p["panel"]))
        estilo.map("TNotebook.Tab", background=[("selected", p["fondo"]), ("active", p["borde"])],
                   lightcolor=[("selected", p["fondo"])], darkcolor=[("selected", p["fondo"])],
                   foreground=[("selected", p["texto"]), ("disabled", p["tenue"])])

        estilo.configure("Treeview", background=p["superficie"], fieldbackground=p["superficie"],
                         foreground=p["texto"], bordercolor=p["borde"], rowheight=28, **plano(p["superficie"]))
        estilo.map("Treeview", background=[("selected", p["seleccion"])], foreground=[("selected", p["texto"])])
        estilo.configure("Treeview.Heading", background=p["panel"], foreground=p["texto"], bordercolor=p["borde"],
                         font=(familia, 12, "bold"), **plano(p["panel"]))
        estilo.map("Treeview.Heading", background=[("active", p["borde"])])

        estilo.configure("Horizontal.TProgressbar", background=p["acento"], troughcolor=p["grilla"],
                         bordercolor=p["grilla"], **plano(p["acento"]))
        estilo.configure("TScrollbar", background=p["panel"], troughcolor=p["superficie"], bordercolor=p["borde"],
                         arrowcolor=p["texto2"], **plano(p["panel"]))
        estilo.map("TScrollbar", background=[("active", p["borde"])])

    def cambiar_tema(self):
        """Pasa de modo claro a oscuro o al revés, y vuelve a dibujar lo que depende del tema.

        Entra: nada
        Sale : nada
        """
        self._aplicar_tema("claro" if self.tema == "oscuro" else "oscuro")
        p = PALETA
        self.texto.configure(background=p["superficie"], foreground=p["texto"], insertbackground=p["texto"],
                             selectbackground=p["seleccion"], selectforeground=p["texto"])
        for etiqueta in ("ok", "aviso", "error"):
            self.texto.tag_configure(etiqueta, foreground=p[etiqueta])
        for grafico in (self.g_lesion, self.g_criterios, self.g_zonas):
            grafico.rehacer_barra()
        self._mostrar_resultados()
        self._texto_boton_tema()

    def _texto_boton_tema(self):
        """Actualiza el texto del botón de tema.

        Entra: nada
        Sale : nada
        """
        self.btn_tema.configure(text="Cambiar a modo claro" if self.tema == "oscuro" else "Cambiar a modo oscuro")

    # ---------------- Armado de la ventana ----------------

    def _armar_ventana(self):
        """Arma la cabecera, los pasos de la izquierda y las pestañas de la derecha.

        Entra: nada
        Sale : nada
        """
        cabecera = ttk.Frame(self, padding=(18, 14, 18, 6))
        cabecera.pack(fill="x")
        cabecera.columnconfigure(0, weight=1)
        titulos = ttk.Frame(cabecera)
        titulos.grid(row=0, column=0, sticky="w")
        ttk.Label(titulos, text="Lunares sospechosos · Regla ABCD", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(titulos, text="Procesamiento de Señales e Imágenes Biomédicas · ITBA · "
                                "Herramienta académica, no es un dispositivo de diagnóstico",
                  style="Sub.TLabel").pack(anchor="w")
        self.btn_tema = ttk.Button(cabecera, command=self.cambiar_tema)
        self.btn_tema.grid(row=0, column=1, sticky="ne")
        self._texto_boton_tema()

        cuerpo = ttk.Frame(self, padding=(18, 8, 18, 18))
        cuerpo.pack(fill="both", expand=True)
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        izquierda = ttk.Frame(cuerpo, width=400)
        izquierda.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        self.pestanas = ttk.Notebook(cuerpo)
        self.pestanas.grid(row=0, column=1, sticky="nsew")

        self._armar_imagen(izquierda)
        self._armar_mejoramiento(izquierda)
        self._armar_segmentacion(izquierda)
        self._armar_procesamiento(izquierda)
        self._armar_registro()
        self._armar_resultado()
        self._armar_criterios()
        self._armar_zonas()
        self._armar_numeros()

    def _armar_imagen(self, padre):
        """Arma el paso 1: de dónde sale la imagen.

        Entra: padre   widget que lo contiene
        Sale : nada
        """
        marco = ttk.LabelFrame(padre, text=" 1 · Imagen ", padding=12)
        marco.pack(fill="x")
        self.nombres_ph2 = carga_de_datos.nombres("prueba") if carga_de_datos.hay_datos() else []
        ttk.Label(marco, text="PH2, conjunto de prueba:", style="Sub.TLabel").pack(anchor="w")
        self.var_ph2 = tk.StringVar(value=self.nombres_ph2[0] if self.nombres_ph2 else "")
        self.cmb_ph2 = ttk.Combobox(marco, textvariable=self.var_ph2, values=self.nombres_ph2, state="readonly")
        self.cmb_ph2.pack(fill="x", pady=(2, 8))
        self.cmb_ph2.bind("<<ComboboxSelected>>", lambda _: self.elegir_ph2())
        fila = ttk.Frame(marco)
        fila.pack(fill="x")
        self.btn_propia = ttk.Button(fila, text="Imagen propia…", command=self.elegir_propia)
        self.btn_propia.pack(side="left")
        self.lbl_imagen = ttk.Label(marco, style="Dato.TLabel", wraplength=370)
        self.lbl_imagen.pack(anchor="w", pady=(10, 2))
        self.lbl_detalle = ttk.Label(marco, style="Sub.TLabel", wraplength=370, justify="left")
        self.lbl_detalle.pack(anchor="w")

    def _armar_mejoramiento(self, padre):
        """Arma el paso 2: método para quitar pelos y vista previa del resultado.

        Entra: padre   widget que lo contiene
        Sale : nada
        """
        marco = ttk.LabelFrame(padre, text=" 2 · Mejoramiento (M1) ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        ttk.Label(marco, text=f"Pelos: se quitan con {NOMBRES_METODO[METODO_PELOS]}, el método elegido en M1.",
                  style="Sub.TLabel", wraplength=370, justify="left").pack(anchor="w")

        imagenes = ttk.Frame(marco)
        imagenes.pack(anchor="w", pady=(8, 0))
        self.img_pelos = ttk.Label(imagenes)
        self.img_pelos.grid(row=0, column=0)
        ttk.Label(imagenes, text="→", style="Flecha.TLabel").grid(row=0, column=1, padx=8)
        self.img_sin_pelos = ttk.Label(imagenes)
        self.img_sin_pelos.grid(row=0, column=2)
        self.lbl_pelos = ttk.Label(imagenes, style="Sub.TLabel", justify="center")
        self.lbl_pelos.grid(row=1, column=0, pady=(4, 0))
        self.lbl_sin_pelos = ttk.Label(imagenes, style="Sub.TLabel", justify="center")
        self.lbl_sin_pelos.grid(row=1, column=2, pady=(4, 0))

    def _armar_segmentacion(self, padre):
        """Arma el paso 3: qué método de segmentación usa el programa.

        Entra: padre   widget que lo contiene
        Sale : nada
        """
        marco = ttk.LabelFrame(padre, text=" 3 · Segmentación (M3) ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        ttk.Label(marco, text=f"La lesión se segmenta con {NOMBRES_METODO[segmentacion.METODO_ELEGIDO]}, "
                              "el método elegido en M3.",
                  style="Sub.TLabel", wraplength=370, justify="left").pack(anchor="w")

    def _armar_procesamiento(self, padre):
        """Arma el paso 4: el botón de procesar y el avance.

        Entra: padre   widget que lo contiene
        Sale : nada
        """
        marco = ttk.LabelFrame(padre, text=" 4 · Procesamiento ", padding=12)
        marco.pack(fill="x", pady=(10, 0))
        self.btn_procesar = ttk.Button(marco, text="▶  Procesar imagen", style="Accion.TButton", command=self.procesar)
        self.btn_procesar.pack(fill="x")
        self.barra = ttk.Progressbar(marco, maximum=TOTAL_PASOS)
        self.barra.pack(fill="x", pady=(8, 4))
        self.lbl_estado = ttk.Label(marco, style="Sub.TLabel", wraplength=370, justify="left")
        self.lbl_estado.pack(anchor="w")

    def _armar_registro(self):
        """Arma la pestaña Registro: el avance de cada etapa y, si algo falla, el error.

        Entra: nada
        Sale : nada
        """
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Registro")
        p = PALETA
        self.texto = tk.Text(marco, wrap="word", state="disabled", relief="flat", padx=10, pady=8, highlightthickness=1,
                             font=("Menlo" if sys.platform == "darwin" else "Courier", 12), background=p["superficie"],
                             foreground=p["texto"], selectbackground=p["seleccion"], selectforeground=p["texto"])
        barra = ttk.Scrollbar(marco, command=self.texto.yview)
        self.texto.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)
        for etiqueta in ("ok", "aviso", "error"):
            self.texto.tag_configure(etiqueta, foreground=p[etiqueta])

    def _tabla(self, padre, columnas, anchos, alto):
        """Crea una tabla sin columna de árbol.

        Entra: padre     widget que la contiene
               columnas  lista de títulos
               anchos    lista con el ancho de cada columna, en píxeles
               alto      cantidad de filas visibles
        Sale : ttk.Treeview
        """
        tabla = ttk.Treeview(padre, columns=columnas, show="headings", height=alto, selectmode="none")
        for i, (columna, ancho) in enumerate(zip(columnas, anchos)):
            tabla.heading(columna, text=columna, anchor="w")
            tabla.column(columna, width=ancho, anchor="w", stretch=i == 0)
        return tabla

    def _armar_resultado(self):
        """Arma la pestaña Resultado: la lesión con su contorno, la clasificación y la tabla.

        Entra: nada
        Sale : nada
        """
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Resultado")
        marco.columnconfigure(0, weight=1)
        # La tabla tiene filas largas («¿Sospechosa? TDS extendido > 3,49»): la columna derecha
        # no se achica por debajo de lo que necesita, y el gráfico se queda con el resto.
        marco.columnconfigure(1, weight=0, minsize=640)
        marco.rowconfigure(0, weight=1)
        self.g_lesion = Grafico(marco)
        self.g_lesion.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        derecha = ttk.Frame(marco)
        derecha.grid(row=0, column=1, sticky="nsew")
        self.lbl_clase = ttk.Label(derecha, style="ClaseSin.TLabel", wraplength=620, justify="left")
        self.lbl_clase.pack(anchor="w", pady=(0, 10))
        self.t_resultado = self._tabla(derecha, ("Parámetro", "Calculado", "Según PH2"), (330, 130, 140), 11)
        self.t_resultado.pack(fill="x")
        self.lbl_resultado = ttk.Label(derecha, style="Sub.TLabel", wraplength=620, justify="left")
        self.lbl_resultado.pack(anchor="w", pady=(10, 0))

    def _armar_criterios(self):
        """Arma la pestaña de los criterios A, B y C.

        Entra: nada
        Sale : nada
        """
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Criterios A, B y C")
        self.g_criterios = Grafico(marco)
        self.g_criterios.pack(fill="both", expand=True)

    def _armar_zonas(self):
        """Arma la pestaña de zonas y mapas de textura.

        Entra: nada
        Sale : nada
        """
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Zonas y textura")
        fila = ttk.Frame(marco)
        fila.pack(fill="x", pady=(0, 8))
        ttk.Label(fila, text="Mapa de textura:", style="Sub.TLabel").pack(side="left")
        self.var_mapa = tk.StringVar(value=NOMBRES_GLCM["entropia"])
        cmb = ttk.Combobox(fila, textvariable=self.var_mapa, values=list(NOMBRES_GLCM.values()), state="readonly", width=20)
        cmb.pack(side="left", padx=(8, 0))
        cmb.bind("<<ComboboxSelected>>", lambda _: self._pedir_mapa())
        ttk.Label(fila, text="Muestra en qué parte de la lesión está la textura: más claro, valor más alto.",
                  style="Sub.TLabel").pack(side="left", padx=(14, 0))
        self.g_zonas = Grafico(marco)
        self.g_zonas.pack(fill="both", expand=True)

    def _armar_numeros(self):
        """Arma la pestaña Números: atributos GLCM y contrastes entre zonas.

        Entra: nada
        Sale : nada
        """
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Números")
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=1)
        izquierda, derecha = ttk.Frame(marco), ttk.Frame(marco)
        izquierda.grid(row=0, column=0, sticky="new", padx=(0, 12))
        derecha.grid(row=0, column=1, sticky="new")
        ttk.Label(izquierda, text="Textura de toda la lesión (GLCM)", style="Dato.TLabel").pack(anchor="w", pady=(0, 6))
        self.t_glcm = self._tabla(izquierda, ("Atributo", "Valor"), (200, 120), len(NOMBRES_GLCM))
        self.t_glcm.pack(fill="x")
        ttk.Label(derecha, text="Contrastes entre zonas", style="Dato.TLabel").pack(anchor="w", pady=(0, 6))
        self.t_contrastes = self._tabla(derecha, ("Atributo", "Periferia − centro", "Piel − lesión"), (160, 140, 120), len(NOMBRES_CONTRASTE))
        self.t_contrastes.pack(fill="x")
        ttk.Label(derecha, text="En entrenamiento, con la máscara de M3, el que más separa a las sospechosas es la saturación "
                                "de periferia − centro.",
                  style="Sub.TLabel", wraplength=430, justify="left").pack(anchor="w", pady=(10, 0))

    # ---------------- Elegir imagen ----------------

    def elegir_ph2(self):
        """Carga la imagen de PH2 elegida en la lista.

        Entra: nada; lee la lista desplegable
        Sale : nada
        """
        caso = carga_de_datos.cargar_caso(self.var_ph2.get())
        detalle = (f"{carga_de_datos.DIAGNOSTICOS[caso.diagnostico]}  ·  "
                   f"{caso.imagen.shape[1]} x {caso.imagen.shape[0]} px")
        self._cargar(caso.imagen, caso.mascara_ref, caso.nombre, detalle, caso)

    def elegir_propia(self):
        """Abre el explorador de archivos y carga la imagen elegida, achicada si hace falta.

        Entra: nada
        Sale : nada
        """
        ruta = filedialog.askopenfilename(title="Elegir una imagen dermatoscópica", filetypes=TIPOS_IMAGEN)
        if not ruta:
            return
        try:
            imagen = Image.open(ruta).convert("RGB")
        except Exception as e:
            messagebox.showerror("No se pudo abrir la imagen", str(e))
            return
        imagen.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
        img = np.asarray(imagen, float) / 255
        detalle = f"{img.shape[1]} x {img.shape[0]} px  ·  sin máscara de referencia ni anotaciones"
        self._cargar(img, None, Path(ruta).name, detalle, None)

    def _cargar(self, img, ref, titulo, detalle, caso):
        """Deja lista una imagen para procesar y borra los resultados anteriores.

        Entra: img       imagen RGB float [0, 1]
               ref       array bool con la máscara de referencia, o None
               titulo    str, nombre de la imagen
               detalle   str, datos de la imagen
               caso      Caso de PH2, o None si no es de PH2
        Sale : nada
        """
        self.img, self.ref, self.titulo, self.caso = img, ref, titulo, caso
        self.lbl_imagen.configure(text=titulo)
        self.lbl_detalle.configure(text=detalle)
        self._invalidar()

    def _invalidar(self):
        """Borra los resultados, porque cambió la imagen, y muestra la vista previa.

        Entra: nada
        Sale : nada
        """
        self.version += 1   # Lo que siga llegando del trabajo anterior se descarta
        self.procesando = False
        self.res = None
        self.barra.configure(value=0)
        self._habilitar(True)
        self.lbl_estado.configure(text="Listo para procesar.")
        self._mostrar_miniatura(self.img_pelos, "pelos", self.img)
        self.lbl_pelos.configure(text="Original")
        self.img_sin_pelos.configure(image="")
        self.lbl_sin_pelos.configure(text="")
        self._mostrar_resultados()

    def _mostrar_miniatura(self, etiqueta, clave, img):
        """Muestra una imagen chica en el panel de la izquierda.

        Entra: etiqueta  ttk.Label donde va la imagen
               clave     str con la que se guarda la foto para que Tk no la borre
               img       imagen RGB float [0, 1]
        Sale : nada
        """
        self.fotos[clave] = a_foto(img, 170, 115)
        etiqueta.configure(image=self.fotos[clave])

    def _habilitar(self, si):
        """Habilita o bloquea los controles que no se pueden tocar mientras se procesa.

        Entra: si   bool
        Sale : nada
        """
        estado = ["!disabled"] if si else ["disabled"]
        self.btn_procesar.state(estado)
        self.btn_propia.state(estado)

    # ---------------- Procesamiento en segundo plano ----------------

    def procesar(self):
        """Arranca el procesamiento en segundo plano.

        Entra: nada; lee la imagen y los métodos elegidos
        Sale : nada
        """
        if self.procesando or self.img is None:
            return
        self.version += 1
        self.procesando = True
        self.res = None
        self._habilitar(False)
        self.barra.configure(value=0)
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.configure(state="disabled")
        self.pestanas.select(0)
        self._escribir(f"{self.titulo}\n\n")
        datos = (self.version, self.img, self.ref)
        threading.Thread(target=self._trabajo, args=datos, daemon=True).start()

    def _trabajo(self, version, img, ref):
        """Corre las etapas del algoritmo. Corre en otro hilo: sólo se comunica por la cola.

        Entra: version   int, para descartar el resultado si la imagen cambió mientras tanto
               img       imagen RGB float [0, 1]
               ref       array bool con la máscara de referencia, o None (sólo para el Dice)
        Sale : nada; manda los avances y al final ("listo", version, resultados) a la cola
        """
        def avisar(*mensaje):
            """Manda un mensaje a la ventana, marcado con la versión de este trabajo.

            Entra: *mensaje   tipo de mensaje ("texto", "paso", "listo" o "fallo") y sus datos
            Sale : nada
            """
            self.cola.put((mensaje[0], version, *mensaje[1:]))

        def etapa(modulo, funcion, *args):
            """Corre una función de un módulo; si falla, lo avisa en el registro.

            Entra: modulo    str, módulo al que pertenece (p. ej. "M3")
                   funcion   la función a correr
                   *args     lo que recibe esa función
            Sale : lo que devuelve la función, o None si falló
            """
            try:
                return funcion(*args)
            except Exception as e:   # que una etapa rota no frene a las demás
                avisar("texto", f"  ✖ {modulo}: {type(e).__name__}: {e}\n", "error")
            return None

        def paso(n, titulo):
            """Anuncia que empieza una etapa, en la barra de avance y en el registro.

            Entra: n        int, número de etapa
                   titulo   str, nombre de la etapa
            Sale : nada
            """
            avisar("paso", n, titulo)
            avisar("texto", f"[{n}/{TOTAL_PASOS}] {titulo}\n")

        res = dict(img=img, ref=ref, automatica=None, mascara=None, crit=None, D=None, atributo_d=None, tds=None,
                   tds_extendido=None, ajuste=None, glcm=None, z=None, contrastes=None, ejes=None, gradientes=None,
                   colores=None, dice=None, mapas={})
        inicio = time.perf_counter()

        paso(1, "Mejoramiento (M1)")
        salida = etapa("M1", mejoramiento.mejorar, img, METODO_PELOS)
        if salida is None:
            avisar("fallo", "No se pudo mejorar la imagen.")
            return
        res["mejorada"], res["valida"], res["pelos"] = salida
        avisar("texto", f"  ✔ Pelos en el {numero(100 * res['pelos'].mean(), 1)} % de la imagen\n", "ok")
        res["v"] = mejoramiento.a_hsv(res["mejorada"])[..., 2]

        paso(2, "Segmentación (M3)")
        res["automatica"] = res["mascara"] = etapa("M3", segmentacion.segmentar, res["mejorada"], res["valida"])
        if res["mascara"] is None:
            avisar("texto", "  ✖ Sin segmentación no se puede seguir con esta imagen\n", "error")
            avisar("listo", res, time.perf_counter() - inicio)
            return
        if ref is not None:
            res["dice"] = etapa("M3", segmentacion.dice, res["mascara"], ref)
            avisar("texto", f"  ✔ Dice con la máscara de PH2: {numero(res['dice'], 3)}\n", "ok")
        else:
            avisar("texto", "  ✔ Lesión segmentada\n", "ok")
        m = res["mascara"]

        paso(3, "Criterios A, B y C (M4)")
        res["crit"] = etapa("M4", criterios_abc.criterios, res["mejorada"], m)
        res["ejes"] = etapa("M4", criterios_abc.fracciones_asimetria, m)
        res["gradientes"] = etapa("M4", criterios_abc.gradiente_por_sector, res["mejorada"], m)
        res["colores"] = etapa("M4", criterios_abc.mapa_de_colores, res["mejorada"], m)
        if res["crit"] is not None:
            avisar("texto", f"  ✔ A = {res['crit']['A']}, B = {res['crit']['B']}, C = {res['crit']['C']}\n", "ok")

        paso(4, "Textura y criterio D (M5)")
        res["glcm"] = etapa("M5", textura_criterio_d.atributos_glcm, res["v"], m)
        ajuste = res["ajuste"] = etapa("M5", ajuste_entrenamiento)
        medidas = etapa("M5", textura_criterio_d.medidas_estructuras, res["mejorada"], m)
        if medidas is not None and ajuste is not None:
            detectadas = textura_criterio_d.detectar_estructuras(medidas, ajuste["umbrales_estructuras"])
            res["atributo_d"] = ", ".join(NOMBRES_ESTRUCTURA[e] for e, si in detectadas.items() if si) or "ninguna"
            res["D"] = sum(detectadas.values())
            avisar("texto", f"  ✔ D = {res['D']} (estructuras detectadas: {res['atributo_d']})\n", "ok")
        elif res["glcm"] is not None:
            avisar("texto", "  ✔ Atributos GLCM calculados\n", "ok")
            avisar("texto", "  ⚠ D sin estimar: falta resultados/caracteristicas.csv (python main.py)\n", "aviso")

        paso(5, "Zonas y mapas (M6)")
        res["z"] = etapa("M6", zonas_y_mapas.zonas, m)
        por_zona = etapa("M6", zonas_y_mapas.atributos_por_zona, res["mejorada"], m)
        if por_zona is not None:
            res["contrastes"] = etapa("M6", zonas_y_mapas.contrastes_entre_zonas, por_zona)
        atributo = self._atributo_mapa()
        mapa = etapa("M6", zonas_y_mapas.mapa_caracteristica, res["v"], m, atributo)
        if mapa is not None:
            res["mapas"][atributo] = mapa
        if res["contrastes"] is not None:
            avisar("texto", "  ✔ Zonas, contrastes y mapa de textura\n", "ok")

        paso(6, "TDS y clasificación (M7)")
        if res["crit"] is not None and res["D"] is not None:
            res["tds"] = etapa("M7", m7.tds, res["crit"]["A"], res["crit"]["B"], res["crit"]["C"], res["D"])
        if res["tds"] is not None and res["contrastes"] is not None:
            # Los términos extra se estandarizan con la media y el desvío de entrenamiento, como en M7
            fila = pd.DataFrame([{"A": res["crit"]["A"], "B": res["crit"]["B"], "C": res["crit"]["C"], "D": res["D"],
                                  **{f"zona_{k}": v for k, v in res["contrastes"].items()}}])
            extendido = etapa("M7", m7.tds_extendido, fila, m7.TERMINOS_EXTRA, ajuste["entrenamiento"])
            res["tds_extendido"] = None if extendido is None else float(extendido[0])
        if res["tds"] is not None:
            avisar("texto", f"  ✔ TDS = {numero(res['tds'])}; TDS extendido = {numero(res['tds_extendido'])}\n", "ok")
        else:
            avisar("texto", "  ⚠ Sin TDS: hacen falta A, B, C y D (ver los avisos de arriba)\n", "aviso")
        avisar("listo", res, time.perf_counter() - inicio)

    def _revisar_cola(self):
        """Lee lo que mandó el trabajo en segundo plano y actualiza la ventana.

        Entra: nada; lee la cola
        Sale : nada; se vuelve a programar para dentro de 50 ms
        """
        try:
            while True:
                tipo, version, *datos = self.cola.get_nowait()
                if version != self.version:
                    continue
                if tipo == "texto":
                    self._escribir(*datos)
                elif tipo == "paso":
                    self.barra.configure(value=datos[0] - 1)
                    self.lbl_estado.configure(text=f"Paso {datos[0]} de {TOTAL_PASOS}: {datos[1]}")
                elif tipo == "listo":
                    self._terminar(*datos)
                elif tipo == "fallo":
                    self.procesando = False
                    self._habilitar(True)
                    self.lbl_estado.configure(text=datos[0])
                elif tipo == "mapa" and self.res is not None:
                    self.res["mapas"][datos[0]] = datos[1]
                    self._dibujar_zonas()
        except queue.Empty:
            pass
        self.after(50, self._revisar_cola)

    def _escribir(self, texto, etiqueta=None):
        """Agrega texto al registro.

        Entra: texto     str
               etiqueta  "ok", "aviso" o "error" para darle color; None para el color normal
        Sale : nada
        """
        self.texto.configure(state="normal")
        self.texto.insert("end", texto, etiqueta or ())
        self.texto.configure(state="disabled")
        self.texto.see("end")

    def _terminar(self, res, segundos):
        """Muestra los resultados cuando termina el procesamiento.

        Entra: res       dict con los resultados
               segundos  float, cuánto tardó
        Sale : nada
        """
        self.procesando = False
        self.res = res
        self._habilitar(True)
        self.barra.configure(value=TOTAL_PASOS)
        self.lbl_estado.configure(text=f"Listo: procesada en {numero(segundos, 1)} s.")
        self._escribir(f"\nTiempo total: {numero(segundos, 1)} s\n")
        if "pelos" in res:
            copia = res["img"].copy()
            copia[res["pelos"]] = (0.0, 1.0, 1.0)
            self._mostrar_miniatura(self.img_pelos, "pelos", copia)
            self.lbl_pelos.configure(text=f"Pelos detectados\n{numero(100 * res['pelos'].mean(), 1)} % de la imagen")
            self._mostrar_miniatura(self.img_sin_pelos, "sin_pelos", res["mejorada"])
            self.lbl_sin_pelos.configure(text="Sin pelos\nentra a la segmentación")
        self._mostrar_resultados()
        self.pestanas.select(1)

    def _atributo_mapa(self):
        """Traduce el atributo elegido en la lista al nombre que usan los módulos.

        Entra: nada; lee la lista desplegable
        Sale : str, una de las claves de NOMBRES_GLCM
        """
        return next(k for k, nombre in NOMBRES_GLCM.items() if nombre == self.var_mapa.get())

    def _pedir_mapa(self):
        """Dibuja el mapa elegido; si todavía no está calculado, lo calcula en segundo plano.

        Entra: nada
        Sale : nada
        """
        if self.res is None or self.res["mascara"] is None:
            return
        atributo = self._atributo_mapa()
        if atributo in self.res["mapas"]:
            self._dibujar_zonas()
            return
        v, m, version = self.res["v"], self.res["mascara"], self.version

        def trabajo():
            """Calcula el mapa en otro hilo y manda el resultado a la cola.

            Entra: nada; usa el atributo, la imagen y la máscara de la función de afuera
            Sale : nada
            """
            try:
                self.cola.put(("mapa", version, atributo, zonas_y_mapas.mapa_caracteristica(v, m, atributo)))
            except Exception as e:
                self.cola.put(("texto", version, f"✖ M6: {type(e).__name__}: {e}\n", "error"))

        threading.Thread(target=trabajo, daemon=True).start()

    # ---------------- Mostrar resultados ----------------

    def _mostrar_resultados(self):
        """Actualiza todas las pestañas de resultados, o las deja vacías si no hay resultados.

        Entra: nada; lee self.res
        Sale : nada
        """
        self._dibujar_lesion()
        self._llenar_resultado()
        self._dibujar_criterios()
        self._dibujar_zonas()
        self._llenar_numeros()

    def _titulo_eje(self, ax, texto):
        """Pone un título con el color del tema (los de graficos.mostrar quedan negros).

        Entra: ax      eje de matplotlib
               texto   str
        Sale : nada
        """
        ax.set_title(texto, color=PALETA["texto"], fontsize=11)

    def _dibujar_lesion(self):
        """Dibuja la lesión con sus contornos: amarillo la segmentación, blanco la referencia.

        Entra: nada; lee self.res, o la imagen elegida si todavía no se procesó
        Sale : nada
        """
        grafico = self.g_lesion
        grafico.limpiar()
        ax = grafico.figura.add_subplot()
        if self.res is None or "mejorada" not in self.res:
            graficos.mostrar(ax, self.img)
        else:
            graficos.mostrar(ax, self.res["mejorada"])
            if self.res["ref"] is not None:
                graficos.contorno(ax, self.res["ref"], referencia=True)
            if self.res["automatica"] is not None:
                graficos.contorno(ax, self.res["automatica"])
        self._titulo_eje(ax, self.titulo)
        grafico.lienzo.draw_idle()

    def _llenar_resultado(self):
        """Llena la clasificación y la tabla de la pestaña Resultado.

        Entra: nada; lee self.res
        Sale : nada
        """
        tabla = self.t_resultado
        tabla.delete(*tabla.get_children())
        if self.res is None:
            self.lbl_clase.configure(text="Apretá «Procesar imagen» para ver el resultado.", style="ClaseSin.TLabel")
            self.lbl_resultado.configure(text="")
            return
        res, c = self.res, self.caso
        crit = res["crit"]
        if res["tds"] is not None:
            texto, color = clasificar(res["tds"])
            self.lbl_clase.configure(text=texto, style=f"Clase{color}.TLabel")
        else:
            self.lbl_clase.configure(text="No se pudo clasificar: el registro dice qué falló.", style="ClaseSin.TLabel")

        def fila(nombre, calculado=None, ph2=None):
            """Agrega una fila a la tabla, con "—" donde no hay valor.

            Entra: nombre            str, nombre del parámetro
                   calculado, ph2    valores de cada columna, o None
            Sale : nada
            """
            tabla.insert("", "end", values=(nombre, "—" if calculado is None else calculado, "—" if ph2 is None else ph2))

        fila("Diagnóstico", None, carga_de_datos.DIAGNOSTICOS[c.diagnostico] if c else None)
        fila("A · Asimetría (0 a 2)", crit["A"] if crit else None, c.asimetria if c else None)
        fila("B · Borde (0 a 8)", crit["B"] if crit else None)
        fila("C · Colores (1 a 6)", crit["C"] if crit else None, c.c_dermatologo if c else None)
        fila("D · Estructuras (0 a 5)", res["D"], c.d_dermatologo if c else None)
        fila("TDS", numero(res["tds"]) if res["tds"] is not None else None)
        fila("TDS extendido", numero(res["tds_extendido"]) if res["tds_extendido"] is not None else None)
        ajuste = res["ajuste"]
        if ajuste is not None:
            def si_no(valor):
                return None if valor is None else ("Sí" if valor else "No")
            ph2 = si_no(c.sospechosa) if c else None
            for nombre, puntaje, umbral in (("TDS", res["tds"], ajuste["umbral_tds"]),
                                            ("TDS extendido", res["tds_extendido"], ajuste["umbral_extendido"])):
                fila(f"¿Sospechosa? {nombre} > {numero(umbral)}", si_no(None if puntaje is None else puntaje > umbral), ph2)
        fila("Circularidad", numero(crit["circularidad"]) if crit else None)
        fila("Dice de la segmentación", numero(res["dice"], 3) if res["dice"] is not None else None)

        notas = []
        if crit:
            notas.append("Colores: " + (", ".join(NOMBRES_COLOR[k] for k in crit["colores"]) or "—") + ".")
        if res["atributo_d"]:
            notas.append(f"Estructuras detectadas: {res['atributo_d']}; umbrales ajustados en entrenamiento.")
        if res["tds"] is not None:
            notas.append(f"El cartel usa los umbrales clínicos de la regla. Las filas «¿Sospechosa?» usan los "
                         f"umbrales de Youden elegidos en entrenamiento (M7); el TDS extendido suma "
                         f"{' y '.join(NOMBRES_TERMINO.get(t, t) for t in m7.TERMINOS_EXTRA)}.")
        if res["automatica"] is not None:
            notas.append("Contorno amarillo: segmentación." + (" Blanco punteado: referencia de PH2." if res["ref"] is not None else ""))
        elif res["ref"] is not None:
            notas.append("Contorno blanco punteado: máscara de referencia, no una segmentación del programa.")
        self.lbl_resultado.configure(text="\n".join(notas))

    def _dibujar_criterios(self):
        """Dibuja los ejes de simetría, el borde por sector y el mapa de colores (M4).

        Entra: nada; lee self.res
        Sale : nada
        """
        grafico = self.g_criterios
        res = self.res
        if res is None or all(res[k] is None for k in ("ejes", "gradientes", "colores")):
            grafico.mensaje("Procesá una imagen para ver los criterios." if res is None
                            else "No se pudieron calcular los criterios: el registro dice qué falló.")
            return
        grafico.limpiar()
        ejes = grafico.figura.subplots(1, 3)
        if res["ejes"] is not None:
            self._dibujar_ejes(ejes[0], *res["ejes"])
        else:
            ejes[0].set_axis_off()
        if res["gradientes"] is not None:
            self._dibujar_sectores(ejes[1], np.asarray(res["gradientes"]))
        else:
            ejes[1].set_axis_off()
        if res["colores"] is not None:
            self._dibujar_colores(ejes[2], res["colores"])
        else:
            ejes[2].set_axis_off()
        grafico.lienzo.draw_idle()

    def _dibujar_ejes(self, ax, f_mayor, f_menor, angulo):
        """Dibuja los dos ejes de simetría de M4 sobre la lesión.

        Supone que `angulo` es el del eje mayor, en grados, medido desde el eje de las columnas.

        Entra: ax                 eje de matplotlib
               f_mayor, f_menor   fracción no superpuesta en cada eje de simetría
               angulo             float, grados
        Sale : nada
        """
        m = self.res["mascara"]
        filas, columnas = np.nonzero(m)
        centro_f, centro_c = filas.mean(), columnas.mean()
        largo = max(np.ptp(filas), np.ptp(columnas)) / 2
        graficos.mostrar(ax, self.res["mejorada"])
        graficos.contorno(ax, m)
        for grados, estilo in ((angulo, "-"), (angulo + 90, "--")):
            dc, df = largo * np.cos(np.deg2rad(grados)), largo * np.sin(np.deg2rad(grados))
            ax.plot([centro_c - dc, centro_c + dc], [centro_f - df, centro_f + df], estilo, color="cyan", linewidth=1.2)
        ax.set_xlim(0, m.shape[1] - 1)
        ax.set_ylim(m.shape[0] - 1, 0)
        a = criterios_abc.contar_ejes_asimetricos(f_mayor, f_menor)
        self._titulo_eje(ax, f"Ejes de simetría: A = {a}\n{numero(f_mayor)} y {numero(f_menor)} sin superponer "
                             f"(asimétrico si > {numero(criterios_abc.UMBRAL_ASIMETRIA)})")

    def _dibujar_colores(self, ax, mapa):
        """Dibuja el mapa de colores de M4 recortado a la lesión, con una leyenda de los colores que aparecen.

        Entra: ax     eje de matplotlib
               mapa   array int de criterios_abc.mapa_de_colores (índice de COLORES, PARECIDO_A_PIEL o -1 afuera)
        Sale : nada
        """
        pintado = np.ones((*mapa.shape, 4))
        pintado[..., 3] = 0                                   # fuera de la lesión: transparente
        pintado[mapa == criterios_abc.PARECIDO_A_PIEL] = to_rgba(PINTURA_PIEL)
        presentes = []
        for k, nombre in enumerate(carga_de_datos.COLORES):
            if np.any(mapa == k):
                pintado[mapa == k] = to_rgba(PINTURA[nombre])
                presentes.append((NOMBRES_COLOR[nombre], PINTURA[nombre]))
        if np.any(mapa == criterios_abc.PARECIDO_A_PIEL):
            presentes.append(("parecido a la piel", PINTURA_PIEL))
        ax.imshow(pintado)
        ax.set_axis_off()
        filas, columnas = np.nonzero(mapa != -1)
        if filas.size:
            ax.set_xlim(columnas.min() - 5, columnas.max() + 5)
            ax.set_ylim(filas.max() + 5, filas.min() - 5)
        ax.legend(handles=[Patch(facecolor=color, edgecolor=PALETA["borde"], label=nombre)
                           for nombre, color in presentes],
                  loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=8, frameon=False, labelcolor=PALETA["texto2"])
        self._titulo_eje(ax, f"Mapa de colores: C = {len(self.res['crit']['colores']) if self.res['crit'] else '—'}\n"
                             f"cuenta si ocupa más del {numero(100 * criterios_abc.FRACCION_MIN_COLOR, 0)} %")

    def _dibujar_sectores(self, ax, gradientes):
        """Dibuja el gradiente medio de cada sector del borde, con el umbral de "abrupto" de M4.

        Entra: ax           eje de matplotlib
               gradientes   array float con un valor por sector
        Sale : nada
        """
        p = PALETA
        sectores = np.arange(1, len(gradientes) + 1)
        ax.set_facecolor(p["superficie"])
        ax.bar(sectores, gradientes, color=p["series"][0])
        ax.axhline(criterios_abc.UMBRAL_BORDE, color=p["series"][1])
        for s in sectores[~np.isfinite(gradientes)]:   # sectores pegados al marco o a la viñeta: no cuentan
            ax.text(s, 0, "toca el\nmarco", ha="center", va="bottom", fontsize=7, color=p["tenue"])
        ax.set_xticks(sectores)
        ax.set_xlabel("sector del borde (desde el eje mayor)", color=p["texto2"])
        ax.set_ylabel("gradiente medio", color=p["texto2"])
        ax.tick_params(colors=p["texto2"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(p["borde"])
        b = criterios_abc.contar_sectores_abruptos(gradientes)
        self._titulo_eje(ax, f"Borde por sector: B = {b}\nabrupto si > {numero(criterios_abc.UMBRAL_BORDE, 4)}")

    def _dibujar_zonas(self):
        """Dibuja las cuatro zonas y, al lado, el mapa de textura recortado a la lesión.

        Entra: nada; lee self.res y el atributo elegido
        Sale : nada
        """
        grafico = self.g_zonas
        res = self.res
        if res is None or res["z"] is None:
            grafico.mensaje("Procesá una imagen para ver las zonas y los mapas de textura.")
            return
        grafico.limpiar()
        izquierda, derecha = grafico.figura.subfigures(1, 2, width_ratios=[1, 1])
        izquierda.set_facecolor(PALETA["superficie"])
        derecha.set_facecolor(PALETA["superficie"])
        for ax, nombre in zip(izquierda.subplots(2, 2).ravel(), zonas_y_mapas.ZONAS):
            graficos.superponer(ax, res["mejorada"], res["z"][nombre])
            self._titulo_eje(ax, nombre.replace("_", " ").capitalize())
        atributo = self._atributo_mapa()
        ax = derecha.subplots()
        ax.set_axis_off()
        if atributo in res["mapas"]:
            filas, columnas = np.nonzero(res["mascara"])
            imagen = ax.imshow(res["mapas"][atributo], cmap="magma")
            ax.set_xlim(columnas.min() - 5, columnas.max() + 5)
            ax.set_ylim(filas.max() + 5, filas.min() - 5)
            barra = derecha.colorbar(imagen, ax=ax, fraction=0.04)
            barra.ax.tick_params(colors=PALETA["texto2"])
            barra.outline.set_visible(False)
            self._titulo_eje(ax, f"Mapa de {NOMBRES_GLCM[atributo].lower()}")
        else:
            ax.text(0.5, 0.5, "Calculando el mapa…", ha="center", va="center", color=PALETA["texto2"], transform=ax.transAxes)
            self._pedir_mapa()
        grafico.lienzo.draw_idle()

    def _llenar_numeros(self):
        """Llena las tablas de atributos GLCM y de contrastes entre zonas.

        Entra: nada; lee self.res
        Sale : nada
        """
        for tabla in (self.t_glcm, self.t_contrastes):
            tabla.delete(*tabla.get_children())
        if self.res is None:
            return
        if self.res["glcm"] is not None:
            for k, nombre in NOMBRES_GLCM.items():
                self.t_glcm.insert("", "end", values=(nombre, numero(self.res["glcm"][k], 3)))
        if self.res["contrastes"] is not None:
            for k, nombre in NOMBRES_CONTRASTE.items():
                self.t_contrastes.insert("", "end", values=(nombre, numero(self.res["contrastes"][f"periferia_vs_centro_{k}"], 3),
                                                            numero(self.res["contrastes"][f"interfase_{k}"], 3)))


def main():
    """Abre la ventana.

    Entra: nada
    Sale : nada; termina cuando se cierra la ventana
    """
    Aplicacion().mainloop()


if __name__ == "__main__":
    main()
