"""Herramienta para marcar a mano las zonas rojas (PH2 no trae máscaras de rojo).

Uso, desde la carpeta del repo:
    python -m modulos.m4_criterios_abc.marcar_rojo            # las imágenes que faltan
    python -m modulos.m4_criterios_abc.marcar_rojo IMD035     # una en particular (la rehace)

Abre, de a una, las imágenes de ENTRENAMIENTO en las que el dermatólogo anotó rojo
(las de prueba no se tocan, para no usarlas en ningún ajuste).

Cómo marcar:
    - Con clics se ponen los vértices de un polígono alrededor de una zona roja; se cierra
      haciendo clic sobre el primer vértice. Se pueden marcar varias zonas en la misma imagen.
    - Conviene marcar sólo lo que es claramente rojo, aunque quede chico: la zona sirve
      para saber cómo es el rojo típico, no para medir cuánto rojo hay.
    - Para hacer zoom, la lupa de la barra de abajo (hay que desactivarla para seguir marcando).
Teclas:
    enter  guardar y pasar a la siguiente
    b      borrar todo lo marcado en esta imagen
    n      no hay una zona roja clara: guarda una máscara vacía y sigue
    esc    descartar el polígono que se está dibujando
Cerrar la ventana termina; lo guardado queda guardado.

Cada máscara se guarda en datos/anotaciones_rojo/<nombre>_rojo.png (blanco = rojo),
del mismo tamaño que la imagen. Esa carpeta sí se sube al repositorio.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import PolygonSelector
from skimage import io
from skimage.draw import polygon2mask

from modulos.m0_carga_de_datos import carga_de_datos
from modulos.m4_criterios_abc.criterios_abc import CARPETA_ANOTACIONES_ROJO


def imagenes_con_rojo():
    """Nombres de las imágenes de entrenamiento en las que el dermatólogo anotó rojo."""
    tabla = carga_de_datos.leer_tabla_ph2()
    con_rojo = set(tabla.loc[tabla["colores"].str.contains("rojo"), "nombre"])
    return [n for n in carga_de_datos.nombres("entrenamiento") if n in con_rojo]


def ruta_de(nombre):
    return CARPETA_ANOTACIONES_ROJO / f"{nombre}_rojo.png"


class Marcador:
    """Una ventana por imagen: polígonos que se van sumando a la máscara."""

    def __init__(self, caso):
        self.caso = caso
        self.mascara = np.zeros(caso.imagen.shape[:2], bool)
        self.siguiente = False
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        self.ax.imshow(caso.imagen)
        self.ax.contour(caso.mascara_ref, levels=[0.5], colors="white", linewidths=0.8, linestyles="--")
        self.capa = self.ax.imshow(np.zeros((*self.mascara.shape, 4)))
        self.ax.set_title(f"{caso.nombre}: marcá las zonas rojas  ·  enter = guardar  ·  b = borrar  ·  "
                          "n = no hay rojo claro", fontsize=9)
        self.ax.set_axis_off()
        self.fig.canvas.mpl_connect("key_press_event", self.tecla)
        self.nuevo_selector()

    def nuevo_selector(self):
        # Después de cerrar un polígono se arma un selector nuevo para poder marcar otra zona
        self.selector = PolygonSelector(self.ax, self.poligono_cerrado, useblit=True,
                                        props=dict(color="yellow", linewidth=1.2))

    def poligono_cerrado(self, vertices):
        filas_columnas = np.array([(y, x) for x, y in vertices])
        self.mascara |= polygon2mask(self.mascara.shape, filas_columnas)
        self.selector.disconnect_events()
        self.selector.set_visible(False)
        self.dibujar()
        self.nuevo_selector()

    def dibujar(self):
        capa = np.zeros((*self.mascara.shape, 4))
        capa[self.mascara] = (1, 1, 0, 0.35)
        self.capa.set_data(capa)
        self.fig.canvas.draw_idle()

    def tecla(self, evento):
        if evento.key == "enter":
            self.guardar()
        elif evento.key == "n":
            self.mascara[:] = False
            self.guardar()
        elif evento.key == "b":
            self.mascara[:] = False
            self.dibujar()

    def guardar(self):
        CARPETA_ANOTACIONES_ROJO.mkdir(parents=True, exist_ok=True)
        io.imsave(ruta_de(self.caso.nombre), (self.mascara * 255).astype(np.uint8), check_contrast=False)
        print(f"{self.caso.nombre}: guardada ({self.mascara.sum()} píxeles rojos)")
        self.siguiente = True
        plt.close(self.fig)


def main(nombres=None):
    tabla = carga_de_datos.leer_tabla_ph2()
    if nombres:
        pendientes = nombres
    else:
        pendientes = [n for n in imagenes_con_rojo() if not ruta_de(n).exists()]
    if not pendientes:
        print("No falta ninguna: las imágenes con rojo ya están marcadas.")
        return
    print(f"Faltan {len(pendientes)}: {', '.join(pendientes)}")
    for nombre in pendientes:
        marcador = Marcador(carga_de_datos.cargar_caso(nombre, tabla))
        plt.show()
        if not marcador.siguiente:       # cerraron la ventana sin guardar: terminar
            print("Terminado sin guardar", nombre)
            return


if __name__ == "__main__":
    main(sys.argv[1:])
