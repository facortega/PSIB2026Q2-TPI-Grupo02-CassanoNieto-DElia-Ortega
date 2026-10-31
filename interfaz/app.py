"""Interfaz gráfica (opcional según la consigna, comprometida en el anteproyecto).

Uso, desde la carpeta del repo:

    streamlit run interfaz/app.py

Muestra cada etapa del algoritmo sobre una imagen de PH2 (o una lesión
inventada, si todavía no está la base): máscara de pelos, contorno, ejes de
simetría, sectores del borde, mapa de colores, zonas, mapa de textura, los
valores de A, B, C y D, el TDS y la clasificación.

Cada etapa que todavía no está programada aparece como "Falta Mx" y la
interfaz sigue con lo que haya. La interfaz NO tiene lógica propia: sólo llama
a las funciones de modulos/ (así lo que se ve es exactamente lo que se evalúa).

PENDIENTE (interfaz):
  - [ ] Ejes de simetría y sectores del borde dibujados sobre la imagen.
  - [ ] D estimado y umbral: leerlos de resultados/ cuando M5 y M7 estén listos.
  - [ ] Subir una imagen propia (st.file_uploader).
  - [ ] Capturas para el informe (sección 2) y ensayo de la demo (cuenta en los 15-20 min).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulos.herramientas import graficos  # noqa: E402
from modulos.herramientas.lesion_de_prueba import generar_lesion  # noqa: E402
from modulos.m0_carga_de_datos import carga_de_datos  # noqa: E402
from modulos.m1_mejoramiento import mejoramiento  # noqa: E402
from modulos.m3_segmentacion import segmentacion  # noqa: E402
from modulos.m4_criterios_abc import criterios_abc  # noqa: E402
from modulos.m6_zonas_y_mapas import zonas_y_mapas  # noqa: E402
from modulos.m7_clasificacion_resultados import clasificacion_resultados as m7  # noqa: E402

st.set_page_config(page_title="TPI PSIB · Lunares sospechosos", layout="wide")
st.title("Detección de lunares sospechosos — regla ABCD")
st.caption("PSIB 2026Q2 · Grupo 2 · Herramienta académica, no es un dispositivo de diagnóstico.")


def etapa(nombre, funcion, *args, **kwargs):
    """Corre una etapa; si falta programarla o falla, lo avisa y devuelve None."""
    try:
        return funcion(*args, **kwargs)
    except NotImplementedError as e:
        st.info(str(e))
    except Exception as e:  # que una etapa rota no tire abajo la interfaz
        st.warning(f"{nombre}: {type(e).__name__}: {e}")
    return None


def figura(img, mascara=None, ref=None, titulo="", **kwargs):
    fig, ax = plt.subplots(figsize=(4, 3))
    graficos.mostrar(ax, img, titulo, **kwargs)
    if ref is not None:
        graficos.contorno(ax, ref, referencia=True)
    if mascara is not None:
        graficos.contorno(ax, mascara)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------- Elegir imagen
with st.sidebar:
    st.header("Imagen")
    caso = None
    if carga_de_datos.hay_datos():
        try:
            nombre = st.selectbox("Imagen de PH2", carga_de_datos.nombres("prueba"))
            caso = carga_de_datos.cargar_caso(nombre)
        except (NotImplementedError, carga_de_datos.FaltanDatos) as e:
            st.info(str(e))
    if caso is None:
        st.write("Sin PH2: se usa una lesión inventada.")
        semilla = st.slider("Semilla", 0, 50, 0)
        pelos = st.slider("Pelos", 0, 60, 20)
        img, info = generar_lesion(semilla=semilla, pelos=pelos, irregularidad=0.25, textura=0.3)
        ref, titulo = info["mascara"], f"Lesión inventada (semilla {semilla})"
    else:
        img, ref = caso.imagen, caso.mascara_ref
        titulo = f"{caso.nombre} — {carga_de_datos.DIAGNOSTICOS[caso.diagnostico]}"
    metodo_pelos = st.radio("Quitar pelos", mejoramiento.METODOS_PELOS, index=2)
    metodo_seg = st.radio("Segmentación", segmentacion.METODOS,
                          index=segmentacion.METODOS.index(segmentacion.METODO_ELEGIDO))

# ------------------------------------------------------------- Etapas
col1, col2, col3 = st.columns(3)
with col1:
    figura(img, titulo=titulo)

salida = etapa("M1", mejoramiento.mejorar, img, metodo_pelos)
mascara = None
if salida is not None:
    mejorada, valida, pelos = salida
    with col2:
        figura(pelos, titulo="Pelos detectados")
    with col3:
        mascara = etapa("M3", segmentacion.segmentar, mejorada, valida, metodo_seg)
        if mascara is not None:
            d = etapa("M3", segmentacion.dice, mascara, ref)
            figura(mejorada, mascara, ref, f"Segmentación (Dice {d:.3f})" if d is not None else "Segmentación")

if mascara is not None:
    st.subheader("Criterios")
    crit = etapa("M4", criterios_abc.criterios, mejorada, mascara)
    c1, c2, c3 = st.columns(3)
    with c1:
        mapa = etapa("M4", criterios_abc.mapa_de_colores, mejorada, mascara)
        if mapa is not None:
            figura(np.ma.masked_less(mapa, 0), titulo="Mapa de colores", cmap="tab10", vmin=0, vmax=9)
    with c2:
        z = etapa("M6", zonas_y_mapas.zonas, mascara)
        if z is not None:
            etiquetas = np.zeros(mascara.shape)
            for i, k in enumerate(zonas_y_mapas.ZONAS, start=1):
                etiquetas[z[k]] = i
            figura(np.ma.masked_equal(etiquetas, 0), titulo="Zonas", cmap="viridis")
    with c3:
        v = mejoramiento.a_hsv(mejorada)[..., 2]
        mt = etapa("M6", zonas_y_mapas.mapa_caracteristica, v, mascara, "contraste")
        if mt is not None:
            figura(mt, titulo="Mapa de contraste (GLCM)", cmap="magma")

    if crit is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("A (asimetría)", crit["A"])
        k2.metric("B (borde)", crit["B"])
        k3.metric("C (colores)", crit["C"])
        k4.metric("Circularidad", f"{crit['circularidad']:.2f}")
        st.write("Colores presentes:", ", ".join(crit["colores"]) or "—")
        D = st.slider("D (mientras M5 no esté integrado acá)", 0, 5, 0)
        tds = etapa("M7", m7.tds, crit["A"], crit["B"], crit["C"], D)
        if tds is not None:
            clase = ("probable melanoma" if tds > m7.UMBRAL_MELANOMA
                     else "sospechosa" if tds > m7.UMBRAL_SOSPECHOSA else "benigna")
            st.metric("TDS", f"{tds:.2f}", clase)
