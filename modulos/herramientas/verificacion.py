"""Comprueba que cada módulo devuelve lo que los demás módulos esperan.

Lo usa main.py al arrancar. Resultado por módulo:
    LISTO   cumple con lo acordado
    FALTA   todavía tiene funciones sin programar (o falta la base PH2)
    ERROR   está programado pero no cumple; se muestra qué falló

Las pruebas usan lesiones inventadas (lesion_de_prueba.py), así que cada
módulo se puede verificar sin la base y sin los módulos anteriores.
Son pruebas de "sentido común" (formas, tipos, casos obvios): no reemplazan
la evaluación con PH2.
"""

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from modulos.herramientas.lesion_de_prueba import generar_lesion, mascara_forma
from modulos.m0_carga_de_datos import carga_de_datos
from modulos.m0_carga_de_datos.carga_de_datos import COLORES, FaltanDatos
from modulos.m1_mejoramiento import mejoramiento
from modulos.m2_compresion import compresion
from modulos.m3_segmentacion import segmentacion
from modulos.m4_criterios_abc import criterios_abc
from modulos.m5_textura_criterio_d import textura_criterio_d
from modulos.m6_zonas_y_mapas import zonas_y_mapas
from modulos.m7_clasificacion_resultados import clasificacion_resultados as m7


def _es_imagen(x, forma, nombre):
    assert isinstance(x, np.ndarray), f"{nombre} tiene que ser un array de numpy"
    assert x.shape == forma, f"{nombre} tiene forma {x.shape} y tendría que ser {forma}"
    assert np.issubdtype(x.dtype, np.floating), f"{nombre} tiene que ser float"
    assert np.all(np.isfinite(x)), f"{nombre} tiene NaN o infinitos"
    assert x.min() >= -1e-6 and x.max() <= 1 + 1e-6, f"{nombre} tiene que estar en [0, 1]"


def _es_mascara(x, forma, nombre):
    assert isinstance(x, np.ndarray) and x.dtype == bool, f"{nombre} tiene que ser un array bool"
    assert x.shape == forma, f"{nombre} tiene forma {x.shape} y tendría que ser {forma}"


# ---------------------------------------------------------------- M0
def verificar_m0():
    if not carga_de_datos.hay_datos():
        raise FaltanDatos("todavía no está la base en datos/PH2Dataset/ (ver datos/LEEME.md)")
    tabla = carga_de_datos.leer_tabla_ph2()
    faltan = {"nombre", "diagnostico", "asimetria", "colores", *carga_de_datos.ESTRUCTURAS} - set(tabla.columns)
    assert not faltan, f"a la tabla de PH2 le faltan columnas: {faltan}"
    assert set(tabla["diagnostico"]) <= {0, 1, 2}, "diagnostico tiene que ser 0, 1 o 2"
    sub = carga_de_datos.seleccionar_subconjunto(tabla)
    assert set(sub["nombre"]) <= set(tabla["nombre"]), "el subconjunto tiene imágenes que no están en PH2"
    part = carga_de_datos.leer_particion()
    assert set(part["conjunto"]) == {"entrenamiento", "prueba"}, "la partición tiene que tener entrenamiento y prueba"
    assert not part["nombre"].duplicated().any(), "hay imágenes repetidas en la partición"
    for nombre in part["nombre"].iloc[:2]:
        caso = carga_de_datos.cargar_caso(nombre, tabla)
        assert caso.imagen.ndim == 3 and caso.imagen.shape[2] == 3, f"{nombre}: la imagen tiene que ser RGB"
        _es_imagen(caso.imagen, caso.imagen.shape, f"{nombre}: imagen")
        _es_mascara(caso.mascara_ref, caso.imagen.shape[:2], f"{nombre}: mascara_ref")
        assert caso.mascara_ref.any(), f"{nombre}: la máscara de referencia está vacía"
        assert caso.asimetria in (0, 1, 2), f"{nombre}: asimetria tiene que ser 0, 1 o 2"
        assert set(caso.colores) <= set(COLORES), f"{nombre}: hay colores fuera de COLORES"
        assert 0 <= caso.d_dermatologo <= 5, f"{nombre}: D del dermatólogo fuera de 0-5"


# ---------------------------------------------------------------- M1
def verificar_m1():
    img, info = generar_lesion(semilla=1, pelos=25)
    forma2 = img.shape[:2]
    mejorada, valida, pelos = mejoramiento.mejorar(img)
    _es_imagen(mejorada, img.shape, "img_mejorada")
    _es_mascara(valida, forma2, "valida")
    _es_mascara(pelos, forma2, "mascara_pelos")
    acuerdo = np.mean(valida == info["mascara_vineta"])
    assert acuerdo >= 0.97, f"la máscara de viñeta coincide en el {100 * acuerdo:.0f} % de los píxeles (mínimo 97 %)"

    igual, vacia = mejoramiento.quitar_pelos(img, "ninguno")
    assert np.allclose(igual, img) and not vacia.any(), "'ninguno' tiene que devolver la imagen sin cambios"
    med, _ = mejoramiento.quitar_pelos(img, "mediana")
    _es_imagen(med, img.shape, "quitar_pelos(mediana)")

    limpia, _ = mejoramiento.quitar_pelos(img, "cierre")
    _es_imagen(limpia, img.shape, "quitar_pelos(cierre)")
    en_pelo = info["mascara_pelos"]
    error_antes = np.abs(img - info["sin_pelos"])[en_pelo].mean()
    error_despues = np.abs(limpia - info["sin_pelos"])[en_pelo].mean()
    assert error_despues < 0.5 * error_antes, \
        f"en los pelos el error baja sólo de {error_antes:.3f} a {error_despues:.3f} (tiene que bajar a menos de la mitad)"
    lejos = info["mascara_vineta"] & ~ndi.binary_dilation(en_pelo, iterations=3)
    cambio = np.abs(limpia - img)[lejos].mean()
    assert cambio < 0.03, f"fuera de los pelos la imagen cambia demasiado (diferencia media {cambio:.3f})"


# ---------------------------------------------------------------- M2
def verificar_m2():
    img, _ = generar_lesion(semilla=2)
    alta, bytes_alta = compresion.comprimir_jpeg(img, 90)
    baja, bytes_baja = compresion.comprimir_jpeg(img, 10)
    _es_imagen(alta, img.shape, "comprimir_jpeg(90)")
    assert int(bytes_alta) > 0 and bytes_baja < bytes_alta, "a menor calidad tiene que ocupar menos bytes"
    assert compresion.tasa_de_compresion(img, bytes_baja) > compresion.tasa_de_compresion(img, bytes_alta) > 1, \
        "la tasa de compresión tiene que ser > 1 y crecer al bajar la calidad"
    assert np.isinf(compresion.psnr(img, img)), "psnr de dos imágenes idénticas tiene que ser inf"
    assert compresion.psnr(img, alta) > compresion.psnr(img, baja), "el PSNR tiene que bajar al bajar la calidad"
    assert compresion.ssim(img, img) > 0.999, "ssim de dos imágenes idénticas tiene que ser 1"
    assert compresion.ssim(img, alta) > compresion.ssim(img, baja), "el SSIM tiene que bajar al bajar la calidad"


# ---------------------------------------------------------------- M3
def verificar_m3():
    a = mascara_forma(ejes=(40, 40))
    assert abs(segmentacion.dice(a, a) - 1) < 1e-9, "dice(A, A) tiene que dar 1"
    assert segmentacion.dice(a, np.roll(a, 150, axis=1)) < 1e-9, "dice de máscaras separadas tiene que dar 0"
    for semilla in (3, 4, 5):
        img, info = generar_lesion(semilla=semilla, irregularidad=0.2)
        m = segmentacion.segmentar(img, info["mascara_vineta"])
        _es_mascara(m, img.shape[:2], "segmentar()")
        _, regiones = ndi.label(m)
        assert regiones == 1, f"la máscara tiene {regiones} regiones (tiene que tener una)"
        assert np.array_equal(m, ndi.binary_fill_holes(m)), "la máscara tiene huecos"
        d = segmentacion.dice(m, info["mascara"])
        assert d >= 0.85, f"en una lesión inventada fácil el Dice es {d:.2f} (mínimo 0,85)"


# ---------------------------------------------------------------- M4
def verificar_m4():
    circulo = mascara_forma(ejes=(60, 60), irregularidad=0)
    f1, f2, _ = criterios_abc.fracciones_asimetria(circulo)
    assert f1 < 0.05 and f2 < 0.05, f"un círculo da fracciones no superpuestas de {f1:.2f} y {f2:.2f}"
    assert criterios_abc.asimetria(circulo) == 0, "un círculo tiene que dar A = 0"
    asim = mascara_forma(ejes=(60, 45), irregularidad=0, lobulo=0.8)
    assert criterios_abc.asimetria(asim) >= 1, "una elipse con un lóbulo de un solo lado tiene que dar A >= 1"
    circ = criterios_abc.circularidad(circulo)
    assert 0.85 <= circ <= 1.1, f"la circularidad de un círculo da {circ:.2f}"
    irregular = mascara_forma(ejes=(60, 60), irregularidad=0.4)
    assert criterios_abc.circularidad(irregular) < circ, "una forma irregular tiene que ser menos circular"

    nitida, info = generar_lesion(semilla=6, desenfoque_borde=0)
    borrosa, _ = generar_lesion(semilla=6, desenfoque_borde=6)
    m = info["mascara"]
    g_n = np.asarray(criterios_abc.gradiente_por_sector(nitida, m))
    g_b = np.asarray(criterios_abc.gradiente_por_sector(borrosa, m))
    assert g_n.shape == (criterios_abc.SECTORES,), "gradiente_por_sector tiene que devolver un valor por sector"
    assert g_n.mean() > g_b.mean(), "un borde nítido tiene que dar más gradiente que uno borroso"
    b_n, b_b = criterios_abc.borde(nitida, m), criterios_abc.borde(borrosa, m)
    assert int(b_n) == b_n and 0 <= b_n <= 8, "B tiene que ser un entero de 0 a 8"
    assert b_n >= b_b, "un borde nítido no puede dar menos B que uno borroso"

    cols = criterios_abc.colores(nitida, m)
    assert isinstance(cols, list) and 1 <= len(cols) <= 6, "colores() tiene que devolver una lista de 1 a 6 nombres"
    assert set(cols) <= set(COLORES), f"colores() devolvió nombres fuera de COLORES: {set(cols) - set(COLORES)}"
    mapa = np.asarray(criterios_abc.mapa_de_colores(nitida, m))
    assert mapa.shape == m.shape and np.all(mapa[~m] == -1), "mapa_de_colores tiene que valer -1 fuera de la lesión"
    assert {"A", "B", "C", "colores", "circularidad"} <= set(criterios_abc.criterios(nitida, m))


# ---------------------------------------------------------------- M5
def verificar_m5():
    m = mascara_forma(ejes=(50, 50), irregularidad=0)
    plana = np.full(m.shape, 0.5)
    p = np.asarray(textura_criterio_d.glcm(plana, m))
    n = textura_criterio_d.NIVELES
    assert p.shape == (n, n), f"glcm() tiene que ser {n} x {n}"
    assert abs(p.sum() - 1) < 1e-6, "la GLCM tiene que estar normalizada (suma 1)"
    at = textura_criterio_d.atributos_glcm(plana, m)
    assert set(textura_criterio_d.ATRIBUTOS) <= set(at), "faltan atributos en atributos_glcm()"
    assert abs(at["contraste"]) < 1e-9 and abs(at["energia"] - 1) < 1e-6, \
        "una zona uniforme tiene que dar contraste 0 y energía 1"

    lisa, info = generar_lesion(semilla=7, textura=0)
    rugosa, _ = generar_lesion(semilla=7, textura=0.6)
    v = lambda x: mejoramiento.a_hsv(x)[..., 2]
    a_l = textura_criterio_d.atributos_glcm(v(lisa), info["mascara"])
    a_r = textura_criterio_d.atributos_glcm(v(rugosa), info["mascara"])
    assert all(np.isfinite(a_r[k]) for k in textura_criterio_d.ATRIBUTOS), "hay atributos NaN en una lesión con textura"
    assert a_r["contraste"] > a_l["contraste"], "una lesión con textura tiene que dar más contraste que una lisa"
    assert a_r["entropia"] > a_l["entropia"], "una lesión con textura tiene que dar más entropía que una lisa"

    rng = np.random.default_rng(0)
    d = rng.integers(0, 4, 200)
    tabla = pd.DataFrame({"glcm_contraste": d + 0.2 * rng.standard_normal(200),
                          "glcm_energia": rng.standard_normal(200), "D_derm": d})
    assert textura_criterio_d.elegir_atributo_d(tabla) == "glcm_contraste", \
        "elegir_atributo_d no elige el atributo que claramente cambia con D (devolver el nombre de la columna)"
    umbrales = np.asarray(textura_criterio_d.ajustar_umbrales_d(tabla["glcm_contraste"].to_numpy(), d))
    assert np.all(np.diff(umbrales) > 0), "los umbrales tienen que ser crecientes"
    est = np.array([textura_criterio_d.estimar_d(x, umbrales) for x in tabla["glcm_contraste"]])
    assert np.mean(est == d) >= 0.8, f"con un atributo casi perfecto, D se acierta sólo en el {100 * np.mean(est == d):.0f} %"
    assert textura_criterio_d.estimar_d(-100, umbrales) == 0, "por debajo de todos los umbrales D tiene que ser 0"


# ---------------------------------------------------------------- M6
def verificar_m6():
    img, info = generar_lesion(semilla=8)
    m = info["mascara"]
    z = zonas_y_mapas.zonas(m)
    assert set(zonas_y_mapas.ZONAS) <= set(z), f"faltan zonas: {set(zonas_y_mapas.ZONAS) - set(z)}"
    for k in zonas_y_mapas.ZONAS:
        _es_mascara(z[k], m.shape, f"zona {k}")
        assert z[k].any(), f"la zona {k} quedó vacía"
    assert not (z["centro"] & z["periferia"]).any(), "centro y periferia se superponen"
    assert np.array_equal(z["centro"] | z["periferia"], m), "centro + periferia tiene que ser toda la lesión"
    assert not (z["interfase_interna"] & ~m).any(), "la interfase interna tiene que estar dentro de la lesión"
    assert not (z["interfase_externa"] & m).any(), "la interfase externa tiene que estar fuera de la lesión"

    por_zona = zonas_y_mapas.atributos_por_zona(img, m)
    assert set(zonas_y_mapas.ZONAS) <= set(por_zona), "atributos_por_zona tiene que tener una entrada por zona"
    contrastes = zonas_y_mapas.contrastes_entre_zonas(por_zona)
    assert isinstance(contrastes, dict) and contrastes, "contrastes_entre_zonas tiene que devolver un dict no vacío"
    assert all(np.isscalar(v) for v in contrastes.values()), "los contrastes tienen que ser números (van a la tabla)"

    v = mejoramiento.a_hsv(img)[..., 2]
    mapa = np.asarray(zonas_y_mapas.mapa_caracteristica(v, m, "contraste"))
    assert mapa.shape == m.shape, "el mapa tiene que tener la forma de la imagen"
    assert np.all(np.isnan(mapa[~m])), "el mapa tiene que ser NaN fuera de la lesión"
    assert np.isfinite(mapa[m]).mean() > 0.5, "el mapa tiene que tener valores en la mayor parte de la lesión"


# ---------------------------------------------------------------- M7
def verificar_m7():
    assert abs(m7.tds(1, 2, 3, 2) - 4.0) < 1e-9, "tds(1, 2, 3, 2) tiene que dar 4,0"
    assert np.allclose(m7.tds(np.array([0, 2]), np.array([0, 8]), np.array([1, 6]), np.array([0, 5])), [0.5, 8.9]), \
        "tds() tiene que funcionar con arrays"
    puntajes = np.array([1.0, 2.0, 3.0, 4.0])
    etiquetas = np.array([False, False, True, True])
    u = m7.umbral_youden(puntajes, etiquetas)
    assert np.array_equal(puntajes > u, etiquetas), "con el umbral de Youden (puntaje > umbral) tiene que separar perfecto"
    met = m7.metricas(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool))
    assert abs(met["sensibilidad"] - 2 / 3) < 1e-9 and abs(met["especificidad"] - 1 / 2) < 1e-9, \
        "metricas(): revisar sensibilidad o especificidad"
    assert abs(m7.auc(puntajes, etiquetas) - 1) < 1e-9 and abs(m7.auc(-puntajes, etiquetas)) < 1e-9, \
        "auc() tiene que dar 1 si separa perfecto y 0 si separa al revés"
    assert abs(m7.kappa_cohen([0, 0, 1, 1], [0, 0, 1, 1]) - 1) < 1e-9, "kappa de dos evaluaciones iguales tiene que ser 1"
    assert abs(m7.kappa_cohen([0, 0, 1, 1], [0, 1, 1, 1]) - 0.5) < 1e-9, "kappa_cohen([0,0,1,1],[0,1,1,1]) tiene que dar 0,5"
    tabla = pd.DataFrame({"A": [0, 1], "B": [0, 0], "C": [1, 1], "D": [0, 0], "x": [1.0, 3.0]})
    ext = np.asarray(m7.tds_extendido(tabla, ["x"], tabla))
    base = np.asarray(m7.tds(tabla["A"], tabla["B"], tabla["C"], tabla["D"]), float)
    x = tabla["x"].to_numpy()
    assert any(np.allclose(ext, base + (x - x.mean()) / x.std(ddof=k)) for k in (0, 1)), \
        "tds_extendido = TDS + (x - media_ref) / desvío_ref"


MODULOS = [
    ("M0 Carga de datos", verificar_m0),
    ("M1 Mejoramiento", verificar_m1),
    ("M2 Compresión", verificar_m2),
    ("M3 Segmentación", verificar_m3),
    ("M4 Criterios A, B, C", verificar_m4),
    ("M5 Textura y criterio D", verificar_m5),
    ("M6 Zonas y mapas", verificar_m6),
    ("M7 Clasificación", verificar_m7),
]


def revisar_modulos():
    """Imprime el estado de cada módulo y devuelve un dict {nombre: True si está LISTO}."""
    listos = {}
    for nombre, verificar in MODULOS:
        try:
            verificar()
            estado = "LISTO"
        except (NotImplementedError, FaltanDatos) as e:
            estado = f"FALTA   {e}"
        except AssertionError as e:
            estado = f"ERROR   {e}"
        except Exception as e:
            estado = f"ERROR   {type(e).__name__}: {e}"
        listos[nombre] = estado == "LISTO"
        print(f"  {nombre:26s}{estado}", flush=True)
    return listos
