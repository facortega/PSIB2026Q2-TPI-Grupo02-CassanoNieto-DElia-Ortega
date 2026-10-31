"""Corre todo el trabajo práctico de punta a punta.

Uso, desde la carpeta del repo:

    python main.py               revisa los módulos y, si están listos, procesa y genera resultados
    python main.py --revisar     sólo revisa el estado de los módulos
    python main.py --rehacer     vuelve a procesar las imágenes aunque ya exista la tabla guardada

1. Revisa el estado de cada módulo (LISTO / FALTA / ERROR).
2. Si M0 a M6 están listos, procesa las imágenes y guarda la tabla de
   características en resultados/caracteristicas.csv (tarda: se hace una vez).
3. Estima el criterio D (ajustado con entrenamiento) y, si M7 está listo,
   genera las figuras y tablas finales en informe/figuras/ e informe/tablas/.

No hace falta modificar este archivo para programar los módulos.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from modulos.herramientas.verificacion import revisar_modulos

TABLA = Path(__file__).resolve().parent / "resultados" / "caracteristicas.csv"


def main():
    parser = argparse.ArgumentParser(description="TPI PSIB — detección de lunares sospechosos")
    parser.add_argument("--revisar", action="store_true", help="sólo revisar los módulos")
    parser.add_argument("--rehacer", action="store_true", help="volver a procesar todas las imágenes")
    args = parser.parse_args()

    print("\n1. Estado de los módulos\n")
    listos = revisar_modulos()
    if args.revisar:
        return
    if not all(v for k, v in listos.items() if not k.startswith("M7")):
        print("\nPara procesar las imágenes, los módulos M0 a M6 tienen que estar en LISTO.\n")
        return

    from modulos.herramientas import procesamiento_completo as pc

    print("\n2. Procesamiento de las imágenes\n")
    if TABLA.exists() and not args.rehacer:
        tabla = pd.read_csv(TABLA, dtype={"nombre": str})
        print(f"  Uso la tabla guardada ({len(tabla)} imágenes). Para rehacerla: python main.py --rehacer")
    else:
        t0 = time.time()
        tabla = pc.tabla_de_caracteristicas(pc.procesar_todos())
        TABLA.parent.mkdir(exist_ok=True)
        tabla.to_csv(TABLA, index=False)
        print(f"  {len(tabla)} imágenes en {time.time() - t0:.0f} s -> {TABLA.relative_to(TABLA.parents[1])}")
    for conjunto, grupo in tabla.groupby("conjunto"):
        print(f"  {conjunto:14s} {len(grupo):4d} imágenes   Dice medio {grupo['dice'].mean():.3f}")

    tabla = pc.completar_d(tabla)
    print(f"  Criterio D estimado con {tabla.attrs['atributo_d']}")

    print("\n3. Resultados finales\n")
    try:
        from modulos.m7_clasificacion_resultados import clasificacion_resultados
        clasificacion_resultados.generar(tabla)
        print("  Figuras y tablas guardadas en informe/figuras/ e informe/tablas/\n")
    except NotImplementedError as e:
        print(f"  FALTA   {e}\n")


if __name__ == "__main__":
    main()
