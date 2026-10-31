# Datos

La base **PH2** (Mendonça et al., 2013) **no se sube al repositorio**: pesa bastante y su
descarga requiere aceptar las condiciones de uso de los autores. Cada integrante la
descarga una vez desde la página oficial de PH2 (Universidade do Porto / Hospital Pedro Hispano)
y la descomprime acá, de modo que quede:

```
datos/
├── PH2Dataset/                  <- NO se sube (está en .gitignore)
│   ├── PH2 Dataset images/
│   │   └── IMD002/ ...          una carpeta por imagen: imagen, máscara de la lesión, máscaras de color
│   └── PH2_dataset.txt          anotaciones (diagnóstico, asimetría, colores, estructuras)
├── particion.csv                <- SÍ se sube: qué imagen es de entrenamiento y cuál de prueba
└── LEEME.md
```

Los nombres exactos de carpetas y archivos hay que **verificarlos con la descarga** y, si
cambian, corregir las rutas al principio de `modulos/m0_carga_de_datos/carga_de_datos.py`.

`particion.csv` se genera **una sola vez** desde el cuaderno de M0 (`armar_particion`) y se
sube, así todos trabajan con la misma partición y los resultados son reproducibles.

**Regla de oro:** los umbrales y parámetros se eligen mirando **sólo entrenamiento**. Las
imágenes de prueba se usan al final, para las métricas del informe.
