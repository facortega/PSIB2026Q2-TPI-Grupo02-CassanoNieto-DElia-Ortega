# Detección de lunares sospechosos mediante imágenes dermatoscópicas

**TPI — 16.63 Procesamiento de Señales e Imágenes Biomédicas · ITBA · 2026Q2 · Grupo 2**
Cassano Nieto, Facundo Tomás · D'Elía, Lucas Valentín · Ortega, Facundo

Algoritmo que segmenta la lesión en una imagen dermatoscópica (base PH2), mide los criterios
de la regla ABCD, los complementa con textura y con descriptores por zona (centro, periferia e
interfase con la piel) y calcula el TDS para estimar si la lesión es sospechosa. Incluye una
interfaz gráfica que muestra cada etapa.

> **Entrega: jueves 05/11/2026 antes de las 17:00** (mail a los 4 docentes).
> Defensa: 12 y 14/11, 19-22 h (15-20 min + 5 de preguntas).

---

## Instalación y ejecución

Requiere Python 3.10 o posterior.

```bash
git clone <link del repositorio>
cd PSIB2026Q2-TPI-Grupo02-CassanoNieto-DElia-Ortega
python -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Descargar la base **PH2** y dejarla en `datos/PH2Dataset/` (ver [`datos/LEEME.md`](datos/LEEME.md)).
   No se sube al repositorio.
2. Correr todo de punta a punta:

```bash
python main.py               # revisa los módulos, procesa las imágenes y genera figuras y tablas
python main.py --revisar     # sólo muestra el estado de cada módulo
python main.py --rehacer     # vuelve a procesar las imágenes (si cambió algún módulo)
```

3. Interfaz gráfica:

```bash
streamlit run interfaz/app.py
```

Funciona aunque falte PH2 o algún módulo: usa una lesión inventada y avisa qué etapas faltan.

## Qué hay en el repositorio

```
PSIB2026Q2-TPI-Grupo02-CassanoNieto-DElia-Ortega/
├── main.py                       corre todo el TP de punta a punta
├── README.md                     este archivo
├── requirements.txt              librerías de Python
├── .gitignore                    archivos que Git no sube (no hace falta tocarlo)
│
├── modulos/                      una carpeta por módulo: su código (.py) + su cuaderno (.ipynb)
│   ├── m0_carga_de_datos/            PH2, alcance acotado, partición entrenamiento/prueba
│   ├── m1_mejoramiento/              viñeta, pelos (cierre vs. mediana), reconstrucción
│   ├── m2_compresion/                JPEG, tasa de compresión, PSNR, SSIM
│   ├── m3_segmentacion/              Otsu, crecimiento de regiones, k-means, Chan-Vese, Dice
│   ├── m4_criterios_abc/             asimetría, borde, colores, circularidad
│   ├── m5_textura_criterio_d/        GLCM y criterio D adaptado
│   ├── m6_zonas_y_mapas/             centro / periferia / interfases y mapas de características
│   ├── m7_clasificacion_resultados/  TDS, TDS extendido, umbrales, métricas, figuras finales
│   └── herramientas/                 ya está hecho, no hace falta tocarlo
│       ├── procesamiento_completo.py     corre M1 → M6 en orden y arma la tabla de características
│       ├── verificacion.py               revisa si cada módulo cumple lo acordado
│       ├── lesion_de_prueba.py           lesiones inventadas para probar sin PH2
│       └── graficos.py                   estilo común de figuras y tablas
│
├── interfaz/app.py               interfaz gráfica (Streamlit)
├── datos/                        PH2 (no se sube) y la partición (sí se sube)
├── resultados/                   tabla de características que genera main.py
├── informe/                      el informe en LaTeX (Overleaf)
│   ├── informe.tex                   archivo principal
│   ├── secciones/                    una sección por archivo
│   ├── figuras/  tablas/             las generan los cuadernos y M7
│   └── referencias.bib
└── presentacion/                 la presentación de la defensa
```

**La idea:** cada integrante trabaja **sólo dentro de la carpeta de su módulo**. El `.py` tiene
las funciones; el cuaderno las prueba, grafica y discute.

### Cómo se relaciona con la consigna, el anteproyecto y la devolución

| Paso obligatorio de la consigna | Dónde |
|---|---|
| Contextualización (imágenes y dataset) | M0 |
| Pre-procesamiento: mejoramiento y reconstrucción | M1 |
| Pre-procesamiento: compresión y descompresión | M2 |
| Procesamiento: segmentación | M3 |
| Extracción de características: criterios y texturas | M4, M5, M6 |
| Resultados: estadística, comparaciones, significancia, visualización | M7 + interfaz |

| Devolución del anteproyecto | Cómo se atiende |
|---|---|
| Mapas de características; interfases piel-lunar; centro y periferia | **M6** (módulo nuevo) |
| Vincularlo con el TDS, más términos en la fórmula | `TERMINOS_EXTRA` y `tds_extendido()` en M7 |
| Contexto frente a las redes neuronales | mapas de M6 + Estado del arte del informe |
| Acotar alcance y cantidad de imágenes | `seleccionar_subconjunto()` en M0 |
| Compresión como objetivo de máxima, no central | M2: núcleo obligatorio (PSNR/SSIM) + máxima opcional |

## Cómo saber qué falta: `python main.py --revisar`

```
1. Estado de los módulos

  M0 Carga de datos         FALTA   todavía no está la base en datos/PH2Dataset/ (ver datos/LEEME.md)
  M1 Mejoramiento           FALTA   M1: falta programar mejorar()
  M2 Compresión             LISTO
  M3 Segmentación           ERROR   en una lesión inventada fácil el Dice es 0.71 (mínimo 0,85)
  ...
```

- `LISTO`: el módulo devuelve lo que los demás esperan.
- `FALTA`: hay funciones sin programar (o falta PH2).
- `ERROR`: está programado pero no cumple; el mensaje dice qué está mal.

Las pruebas usan **lesiones inventadas** (`herramientas/lesion_de_prueba.py`), así que cada módulo
se puede verificar sin PH2 y sin esperar al anterior. Son pruebas de sentido común: que un círculo
dé A = 0, que el cierre recupere los píxeles de pelo, que una lesión con textura tenga más
contraste que una lisa, etc. **No** sirven para elegir umbrales: eso se hace con PH2.

Antes de subir cambios, tu módulo tiene que estar en `LISTO`.

## Lo que cada módulo recibe y devuelve

Esto es lo que permite trabajar por separado. Si alguien necesita cambiarlo, lo avisa al grupo primero.

| Función | Devuelve |
|---|---|
| `cargar_caso(nombre)` | un `Caso`: `imagen` RGB float [0, 1], `mascara_ref` bool, `diagnostico`, `asimetria`, `colores`, `estructuras` |
| `mejorar(img, metodo_pelos)` | `(img_mejorada, valida, mascara_pelos)` — misma forma que la entrada |
| `comprimir_jpeg(img, calidad)` | `(img_descomprimida, cantidad_de_bytes)` |
| `segmentar(img, valida, metodo)` | máscara bool, **una sola región sin huecos** |
| `criterios(img, mascara)` | dict con `A` (0-2), `B` (0-8), `C` (1-6), `colores`, `circularidad` |
| `atributos_glcm(v, mascara)` | dict con los 7 atributos de `ATRIBUTOS` |
| `zonas(mascara)` | dict de máscaras: `centro`, `periferia`, `interfase_interna`, `interfase_externa` |
| `contrastes_entre_zonas(...)` | dict plano de números (van a la tabla y al TDS extendido) |
| `tds(A, B, C, D)` | `1,3 A + 0,1 B + 0,5 C + 0,5 D` (números o arrays) |

Convenciones: imágenes en float [0, 1]; máscaras en bool; "sospechosa" = `puntaje > umbral`.

**Regla de oro:** todo umbral o parámetro se elige con **entrenamiento**. **Prueba** se usa sólo
para las métricas finales del informe (`datos/particion.csv` es la misma para todos).

Desde un cuaderno: `procesar(caso, hasta="segmentacion")` corre una imagen sólo hasta la etapa que ya
esté programada, y `procesar(caso, metodo_pelos="mediana")` o `procesar(caso, calidad_jpeg=25)` sirven
para las hipótesis.

## División del trabajo

| Integrante | Módulos | Informe | Rama(s) |
|---|---|---|---|
| **A** | M0, M1, M2 | 1.2 Marco teórico (color, artefactos, compresión), 2.1, 2.3, 3.1 | `m0-carga`, `m1-mejoramiento`, `m2-compresion` |
| **B** | M3, M4, interfaz | 2.4, 2.5, 2.10, 3.2, 3.3 | `m3-segmentacion`, `m4-criterios`, `interfaz` |
| **C** | M5, M6 | 1.2 (textura, zonas), 1.3 Estado del arte, 2.6, 2.7, 3.4 | `m5-textura`, `m6-zonas` |
| **Todos** | M7 | 2.8, 2.9, 3.5, Conclusiones, presentación | `m7-clasificacion`, `informe` |

Completar con los nombres. M7 conviene que lo empiece quien termine primero y lo revisen los otros
dos. **La consigna evalúa la distribución de commits entre integrantes**: cada uno sube su propio
trabajo desde su cuenta.

### Todos pueden empezar hoy

No hace falta esperar al módulo anterior:

| Módulo | Mientras no está lo anterior, prueba con |
|---|---|
| M1, M2 | `generar_lesion(pelos=25)`: trae la imagen con pelos, la misma sin pelos y dónde están |
| M3 | `generar_lesion()` sin pelos + `info["mascara"]` como verdad |
| M4, M5, M6 | `info["mascara"]` en lugar de la segmentación; `mascara_forma(lobulo=...)` para asimetría |
| M7 | puntajes y etiquetas inventados (ver su cuaderno) |

Lo único urgente es **M0 y la partición**: el resto de las decisiones con PH2 dependen de eso.

## Cómo trabajar con Git

**La consigna lo evalúa**: historial incremental, aportes repartidos entre los tres y mensajes
claros. Un único commit final o aportes desbalanceados **bajan la nota**.

**Una sola persona** sube esta carpeta: crear un repositorio vacío llamado
`PSIB2026Q2-TPI-Grupo02-CassanoNieto-DElia-Ortega` y, desde esta carpeta:

```bash
git init -b main
git add .
git commit -m "Esqueleto del TP: estructura, contratos entre módulos y verificación"
git remote add origin https://github.com/facortega/PSIB2026Q2-TPI-Grupo02-CassanoNieto-DElia-Ortega.git
git push -u origin main
```

Después, sumar a los otros dos como colaboradores. Si el repositorio es **privado**, dar acceso de
lectura a los usuarios de GitHub de los cuatro docentes.

1. **No subir cambios directo a `main`.** Cada uno trabaja en su rama:
   ```bash
   git checkout -b m3-segmentacion             # crear tu rama (una vez)
   git add modulos/m3_segmentacion
   git commit -m "M3: umbral de Otsu sobre el canal V"
   git push -u origin m3-segmentacion
   ```
2. Commits **chicos y frecuentes** (uno por función o por decisión), no uno por día.
3. Cuando el módulo está en `LISTO`, abrir un **Pull Request** hacia `main`. Otro integrante lo
   revisa y lo aprueba (la revisión también queda en el historial).
4. Antes del Pull Request, traer lo último de `main`: `git pull origin main`.
5. Cada uno edita **sólo su carpeta de `modulos/` y sus secciones del informe**.
6. Mensajes cortos que empiecen por el módulo: `M4: umbral de asimetría elegido con entrenamiento`.
7. Si programan de a dos en una compu, agregar al final del mensaje
   `Co-authored-by: Nombre <mail>` para que el aporte cuente para ambos.

### Antes de abrir un Pull Request

- [ ] `python main.py --revisar` muestra tu módulo en `LISTO`
- [ ] Las funciones nuevas están en el `.py`, no sueltas en el cuaderno
- [ ] Los parámetros están al principio del `.py`, con su justificación
- [ ] Las figuras se guardan con `graficos.guardar_figura()`
- [ ] No usaste imágenes de prueba para elegir nada
- [ ] No modificaste archivos fuera de tu carpeta (o lo avisaste)

## Informe

- LaTeX, una sección por archivo en `informe/secciones/` para no pisarse. En Overleaf: subir la
  carpeta `informe/` completa y compilar `informe.tex` (usa `biber`).
- Estructura de la consigna: Introducción (marco teórico, estado del arte, objetivos) · Materiales y
  métodos (incluye la interfaz y qué contiene cada `.py`) · Resultados · Conclusiones.
- **Máximo 8 páginas** de Introducción a Conclusiones (con figuras y tablas, sin carátula ni
  referencias) + anexo opcional de hasta 4.
- Lo que falta aparece **en rojo** con `\pendiente{...}`. Para la entrega, cambiar `\borradortrue`
  por `\borradorfalse`.
- Citas con `\parencite{clave}`; bibliografía APA en `referencias.bib`. **Revisar cada entrada
  contra la fuente original.**

## Entrega (jueves 05/11, antes de las 17:00)

- [ ] `\borradorfalse`, sin `[PENDIENTE]`, ≤ 8 páginas
- [ ] PDF del informe: `PSIB2026Q2-TPI-Grupo02-CassanoNieto DElia Ortega.pdf`
- [ ] Presentación (también en `presentacion/`)
- [ ] `python main.py` corre de cero en una compu limpia siguiendo este README
- [ ] Repositorio con acceso para los docentes (si es privado)
- [ ] Mail a **los 4 docentes** con asunto `[PSIB2026Q2-TPI] Grupo 02 - CassanoNieto DElia Ortega`,
      adjuntando el PDF y la presentación, y con el link al repositorio

Pasadas las 17:00 la nota máxima es 7; si no se entrega antes de la primera clase de defensa, el
TP queda desaprobado (recuperatorio el 28/11).

## Cronograma sugerido

| Fecha | Objetivo |
|---|---|
| sáb 31/10 | Repositorio creado, responsables asignados, PH2 descargada por los tres, **M0 y `particion.csv` en `main`** |
| dom 01/11 | M1, M2 y M3 en `main` · análisis exploratorio · borrador del marco teórico |
| lun 02/11 | M4 y M5 en `main` · borrador de Materiales y métodos |
| mar 03/11 | M6 y M7 en `main` · `python main.py` completo · Resultados |
| mié 04/11 | Conclusiones, revisión cruzada, interfaz terminada, presentación |
| jue 05/11 | Revisión final, `\borradorfalse` y **envío antes de las 17:00** |
| 12 y 14/11 | Defensa (ensayarla cronometrada; la demo de la interfaz entra en los 20 min) |
