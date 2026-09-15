# Clasificación de piezas LEGO por color, forma y orientación

La idea es coger una pieza de LEGO puesta delante de una cámara, decir de cuál de las 12 piezas de nuestro set se trata, y además decir en qué orientación está apoyada (girada en yaw e inclinada en pitch, 8 orientaciones distintas). Todo en tiempo real, sin red neuronal de por medio — solo segmentación clásica, características hechas a mano y un SVM.

## De MATLAB a una app de escritorio

Empezamos prototipando todo en MATLAB porque era lo que teníamos más a mano para iterar rápido con imágenes. En `Matlab/Segmentación` hay tres scripts que probamos con distintos canales de color para separar la pieza del fondo (`Segmentacion_Legos_canal_S.m`, uno "multi canal" y una versión "final"). El canal S (saturación) del HSV terminó siendo el que mejor funcionaba en general, pero nos dimos bastantes cabezazos con casos concretos: el morado no segmentaba bien por canal y probamos correcciones de gamma que solo funcionaban para una imagen concreta, y la pieza amarilla (la 12) daba problemas con Otsu multinivel porque el fondo blanco confundía al umbralizador.

En paralelo fuimos probando clasificadores en `Matlab/Clasificador`, separados en carpetas por lo que estábamos evaluando: **Color**, **Forma**, **Mixto** (color+forma) y **Orientación**. El flujo era extraer características, meterlas en el Classification Learner de MATLAB y entrenar un SVM cuadrático. Con solo características de color (8 características) sacamos un 97.3% de validación y un 100% (335/335) sobre nuestro propio test, pero al probarlo contra una carpeta de imágenes que no habíamos usado para nada cayó a un 82.93%. Con el clasificador mixto (color+forma, reducido de 22 a 12 características) los números de validación eran aún mejores (99.6%, luego 99.81% en test propio) pero también bajaba a 85.37% contra esa carpeta "difícil". El patrón se repetía siempre igual: las piezas 3/6 y 9/12 se confundían entre sí porque comparten forma y solo cambian ligeramente de color, así que tuvimos que añadir un segundo clasificador "refinador" que solo decide entre ese par cuando el clasificador global duda.

Una vez validamos que el enfoque (segmentación + características + SVM en cascada) funcionaba, decidimos pasarlo a C++ con Qt y OpenCV. MATLAB nos servía para prototipar rápido, pero no era una opción realista para procesar vídeo en vivo con la cámara industrial del laboratorio, así que la aplicación final (`ProyectoPSM/`) hace exactamente el mismo pipeline pero integrado en una interfaz de escritorio con captura en tiempo real.

## Pipeline

La app usa una cámara Basler (SDK Pylon, no una webcam ni una cámara IP) y, para cada frame, hace lo siguiente:

```
Cámara Basler (Pylon SDK)
        │
        ▼
Segmentación (balance de blancos → HSV → Otsu sobre canal S
              → morfología → filtro área/aspect-ratio/saturación)
        │  (recorte de cada pieza, fondo a negro)
        ▼
Extracción de 11 características (5 color + 6 forma)
        │
        ▼
Clasificador global — SVM RBF, 12 clases (modelM.yml)
        │
        ├─ ¿predicción ∈ {9,10,11,12}? ─sí─► Refinador 9-12 — SVM RBF (model912.yml)
        │                                          │
        └─ no ─────────────────────────────────────┤
                                                     ▼
                                           Clase final de la pieza
                                                     │
                                                     ▼
                        Clasificador de orientación — template matching
                        (128×128, plantillas por clase × 8 yaw × 4 pitch)
                                                     │
                                                     ▼
                              Resultado: clase + orientación (yaw, pitch)
```

**Segmentación** (`Segmentacion.cpp`): balance de blancos por canal, boost de saturación (×1.55) y corrección de iluminación dividiendo el canal V por una versión muy desenfocada de sí mismo (para aplanar la iluminación desigual de la mesa), Otsu sobre el canal S resultante, y una cadena de operaciones morfológicas (cierre → relleno de huecos → apertura → limpieza de bordes → cierre grande → apertura final) para quedarnos con máscaras limpias. Después se descartan los contornos por área relativa/absoluta, por aspect ratio (para no colar piezas alargadas que no son LEGO) y por saturación media (para descartar sombras y reflejos).

**Características** (`ExtractCaracteristicas.cpp`): 11 en total, sin nada de deep learning. 5 de color — media circular del tono (H), media/mediana de saturación, media/rango intercuartílico del valor — calculadas tras pasar por Lab+CLAHE para corregir el contraste antes de volver a HSV. Y 6 de forma sobre la máscara ya alineada por su ángulo principal: circularidad, extent, solidity, excentricidad, longitud del esqueleto normalizada por el área, y un descriptor de Fourier del contorno (FD5). En algún momento tuvimos también el número de Euler como característica, pero lo quitamos porque no aportaba (ahora mismo son 11, no 12).

**Clasificación** (`Clasificador.cpp` + `TrainingWorker.cpp`): SVM con kernel RBF (`cv::ml::SVM`, `C_SVC`), entrenado con grid search por K-fold sobre C y gamma. Las características se normalizan (z-score) con un scaler guardado junto al modelo. El modelo global (`modelM.yml`) distingue las 12 clases; cuando predice alguna del grupo confuso 9-12, se llama a un segundo SVM (`model912.yml`) entrenado solo con esas 4 clases para intentar corregir el error.

**Orientación** (`ClasificadorOrientacion.cpp` + `TemplateGenerator.cpp`): aquí no usamos SVM, sino *template matching*. Por cada clase y cada combinación de yaw (8 valores, cada 45°) y pitch (4 inclinaciones de cámara) se genera una plantilla de 128×128 en escala de grises, normalizada a media 0 y norma L2 = 1. En inferencia se recorta y normaliza la pieza (probando pequeños desplazamientos ±4 px para compensar el centrado) y se compara por producto escalar contra todas las plantillas de esa clase; se queda con la de mayor similitud y reporta también el margen respecto a la segunda mejor, como medida de confianza.

## Resultados

Los números salen de los `.txt` de evaluación que fue generando la propia app (`Matlab/Clasificador/resultados_*.txt` y `Database/Models/eval_report.txt`), no son estimaciones:

- **Split de test real** (`SEGMENTED_TEST`, held-out, nunca visto en entrenamiento): **98.47%** (385/391) con el modelo global + refinador.
- **Sobre las ~1923 imágenes segmentadas** (dataset completo, no un held-out limpio): **99.74%** de acierto (1918/1923), con el refinador 9-12 activándose 294 veces y corrigiendo 1 caso que el clasificador global tenía mal.
- **Clasificador de orientación (yaw)** sobre las mismas ~1923 imágenes: **99.32%** de acierto (1910/1923).
- El caso más duro que probamos fue una carpeta de test con la peor condición de cámara (pitch más bajo) y muchas piezas de las clases confusas: ahí, con solo 6 características de forma, la precisión caía a **62.5%** (35/56); subiendo a 24 características de forma y activando bien el refinador 9-12 mejoró a **71.43%** (40/56). Es la prueba más honesta de que el sistema generaliza bastante peor de lo que sugieren los números "sobre todo el dataset", y de que más características de forma + el refinador sí ayudan, pero no lo arreglan del todo.

En resumen: la configuración que mejor funciona es 11 características (color+forma) + SVM RBF + cascada de refinador 9-12; la que peor funciona es cualquier variante con pocas características de forma evaluada en condiciones de iluminación/ángulo que no estaban bien representadas en el set de entrenamiento.

## Compilar y ejecutar

El proyecto de escritorio está en `ProyectoPSM/ProyectoPSM.sln` (Visual Studio 2022, toolset v143, plataforma x64 — no hay proyecto qmake/.pro, solo el `.sln`/`.vcxproj`). Para compilarlo hace falta:

- **Visual Studio 2022** con la extensión **Qt VS Tools** y **Qt 6.9.2** (MSVC2022 64-bit) instalado y registrado en la extensión.
- **OpenCV 4.12** (`opencv_world4120.lib` / `opencv_world4120d.lib` en Debug) con sus include/lib paths configurados en el proyecto.
- **Basler Pylon SDK** (la app usa una cámara Basler vía `pylon::CBaslerUniversalInstantCamera`; el instalador de Pylon define la variable de entorno `PYLON_DEV_DIR` que usa el `.vcxproj`).

Pasos:

1. Abrir `ProyectoPSM/ProyectoPSM.sln` en Visual Studio.
2. Elegir configuración `Release|x64` (o `Debug|x64`).
3. Compilar y ejecutar. Si no hay cámara Basler conectada, la app arranca igualmente pero avisa de que la cámara no está lista — se puede seguir usando el modo offline (cargar imagen de disco, segmentar, clasificar) desde la interfaz.

La pestaña de "Entrenamiento" de la propia app permite regenerar todo el pipeline (segmentar → extraer características → generar plantillas → entrenar → evaluar) apuntando a las carpetas de `Database/`.

## Estructura del repo

- `Matlab/` — prototipos: segmentación por canal y los distintos clasificadores que probamos antes de pasar a C++.
- `ProyectoPSM/` — la aplicación final en C++/Qt/OpenCV.
- `Database/` — `RAW` (fotos originales), `SEGMENTED` (piezas ya recortadas), `SEGMENTED_TRAIN`/`SEGMENTED_TEST` (split fijo, ver `split_manifest.csv`: 1537 train / 386 test sobre 1923 imágenes en 12 clases).
- `Proceso Clasificacion.txt` — apuntes internos sobre el flujo de entrenamiento/inferencia que usamos de chuleta mientras desarrollábamos.

## Limitaciones / cosas que mejoraríamos

- La segmentación depende de un único umbral de Otsu sobre el canal S y de unos cuantos umbrales fijos (área, aspect ratio, saturación) que ajustamos a ojo para nuestra mesa y nuestra iluminación. Cambiar el fondo o la luz probablemente la rompe.
- Como muestran los propios resultados, en la condición de cámara más dura la precisión baja bastante (62-71% frente al ~98-99% "normal"). El sistema no generaliza tan bien como parece a primera vista si solo se mira la métrica sobre todo el dataset.
- Las clases 3/6 y 9/12 son casi indistinguibles por forma y color con nuestras 11 características; el refinador ayuda pero a veces introduce fallos nuevos donde antes acertaba (lo vimos literalmente en los `.txt` de resultados de MATLAB).
- El clasificador de orientación necesita una plantilla por cada combinación de clase × yaw × pitch generada de antemano — añadir una pieza nueva implica volver a fotografiarla en todas las orientaciones y regenerar plantillas, no es nada plug-and-play.
- La app solo funciona con la cámara Basler concreta que usamos (vía Pylon) y solo compila en Windows/Visual Studio; no hay build para Linux ni soporte de webcam genérica.
