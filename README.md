# Sistema de visión artificial para reconocimiento de códigos 3D

Proyecto de Procesado de Señales Multimedia (Máster en Ingeniería de Telecomunicación, UPNA) de Iñaki Janices, Juan Lacosta y Natalia Salvador. El objetivo: coger una pieza de LEGO puesta delante de una cámara, decir cuál de las 12 piezas de nuestro set es, y en qué orientación está apoyada (girada en yaw cada 45°, inclinada en pitch a 10°/40°/70°/90° — 12 formas × 8 yaw = 96 estados posibles). Todo en tiempo real, sin red neuronal de por medio — solo segmentación clásica, características hechas a mano y un SVM.

Este trabajo se enmarca dentro de "Smart-Bus IoT", un proyecto integrador de modernización del transporte urbano en la Mancomunidad de Pamplona basado en fusión de sensores (rastreo de dispositivos móviles + visión artificial) para estimar la ocupación de autobuses. El reconocimiento de códigos 3D sirve aquí como prueba de concepto: la misma arquitectura (captura → segmentación → clasificación supervisada → interfaz) es directamente extrapolable al conteo de personas, validando que se puede procesar imágenes y extraer características en tiempo real in-situ antes de mandar nada a un servidor central.

## De MATLAB a una app de escritorio

Empezamos prototipando todo en MATLAB porque era lo que teníamos más a mano para iterar rápido con imágenes. En `Matlab/Segmentación` hay scripts que probamos con distintos canales de color para separar la pieza del fondo. El canal S (saturación) del HSV terminó siendo el que mejor funcionaba en general.

En paralelo fuimos probando clasificadores en `Matlab/Clasificador`, separados en carpetas por lo que estábamos evaluando: **Color**, **Forma**, **Mixto** (color+forma) y **Orientación**. Una vez validamos que el enfoque (segmentación + características + SVM) funcionaba, decidimos pasarlo a C++ con Qt y OpenCV: MATLAB nos servía para prototipar rápido, pero no era una opción realista para procesar vídeo en vivo con la cámara del laboratorio. La aplicación final (`ProyectoPSM/`) hace el mismo pipeline pero integrado en una interfaz de escritorio con captura en tiempo real, sin depender de ningún código externo para segmentar, entrenar o clasificar.

## Pipeline

```
Cámara (VideoAcquisition — Basler vía Pylon SDK)
        │
        ▼
Segmentación (balance de blancos → HSV → boost de saturación ×1.55
              → corrección de sombreado en V → Otsu sobre canal S
              → morfología → filtro área/aspect-ratio/saturación)
        │  (recorte de cada pieza, fondo a negro)
        ▼
Extracción de 12 características (5 color + 7 forma)
        │
        ▼
Clasificador único — SVM, 12 clases
        │
        ▼
Clasificador de orientación — template matching
        (128×128, plantillas por clase × 8 yaw × 4 pitch)
        │
        ▼
Resultado: clase + orientación (yaw, pitch)
```

**Captura** (`VideoAcquisition`): encapsula la comunicación con la cámara (Basler, SDK Pylon) y el control de parámetros como la exposición automática, gestiona desconexiones sin lanzar excepciones y permite reconectar desde la interfaz. Sirve tanto para la operación en vivo como para generar la base de datos de entrenamiento: el protocolo de captura registra cada uno de los 12 códigos en sus 8 orientaciones, variando ángulo cenital e iluminación, con un mínimo de 20 muestras por configuración. La clase auxiliar `NameHelper` codifica clase, orientación y número de secuencia en el nombre de cada archivo para poder extraer la etiqueta automáticamente en el entrenamiento supervisado.

**Segmentación** (`Segmentacion.cpp`): balance de blancos (ajusta R y B tomando el canal verde como referencia) y normalización a coma flotante, paso a HSV, boost lineal de saturación (×1.55) y corrección de sombreado dividiendo el canal V por una versión muy desenfocada de sí mismo. Sobre el canal S resultante se aplica Otsu, y una cadena de operaciones morfológicas (cierre → relleno de huecos → apertura para ruido sal-y-pimienta → eliminación de componentes que tocan el borde → cierre grande para unir partes separadas por reflejos → apertura suave final) deja una máscara limpia. Los componentes conexos se filtran por área relativa, por aspect ratio (para descartar objetos alargados que no son LEGO) y por saturación media (para descartar sombras). El resultado se recorta sobre fondo negro con mejora de contraste local antes de pasar a extracción de características.

Durante el prototipado en MATLAB se probaron y descartaron varias alternativas: Otsu multinivel (demasiado difícil de automatizar la asignación de la clase intermedia), máscaras específicas por rango de matiz para colores de baja saturación como morado/rosa (sustituido por el boost global de saturación, más simple y generalizable), segmentación por luminosidad inversa para piezas negras (confundía piezas oscuras con sus propias sombras) y corrección Gamma frente a ganancia lineal (mismo resultado, más coste computacional).

**Características** (`ExtractCaracteristicas.cpp`): 12 en total (5 de color + 7 de forma), elegidas mediante un ranking combinado de varios métodos de feature selection (chi-cuadrado, mRMR, ReliefF y la importancia de un árbol de decisión) sobre un conjunto más amplio de candidatas (se llegó a probar un vector de forma de 24 características). Color: H_mean_circ (media circular del tono), S_mean y S_median (saturación media/típica), V_mean y V_IQR (brillo medio y su variabilidad), todo calculado en HSV tras corregir el contraste con CLAHE en el canal L de Lab. Forma, sobre la máscara ya limpiada y normalizada de orientación: Circularity, Extent, Solidity, Eccentricity, EulerNumber (agujeros grandes de la pieza) y SkelLenNorm (longitud del esqueleto normalizada), más el descriptor de Fourier FD5 del contorno.

**Clasificación** (`Clasificador.cpp` + `TrainingWorker.cpp`): un único SVM (kernel RBF, `cv::ml::SVM`, `C_SVC`) entrenado con grid search por 5-fold cross-validation sobre C y gamma, con las características normalizadas por z-score (scaler guardado junto al modelo). En MATLAB se había explorado también una cascada de clasificadores: un primer clasificador mixto y, si la pieza caía en el grupo confuso 3/6 (rojas) o 9/12 (amarillas), un segundo clasificador de forma (KNN de 3 vecinos para 3/6, con 95.7% de acierto; SVM cuadrático para 9/12, con 96.7%) que intentaba corregir el error. **Esa cascada se probó en serio pero se descartó al portar el sistema a C++**: las características extraídas en C++ no coincidían exactamente con las de MATLAB, y en la práctica el refinador introducía más errores de los que corregía — el clasificador de color acertaba y el refinador, al aplicarse después, lo estropeaba. Se optó en su lugar por un único clasificador mixto (color + forma) con el vector de 12 características optimizado, que además de simplificar mucho el código en C++ (evita tener que portar varios tipos de clasificador con comportamientos poco predecibles) dio mejor resultado.

**Orientación** (`ClasificadorOrientacion.cpp` + `TemplateGenerator.cpp`): no se usa un clasificador, sino *template matching*. Por cada combinación de clase, yaw (8 valores cada 45°) y pitch se genera una plantilla de 128×128 en escala de grises, centrada a media 0 y normalizada por norma L2; la plantilla final es el promedio de varios parches normalizados de ese grupo, para reducir ruido. En inferencia se recorta y normaliza la pieza igual, probando pequeños desplazamientos (±unos píxeles) para compensar el centrado, y se compara por producto escalar contra las plantillas de la clase ya predicha; se reporta también el margen (*gap*) respecto a la segunda mejor plantilla como medida de confianza. Se prefirió este enfoque a entrenar un segundo clasificador porque no requiere modelo adicional (basta con los promedios por orientación), es mucho más barato computacionalmente y es directo de depurar si falla una orientación concreta.

## Resultados

Cifras extraídas de las matrices de confusión de la memoria del proyecto:

- **Clasificador de color puro** (8 características, Quadratic SVM, validación MATLAB): **97.2%**, con confusión clara entre las clases 3/6 (rojas) y 9/12 (amarillas), que comparten forma y solo difieren ligeramente de color.
- **Refinadores de forma probados en MATLAB** para esos pares confusos: KNN (3 vecinos, 24 características de forma) sobre 3/6 → **95.7%**; SVM cuadrático (24 características) sobre 9/12 → **96.7%**. Buenos resultados aislados, pero la cascada completa no sobrevivió al paso a C++ (ver arriba).
- **Clasificador mixto final** (12 características: 5 color + 7 forma), Quadratic SVM en validación MATLAB: **99.4%**.
- **Mismo clasificador mixto, ya implementado en C++/OpenCV**, evaluado sobre el conjunto de test tras regenerar la base de datos segmentada directamente con la implementación definitiva: **99.76%** de acierto.

En resumen: la configuración que mejor funciona es el vector mixto de 12 características (color+forma) con un único SVM; la cascada de refinadores ayudaba en MATLAB pero acumulaba errores en C++ y se abandonó por eso, no porque la idea fuera mala en sí — simplemente la inconsistencia entre las características extraídas en cada entorno la hacía perder más de lo que ganaba.

### Notas sobre la base de datos

- En el código 3, las imágenes capturadas con un ángulo de elevación (pitch) de 40° presentan una orientación diferente respecto al resto de ángulos, lo que empeora el rendimiento del clasificador de orientación para esa clase.
- En el código 6, la definición de 0° de orientación en la base de datos no coincide con la del documento de referencia del proyecto. Es un desfase sistemático que no afecta a la precisión del clasificador, pero significa que la orientación predicha es relativa al sistema de coordenadas de la base de datos, no al estándar del enunciado.

## Interfaz

La app (Qt) se organiza en cuatro pestañas:

- **En Vivo**: conexión con la cámara, segmentación y clasificación en tiempo real sobre el flujo de vídeo, con miniaturas de las piezas detectadas y botón de captura.
- **Análisis**: modo offline para cargar o capturar una imagen estática y ejecutar segmentación/clasificación paso a paso, con sub-pestañas para ver el resultado final, el desglose de la segmentación (canales HSV, umbralización, morfología) o herramientas de evaluación del modelo (matriz de confusión, dispersión).
- **Entrenamiento**: automatiza las cinco etapas del pipeline completo (segmentar base de datos → extraer características → generar plantillas de orientación → entrenar el SVM → evaluar sobre el test), con la posibilidad de saltar etapas costosas si los datos no han cambiado, barras de progreso por etapa y un log detallado.
- **Ajustes**: rutas de plantillas, modelo SVM y scaler, y activación del generador de nombres de archivo con la convención de la base de datos.

## Compilar y ejecutar

El proyecto de escritorio está en `ProyectoPSM/ProyectoPSM.sln` (Visual Studio 2022, toolset v143, plataforma x64 — no hay proyecto qmake/.pro, solo el `.sln`/`.vcxproj`). Para compilarlo hace falta:

- **Visual Studio 2022** con la extensión **Qt VS Tools** y **Qt 6.9.2** (MSVC2022 64-bit) instalado y registrado en la extensión.
- **OpenCV 4.12** (`opencv_world4120.lib` / `opencv_world4120d.lib` en Debug) con sus include/lib paths configurados en el proyecto.
- **Basler Pylon SDK** (la app usa una cámara Basler vía `pylon::CBaslerUniversalInstantCamera`; el instalador de Pylon define la variable de entorno `PYLON_DEV_DIR` que usa el `.vcxproj`).

Pasos:

1. Abrir `ProyectoPSM/ProyectoPSM.sln` en Visual Studio.
2. Elegir configuración `Release|x64` (o `Debug|x64`).
3. Compilar y ejecutar. Si no hay cámara Basler conectada, la app arranca igualmente pero avisa de que la cámara no está lista — se puede seguir usando el modo offline (cargar imagen de disco, segmentar, clasificar) desde la interfaz.

La pestaña de "Entrenamiento" de la propia app permite regenerar todo el pipeline apuntando a las carpetas de `Database/`.

## Estructura del repo

- `Matlab/` — prototipos: segmentación por canal y los distintos clasificadores (color, forma, mixto, orientación) que probamos antes de pasar a C++.
- `ProyectoPSM/` — la aplicación final en C++/Qt/OpenCV.
- `Database/` — `RAW` (fotos originales), `SEGMENTED` (piezas ya recortadas), `SEGMENTED_TRAIN`/`SEGMENTED_TEST` (split fijo, ver `split_manifest.csv`).
- `Proceso Clasificacion.txt` — apuntes internos sobre el flujo de entrenamiento/inferencia que usamos de chuleta mientras desarrollábamos.

## Limitaciones

- **Segmentación**: necesita un fondo homogéneo; texturas marcadas cerca de la pieza meten ruido en la máscara. Los reflejos especulares intensos (piezas de plástico brillante) pueden dejar huecos en la detección del canal S que la morfología no siempre corrige del todo. El sistema no separa piezas en contacto físico (las detecta como un único objeto fusionado) y descarta por diseño cualquier pieza que toque el borde de la imagen.
- **Clasificación**: el SVM depende de que las características extraídas en inferencia sean coherentes con las de entrenamiento; pequeñas discrepancias de segmentación, recorte o iluminación se acumulan en el vector de características y degradan la precisión. Descriptores de contorno como los de Fourier son especialmente sensibles a imperfecciones del contorno. El sistema no tiene mecanismo de recalibración automática ante cambios de cámara, distancia o iluminación, y al ser un clasificador de conjunto cerrado, una pieza no vista en entrenamiento se asignará siempre a la clase más parecida sin detectar que es "desconocida".
- **Orientación (template matching)**: depende de que el preprocesado/segmentación sea consistente entre plantilla y consulta; asume orientaciones discretas (no estima ángulos intermedios); piezas con simetrías parciales pueden confundirse entre orientaciones separadas 180°; y la calidad de cada plantilla depende de tener suficientes muestras representativas para esa combinación (clase, yaw, pitch).
- El clasificador de orientación necesita una plantilla por cada combinación de clase × yaw × pitch generada de antemano — añadir una pieza nueva implica volver a fotografiarla en todas las orientaciones y regenerar plantillas.
- La app solo funciona con la cámara Basler concreta que usamos (vía Pylon) y solo compila en Windows/Visual Studio; no hay build para Linux ni soporte de webcam genérica.
