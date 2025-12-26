DataSet de características guardado en: "legoFeatures_TRAIN_color_shape_22carac.mat"

Para sacar M_train y M_test, cargar I_random.

Meter en classificationLearner y hacer un quadratic SVM. Sale accuracy de 99.6%

Ahora al ejecutar el script con M_test obtenemos:
Aciertos: 534 | Fallos: 1 | Missing: 0 | Acc: 99.81%

He reducido caracteristicas a 12:
Aciertos: 534 | Fallos: 1 | Missing: 0 | Acc: 99.81%


Vamos a comprobar con otras fotos que no sean de esas:
Cogemos la carpeta SEGMENTED_test2_local (test2 con muchas rojas y amarillas previamente segmentadas)

Ejecutamos la parte del script de abajo del todo que son para carpetas random y obtenemos esto:

Aciertos: 35 | Fallos: 6 | Missing: 0 | Acc: 85.37%

Falla sobretodo aun en las 3,6,9,12. Sobre todo las 9,12:

LISTA DE FALLOS (si los hay)
----------------------------
03_5.jpg | REAL=03 | PRED=06
09_1.jpg | REAL=09 | PRED=05
12_2.jpg | REAL=12 | PRED=11
12_5.jpg | REAL=12 | PRED=09
12_7.jpg | REAL=12 | PRED=09
12_8.jpg | REAL=12 | PRED=09