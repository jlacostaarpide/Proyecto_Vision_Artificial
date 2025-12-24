DataSet de características guardado en: "legoFeatures_TRAIN_todas_8carac.mat"

Para sacar T_train y T_test, cargar I_random.

Meter en classificationLearner y hacer un quadratic SVM. Sale accuracy de 97.3%

Ahora al ejecutar el script con T_test obtenemos:
Aciertos: 335 | Fallos: 0 | Missing: 0 | Acc: 100.00%
Cambios 3-6: 4/48 | Cambios 9-12: 0/39

Los cambios han sido los siguientes:
LISTA DE CAMBIOS (BASE -> FINAL)
--------------------------------
03_180_70_004.jpg | REAL=03 | BASE=06 -> FINAL=03 | REF=3-6 | OK
06_090_10_005.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
03_225_70_005.jpg | REAL=03 | BASE=06 -> FINAL=03 | REF=3-6 | OK
06_045_10_004.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK


Vamos a comprobar con otras fotos que no sean de esas:
Cogemos la carpeta SEGMENTED_test2_local (test2 con muchas rojas y amarillas previamente segmentadas)

Ejecutamos la parte del script de abajo del todo que son para carpetas random y obtenemos esto:

GT=41 | OK=34 | FAIL=7 | ACC=82.93%

LISTA DE FALLOS (si hay GT)
---------------------------
09_2.jpg | REAL=09 | BASE=09 | FINAL=12 | REF=9-12
09_6.jpg | REAL=09 | BASE=09 | FINAL=12 | REF=9-12
09_8.jpg | REAL=09 | BASE=05 | FINAL=05 | REF=none
12_2.jpg | REAL=12 | BASE=05 | FINAL=05 | REF=none
12_4.jpg | REAL=12 | BASE=05 | FINAL=05 | REF=none
12_5.jpg | REAL=12 | BASE=09 | FINAL=09 | REF=9-12
12_9.jpg | REAL=12 | BASE=09 | FINAL=09 | REF=9-12

LISTA DE CAMBIOS (BASE -> FINAL)
--------------------------------
03_2.jpg | REAL=03 | BASE=06 -> FINAL=03 | REF=3-6 | OK
03_3.jpg | REAL=03 | BASE=06 -> FINAL=03 | REF=3-6 | OK
03_7.jpg | REAL=03 | BASE=06 -> FINAL=03 | REF=3-6 | OK
06_2.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
06_4.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
06_5.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
06_6.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
06_7.jpg | REAL=06 | BASE=03 -> FINAL=06 | REF=3-6 | OK
09_2.jpg | REAL=09 | BASE=09 -> FINAL=12 | REF=9-12 | FAIL
09_6.jpg | REAL=09 | BASE=09 -> FINAL=12 | REF=9-12 | FAIL
12_1.jpg | REAL=12 | BASE=09 -> FINAL=12 | REF=9-12 | OK
12_3.jpg | REAL=12 | BASE=09 -> FINAL=12 | REF=9-12 | OK
12_6.jpg | REAL=12 | BASE=09 -> FINAL=12 | REF=9-12 | OK
12_7.jpg | REAL=12 | BASE=09 -> FINAL=12 | REF=9-12 | OK
12_8.jpg | REAL=12 | BASE=09 -> FINAL=12 | REF=9-12 | OK

