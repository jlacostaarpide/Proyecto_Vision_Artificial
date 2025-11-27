% Juan:
clc;
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\tests");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789");

%% 2. SEGMENTAR la escena de TEST
%imagen_test = 'IMG_rojas.jpg';
%imagen_test = 'IMG_morada_verde_naranja.jpg';
%imagen_test = '06_270_70_004.jpg';
%imagen_test = '08_270_70_004.jpg';

[images_test, stats_test, num_test, I_corr_test] = segmentarPiezas(imagen_test);

fprintf("Piezas detectadas en la imagen test: %d\n", num_test);

%% 3. VISUALIZAR piezas detectadas
figure;
for k = 1:num_test
    subplot(2,2,k);
    imshow(images_test{k});
    title(sprintf("Test pieza %d", k));
end

%% 4. CLASIFICAR cada pieza
figure;
for k = 1:num_test
    feat_k = extractColorFeatures(images_test{k});
    label_k = predict(Mdl, feat_k);

    subplot(2,2,k);
    imshow(images_test{k});
    title(sprintf("Pred: %s", string(label_k)));
end
