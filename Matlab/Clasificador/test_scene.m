%% TEST DE UNA ESCENA

clear; close all; clc;

%basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123";
%basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456";
basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789";
basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\tests\";


addpath(basePath);

%load('legoModel_porCodigoPrimerBloque.mat','Mdl','classNames','codigoClases');
%load('legoModel_porCodigoSegundoBloque.mat','Mdl','classNames','codigoClases');
load('legoModel_porCodigoTercerBloque.mat','Mdl','classNames','codigoClases');

nombre_imagen = fullfile(basePath, 'amarillas.jpg');  % por ejemplo
[images_final, stats_final, num_final, I_corrected] = segmentarPiezas2(nombre_imagen);

numPiezas = sum(~cellfun('isempty', images_final));

figure;
for k = 1:numPiezas
    I_piece = images_final{k};
    feat_k  = extractColorFeatures(I_piece);
    label_k = predict(Mdl, feat_k);

    subplot(2,2,k);
    imshow(I_piece);
    title(sprintf('Pred: %s', string(label_k)));
end
