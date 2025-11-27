%% CLASIFICADOR PIEZAS LEGO 

clear; close all; clc;

% Juan:
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\tests");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789");

%% 1. SEGMENTAR UNA IMAGEN (ESCENA)
nombre_imagen = 'IMG_7650.jpg';

[images_final, stats_final, num_final, I_corrected] = segmentarPiezas(nombre_imagen);

% Guardamos las piezas originales
images_orig   = images_final;
numPiezasOrig = sum(~cellfun('isempty', images_orig));

% Mostrar las piezas segmentadas originales
figure;
for k = 1:numPiezasOrig
    subplot(2,2,k);
    imshow(images_orig{k});
    title(sprintf('Pieza %d', k));
end

%% 2. DATA AUGMENTATION: rotaciones 0:45:315

angles = 0:45:315;          % 0,45,...,315 (8 rotaciones)
images_train = {};
labels_cell  = {};

for c = 1:numPiezasOrig
    Ibase = images_orig{c};
    for ang = angles
        % Rotar la pieza
        Irot = imrotate(Ibase, ang, 'crop');
        % Añadir a la lista de entrenamiento
        images_train{end+1} = Irot;           %#ok<SAGROW>
        labels_cell{end+1}  = sprintf('clase%d', c); %#ok<SAGROW>
    end
end

% Convertimos etiquetas a categorical
labels_train = categorical(labels_cell);

%% 3. EXTRAER CARACTERÍSTICAS PARA TODAS LAS ROTACIONES

numTrain = numel(images_train);
Xtrain   = zeros(numTrain, 6);   % 6: mean/std de H,S,V

for i = 1:numTrain
    I = images_train{i};
    Xtrain(i,:) = extractColorFeatures(I);
end
Ytrain = labels_train;

%% 4. ENTRENAR k-NN

Mdl = fitcknn(Xtrain, Ytrain, ...
              'NumNeighbors', 3, ...   % ahora tiene sentido K=3
              'Standardize', true);

%% 5. GUARDAR MODELO ENTRENADO
save('legoModelFrom7650.mat','Mdl');
disp("✔ Modelo entrenado y guardado como legoModelFrom7650.mat");

%% 6. CLASIFICAR LAS 4 PIEZAS ORIGINALES (COMPROBAR)

figure;
for k = 1:numPiezasOrig
    I_piece = images_orig{k};
    feat_k  = extractColorFeatures(I_piece);
    label_k = predict(Mdl, feat_k);

    subplot(2,2,k);
    imshow(I_piece);
    title(sprintf('Pred: %s', string(label_k)));
end
