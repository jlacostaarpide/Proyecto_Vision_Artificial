%% TEST CLASIFICADOR PIEZAS LEGO


%% === 1) CARGAR MODELO ENTRENADO ===

load('legoFeatures_TrainingSet8carac.mat');   % el modelo exportado desde Classification Learner
% Alternativa si usaste fitcknn:
% load('legoModel_porCodigo.mat'); % variable Mdl

%% === 2) SELECCIONAR IMAGEN A TESTEAR ===

testImage = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\angulos225\05_225_10_003.jpg';

fprintf("\n--- Clasificando imagen: %s ---\n", testImage);

%% === 3) SEGMENTAR LA IMAGEN ===

[pieces_test, stats_test, num_test, Icorr] = segmentarPiezas2(testImage);

if num_test == 0
    error("No se detectaron piezas en la imagen de test");
end

%% === 4) CLASIFICAR CADA PIEZA DETECTADA ===
figure('Name','Original + Clasificación','NumberTitle','off');

% -------------------------------------------------------------
% SUBPLOT 1: IMAGEN ORIGINAL COMPLETA
% -------------------------------------------------------------
subplot(2,1,1);   % fila 1 de 2, columna única
imshow(testImage);
title('Imagen original', 'FontSize', 14);

% -------------------------------------------------------------
% SUBPLOT 2: PIEZAS SEGMENTADAS CON PREDICCIÓN
% -------------------------------------------------------------
subplot(2,1,2);   % fila 2 de 2, columna única

% Creamos una cuadrícula interna para las piezas
rows = 1;
cols = num_test;

tiledlayout(rows, cols, 'TileSpacing','compact');

for k = 1:num_test
    
    Ipiece = pieces_test{k};

    % 1) Extraer características
    feat = extractColorFeatures(Ipiece);

    % 2) Convertir a tabla si usas trainedModel de Classification Learner
    featTable = array2table(feat, ...
        'VariableNames', trainedModel.RequiredVariables);

    % 3) Predicción
    predictedLabel = trainedModel.predictFcn(featTable);

    % Mostrar
    nexttile;
    imshow(Ipiece);
    title(sprintf('Pred: %s', string(predictedLabel)), 'FontSize', 12);
end
