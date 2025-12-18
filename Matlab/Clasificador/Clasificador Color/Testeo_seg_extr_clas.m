%% ================================================================
%% CLASIFICAR TODAS LAS IMÁGENES DE UNA CARPETA (SEGMENTA -> FEATURES -> PREDICE)
% - Lee TODAS las imágenes originales de una carpeta
% - Segmenta con segmentarPiezas2 (solo acepta si n==1)
% - Extrae features con extractColorFeatures(piece)
% - Clasifica con trainedModel (Classification Learner)
% - Muestra resultados por consola y (opcional) figura
%% ================================================================
clear; clc;

%% 1) CARGAR MODELO
load("TrainedModelWith_Ttrain.mat");   % debe dejar: trainedModel
% Si en tu .mat la variable se llama trainedModel_Ttrain, descomenta:
% if exist('trainedModel','var')~=1 && exist('trainedModel_Ttrain','var')==1
%     trainedModel = trainedModel_Ttrain;
% end

%% 2) CARPETA ORIGEN (IMÁGENES SIN SEGMENTAR)
folderPath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test2";

%% 3) OPCIONES
showFigure = false;     % false si no quieres figuras

%% 4) LISTAR TODAS LAS IMÁGENES
files = [dir(fullfile(folderPath,'*.png')); dir(fullfile(folderPath,'*.jpg')); dir(fullfile(folderPath,'*.jpeg'))];
Nall = numel(files);
if Nall == 0
    error('No hay imágenes en: %s', folderPath);
end

%% 5) NOMBRES DE FEATURES (debe coincidir con tu entrenamiento: 8 variables)
featNames = {'H_mean_circ','H_var_circ','S_median','S_IQR','V_median','V_IQR','S_mean','V_mean'};

fprintf('Clasificando %d imágenes (con segmentación) de: %s\n\n', Nall, folderPath);

%% 6) LOOP: SEGMENTAR -> FEATURES -> PREDICCIÓN
nOkSeg   = 0;
nDiscard = 0;

for i = 1:Nall
    imgName = files(i).name;
    imgPath = fullfile(files(i).folder, imgName);

    fprintf('[%04d/%04d] %s', i, Nall, imgName);

    % --- 6.1) SEGMENTAR (solo útil si devuelve 1 pieza) ---
    try
        [piece, ~, n] = segmentarPiezas2(imgPath);
    catch ME
        warning('   >> Error en segmentarPiezas2: %s (saltando)', ME.message);
        nDiscard = nDiscard + 1;
        continue;
    end

    if n ~= 1 || isempty(piece)
        fprintf('   >> Descartada (n=%d)\n', n);
        nDiscard = nDiscard + 1;
        continue;
    end
    nOkSeg = nOkSeg + 1;

    % --- 6.2) EXTRAER FEATURES DE LA PIEZA SEGMENTADA ---
    feat = extractColorFeatures(piece);          % 1x8
    featTable = array2table(feat, 'VariableNames', featNames);

    % --- 6.3) PREDECIR ---
    pred = trainedModel.predictFcn(featTable);
    fprintf('   -> Pred: %s\n', string(pred));

    % --- 6.4) VISUALIZACIÓN ---
    if showFigure
        figure('Name','Segmentación + Predicción','NumberTitle','off');
        tiledlayout(1,2,'TileSpacing','compact');

        nexttile; imshow(imread(imgPath));
        title(sprintf('Original: %s', imgName), 'Interpreter','none', 'FontSize', 11);

        nexttile; imshow(piece);
        title(sprintf('Pred: %s', string(pred)), 'FontSize', 12);
    end
end

fprintf('\nResumen:\n');
fprintf('  Segmentadas OK (n==1): %d\n', nOkSeg);
fprintf('  Descartadas           : %d\n', nDiscard);
