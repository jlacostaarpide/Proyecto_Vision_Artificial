%% ================================================================
%% CLASIFICAR TODAS LAS IMÁGENES DE UNA CARPETA (SEGMENTA -> FEATURES -> PREDICE)
% - Lee TODAS las imágenes originales de una carpeta
% - Segmenta con segmentarPiezas2 (solo acepta si n==1)
% - Extrae features con extractShapeFeatures(piece)
% - Clasifica con trainedModel (Classification Learner)
% - Guarda resultados en tabla F_run (una fila por imagen válida)
%% ================================================================
%clear; clc;

%% 1) CARGAR MODELO (descomenta el que uses)
% load("TrainedModelWith_Mtrain.mat");

% Si tu .mat no deja una variable llamada trainedModel, adapta aquí:
% trainedModel = trainedModel_Ttrain;  % ejemplo

%% 2) CARPETA ORIGEN (IMÁGENES SIN SEGMENTAR)
%folderPath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test_amarillas";
%folderPath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test_rojas";
folderPath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test2";

%% 3) OPCIONES
showFigure = false;

%% 4) LISTAR TODAS LAS IMÁGENES Y ORDENAR NUMÉRICAMENTE (1.jpg, 2.jpg, 12.jpg...)
files = [dir(fullfile(folderPath,'*.png')); ...
         dir(fullfile(folderPath,'*.jpg')); ...
         dir(fullfile(folderPath,'*.jpeg'))];

Nall = numel(files);
if Nall == 0
    error('No hay imágenes en: %s', folderPath);
end

nums = nan(Nall,1);
for i = 1:Nall
    [~, name, ~] = fileparts(files(i).name);  % '12'
    nums(i) = str2double(name);               % 12 o NaN si no es numérico
end

isNum = ~isnan(nums);

idxNum = find(isNum);                 % índices de ficheros con nombre numérico
[~, ord] = sort(nums(isNum), 'ascend');
idxSort = [idxNum(ord); find(~isNum)]; % primero numéricas ordenadas, luego el resto

files = files(idxSort);
Nall  = numel(files);

%% 5) NOMBRES DE FEATURES (15) -> deben coincidir con extractShapeFeatures y el modelo
% featNames = {'Area','Perimeter','Circularity','Eccentricity','Solidity','Extent', ...
%              'AspectRatio','BBoxRatio', ...
%              'Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7'};

featNames = { ...
 'H_mean_circ','H_var_circ','S_median','S_IQR','V_median','V_IQR','S_mean','V_mean', ...
 'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
 'SkelLenNorm','SkelEndpoints','SkelBranchpoints','FD2','FD3','FD4','FD5' };


fprintf('Clasificando %d imágenes (con segmentación) de: %s\n\n', Nall, folderPath);

%% 6) CONTENEDORES PARA "UNA FILA POR IMAGEN"
Xrun      = zeros(0, numel(featNames));  % Nx15
file_cell = {};
pred_cell = {};

nOkSeg   = 0;
nDiscard = 0;

%% 7) LOOP: SEGMENTAR -> FEATURES -> PREDICCIÓN
for i = 1:Nall
    imgName = files(i).name;
    imgPath = fullfile(files(i).folder, imgName);

    fprintf('%s', imgName);

    % 7.1) SEGMENTAR (solo si devuelve 1 pieza)
    try
        [piece, ~, n] = segmentarPiezas2(imgPath);
    catch ME
        fprintf(' -> ERROR segmentando (%s)\n', ME.message);
        nDiscard = nDiscard + 1;
        continue;
    end

    if n ~= 1 || isempty(piece)
        fprintf(' -> DESCARTADA (n=%d)\n', n);
        nDiscard = nDiscard + 1;
        continue;
    end
    nOkSeg = nOkSeg + 1;

    % 7.2) EXTRAER FEATURES (1x15)
    feat = extractColorShapeFeatures(piece);

    % 7.3) PREDECIR
    featTable = array2table(feat, 'VariableNames', featNames);
    pred = trainedModel.predictFcn(featTable);

    fprintf(' -> Pred: %s\n', string(pred));

    % 7.4) GUARDAR UNA FILA (UNA IMAGEN)
    Xrun(end+1, :)   = feat;               %#ok<AGROW>
    file_cell{end+1} = imgName;            %#ok<AGROW>
    pred_cell{end+1} = char(pred);         %#ok<AGROW>

    % 7.5) VISUALIZACIÓN opcional
    if showFigure
        figure('Name','Segmentación + Predicción','NumberTitle','off');
        tiledlayout(1,2,'TileSpacing','compact');
        nexttile; imshow(imread(imgPath));
        title(sprintf('Original: %s', imgName), 'Interpreter','none', 'FontSize', 11);
        nexttile; imshow(piece);
        title(sprintf('Pred: %s', string(pred)), 'FontSize', 12);
    end
end

%% 8) TABLA FINAL: UNA FILA POR IMAGEN VÁLIDA
F_run = array2table(Xrun, 'VariableNames', featNames);
F_run.FileName  = string(file_cell(:));
F_run.PredLabel = categorical(pred_cell(:));

fprintf('\nResumen:\n');
fprintf('  Segmentadas OK (n==1): %d\n', nOkSeg);
fprintf('  Descartadas           : %d\n', nDiscard);

disp('Ejemplo de F_run:');
disp(F_run(1:min(5,height(F_run)), :));

%% 9) (OPCIONAL) Calcular centroides F_mean SOLO de tu dataset F (si existe)
% Esto NO usa lo de la carpeta, usa tu F (train) si está en workspace.
if exist('F','var') == 1
    featNamesF = F.Properties.VariableNames;
    featNamesF = featNamesF(1:15);
    F_mean = groupsummary(F, 'Label', 'mean', featNamesF);
    disp('✔ F_mean (centroides por Label) calculada a partir de F.');
end
