clear all;close all;clc;
load('legoFeatures_TEST_8carac.mat'); % T todas 
load('legoFeatures_TEST_225_8carac.mat'); % T2 225
load('legoFeatures_TEST_todas_sin225_8carac.mat'); % T3 todas - 225

%%

figure; gplotmatrix(table2array(T(:,1:end-3)),[],T.Label); % como le hemos metido 3 columnas nuevas que no queremos ver, hacemos desde columna 1 hasta end-3.
figure; gplotmatrix(table2array(T2(:,1:end-3)),[],T2.Label);
figure; gplotmatrix(table2array(T3(:,1:end-3)),[],T3.Label);

%%
figure; gplotmatrix(table2array(T(:,1:4)),[],T.Label,[],'*'); % como le hemos metido 3 columnas nuevas que no queremos ver, hacemos desde columna 1 hasta end-3.
figure; gplotmatrix(table2array(T2(:,1:4)),[],T2.Label,[],'*');
figure; gplotmatrix(table2array(T3(:,1:4)),[],T3.Label,[],'*');

%% Seleccionar de forma aleatoria las Train y las Test

I=randperm(1642);
T_train=T(I(1:1200),:);
T_test=T(I(1201:end),:);

L_train = T.Label(I(1:1200),:);
L_test = T.Label(I(1201:end),:);

% Quiero entrenar ahora con las train en classLearner. SVM funciona muy bien. Hacer crossvalidation y si quiero test tb, en el menu de new session. Y luego clasificar
% con las test con el predict.

%% 2) RUTA DE LA IMAGEN A CLASIFICAR
imgPath = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\Clasificador\02_270_10_003.jpg';


fprintf('\n--- Clasificando imagen: %s ---\n', imgPath);

% 3) SEGMENTAR LA IMAGEN
[pieces_test, stats_test, num_test, Icorr] = segmentarPiezas2(imgPath);

if num_test == 0
    error('No se detectaron piezas en la imagen de test');
end

% 4) CLASIFICAR CADA PIEZA
for k = 1:num_test
    
    Ipiece = pieces_test{k};

    % 1) Extraer características (1 x D)
    feat = extractColorFeatures(Ipiece);   % p.ej. 1x8 double

    % 2) Convertir a tabla con los mismos nombres que en el entrenamiento
    featTable = array2table(feat, ...
        'VariableNames', trainedModel.RequiredVariables);

    % 3) Predecir usando el modelo exportado
    predictedLabel = trainedModel.predictFcn(featTable);

    fprintf('Pieza %d -> Predicción: %s\n', k, string(predictedLabel));

    % (Opcional) Mostrar la pieza con el label
    figure;
    imshow(Ipiece);
    title(sprintf('Pieza %d - Pred: %s', k, string(predictedLabel)), 'FontSize', 14);
end

%% CLASIFICAR PIEZAS (UNA POR IMAGEN) Y GUARDAR RESULTADOS EN .TXT
clear; clc;

% 1) CARGAR MODELO EXPORTADO DESDE CLASSIFICATION LEARNER
%load('trainedModel_todas_aleatorias.mat');   % contiene la struct trainedModel
load('trainedModel_Ttrain.mat');   % contiene la struct trainedModel_Ttrain

% 2) CARPETA CON LAS PIEZAS SEGMENTADAS (UNA POR IMAGEN)
segFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_Una_Pieza';

outputTxt = fullfile("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador", ...
                     'resultados_clasificacion_segmentadas.txt');

% 3) LISTAR PIEZAS
filesPNG = dir(fullfile(segFolder, '*.png'));
filesJPG = dir(fullfile(segFolder, '*.jpg'));
files    = [filesPNG; filesJPG];

fprintf('Se han encontrado %d piezas en %s\n', numel(files), segFolder);

if isempty(files)
    error('No hay piezas en la carpeta especificada.');
end

% 4) OBTENER CÓDIGO (01..09) DE CADA PIEZA Y ELEGIR HASTA 50 POR CÓDIGO

Nall   = numel(files);
codes  = cell(Nall,1);  % código de cada pieza (primer bloque)

for i = 1:Nall
    fname = files(i).name;
    [~, baseName, ~] = fileparts(fname);        % ej: '07_225_40_003_piece01'
    partes = split(baseName, '_');              % {'07','225','40','003','piece01'}
    if ~isempty(partes)
        codes{i} = char(partes(1));             % '07'
    else
        codes{i} = 'UNKNOWN';
    end
end

uniqueCodes   = unique(codes);
selectedFiles = [];

for i = 1:numel(uniqueCodes)
    code = uniqueCodes{i};

    % Índices de las piezas que tienen este código
    idx = find(strcmp(codes, code));

    nAvailable = numel(idx);
    if nAvailable == 0
        continue;
    end

    nSelect = min(50, nAvailable);          % máximo 50 por código
    idxSel  = idx(randperm(nAvailable, nSelect));

    selectedFiles = [selectedFiles; files(idxSel)]; %#ok<AGROW>

    fprintf('Código %s -> %d disponibles, seleccionadas %d\n', ...
            code, nAvailable, nSelect);
end

files = selectedFiles;
N     = numel(files);

fprintf('\nTotal de piezas seleccionadas para clasificar: %d\n\n', N);

if N == 0
    error('No se ha seleccionado ninguna pieza (revisa nombres/códigos).');
end

% 5) ABRIR ARCHIVO .TXT PARA GUARDAR RESULTADOS
fid = fopen(outputTxt, 'w');
if fid == -1
    error('No se pudo crear el archivo de salida: %s', outputTxt);
end

fprintf(fid, 'Resultados de clasificación de piezas segmentadas (una por imagen)\n');
fprintf(fid, '=================================================================\n\n');

% 6) CLASIFICAR CADA PIEZA SELECCIONADA
for n = 1:N
    pieceName = files(n).name;
    piecePath = fullfile(files(n).folder, pieceName);

    % Leer imagen (pieza)
    Ipiece = imread(piecePath);

    % Extraer características
    feat = extractColorFeatures(Ipiece);   % 1 x D

    % Convertir a tabla con los nombres que espera el modelo
    featTable = array2table(feat, ...
        'VariableNames', trainedModel.RequiredVariables);

    % Clasificación
    predictedLabel = trainedModel.predictFcn(featTable);

    % Nombre base de la imagen original (sin _piece)
    [~, baseName, ~] = fileparts(pieceName);   % p.ej. '07_225_40_003_piece01'
    partes = split(baseName, '_piece');
    if numel(partes) >= 2
        imgBaseName = partes{1};               % '07_225_40_003'
    else
        imgBaseName = baseName;
    end

    % Escribir resultado en el txt
    fprintf(fid, 'Imagen %s -> clase predicha %s\n', ...
            imgBaseName, string(predictedLabel));

    % Progreso
    fprintf('Procesada pieza %d/%d\n', n, N);
end

% 7) CERRAR ARCHIVO
fclose(fid);

fprintf('\nClasificación finalizada.\nResultados guardados en:\n%s\n', outputTxt);
