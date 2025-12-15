clear all;close all;clc;
load('legoFeatures_TRAIN_todas_8carac.mat'); % T todas 
%load('legoFeatures_TRAIN_225_8carac.mat'); % T2 225
%load('legoFeatures_TRAIN_todas_sin225_8carac.mat'); % T3 todas - 225



%% Seleccionar de forma aleatoria las Train y las Test
%load('I_random.mat');
%I=randperm(1552);
T_train=T(I(1:1200),:);
T_test=T(I(1201:end),:);

L_train = T.Label(I(1:1200),:);
L_test = T.Label(I(1201:end),:);

% Quiero entrenar ahora con las train en classLearner. SVM funciona muy bien. Hacer crossvalidation y si quiero test tb, en el menu de new session. Y luego clasificar
% con las test con el predict.

%% CLASIFICAR UNA IMAGEN ALEATORIA DEL CONJUNTO T_test

% 1) CARGAR MODELO (si no está ya en workspace)
% if exist('trainedModel','var') ~= 1
%     load('trainedModel_Ttrain.mat');   % ajusta si hace falta
%     if exist('trainedModel_Ttrain','var') == 1
%         trainedModel = trainedModel_Ttrain;
%     end
% end

% 2) ELEGIR UNA FILA ALEATORIA DE T_test
rng('shuffle');   % para que sea distinta cada vez
idx = randi(height(T_test));

row = T_test(idx, :);

fprintf('Fila seleccionada: %d de %d\n', idx, height(T_test));

% 3) RECUPERAR NOMBRE DE IMAGEN Y CLASE REAL
imgName = row.FileName{1};     % p.ej. '07_225_40_003.png'
trueLabel = row.Label;

% 4) CONSTRUIR RUTA COMPLETA A LA IMAGEN
segFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED';
imgPath = fullfile(segFolder, imgName);

if ~isfile(imgPath)
    error('No se encuentra la imagen: %s', imgPath);
end

% 5) LEER IMAGEN
Ipiece = imread(imgPath);

% 6) EXTRAER FEATURES
feat = extractColorFeatures(Ipiece);   % 1 x 8

feat_test = table2array(T_test(idx, 1:end-3));  % 1×8 double
a=0;
for i = 1:length(feat)
    if feat(i) == feat_test(i)
        a=a+1;
    else
        continue;
    end
end
if a==8 
    disp("Coinciden") 
end

% 7) CREAR TABLA PARA EL MODELO
featTable = array2table(feat, ...
    'VariableNames', trainedModel.RequiredVariables);

% 8) PREDECIR
predictedLabel = trainedModel.predictFcn(featTable);

% 9) MOSTRAR RESULTADO
fprintf('\nImagen: %s\n', imgName);
fprintf('Clase REAL     : %s\n', string(trueLabel));
fprintf('Clase PREDICHA : %s\n', string(predictedLabel));

figure('Name','Test sobre T\_test','NumberTitle','off');
imshow(Ipiece);
title(sprintf('Real: %s | Pred: %s', ...
      string(trueLabel), string(predictedLabel)), 'FontSize', 14);


%% CLASIFICAR PIEZAS (UNA POR IMAGEN) Y GUARDAR RESULTADOS EN .TXT
% clear; clc;
% 
% % 1) CARGAR MODELO EXPORTADO DESDE CLASSIFICATION LEARNER
% %load('trainedModel_todas_aleatorias.mat');   % contiene la struct trainedModel
% load('trainedModel_Ttrain.mat');   % contiene la struct trainedModel_Ttrain
% 
% % 2) CARPETA CON LAS PIEZAS SEGMENTADAS (UNA POR IMAGEN)
% segFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_Una_Pieza';
% 
% outputTxt = fullfile("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador", ...
%                      'resultados_clasificacion_segmentadas.txt');
% 
% % 3) LISTAR PIEZAS
% filesPNG = dir(fullfile(segFolder, '*.png'));
% filesJPG = dir(fullfile(segFolder, '*.jpg'));
% files    = [filesPNG; filesJPG];
% 
% fprintf('Se han encontrado %d piezas en %s\n', numel(files), segFolder);
% 
% if isempty(files)
%     error('No hay piezas en la carpeta especificada.');
% end
% 
% % 4) OBTENER CÓDIGO (01..09) DE CADA PIEZA Y ELEGIR HASTA 50 POR CÓDIGO
% 
% Nall   = numel(files);
% codes  = cell(Nall,1);  % código de cada pieza (primer bloque)
% 
% for i = 1:Nall
%     fname = files(i).name;
%     [~, baseName, ~] = fileparts(fname);        % ej: '07_225_40_003_piece01'
%     partes = split(baseName, '_');              % {'07','225','40','003','piece01'}
%     if ~isempty(partes)
%         codes{i} = char(partes(1));             % '07'
%     else
%         codes{i} = 'UNKNOWN';
%     end
% end
% 
% uniqueCodes   = unique(codes);
% selectedFiles = [];
% 
% for i = 1:numel(uniqueCodes)
%     code = uniqueCodes{i};
% 
%     % Índices de las piezas que tienen este código
%     idx = find(strcmp(codes, code));
% 
%     nAvailable = numel(idx);
%     if nAvailable == 0
%         continue;
%     end
% 
%     nSelect = min(50, nAvailable);          % máximo 50 por código
%     idxSel  = idx(randperm(nAvailable, nSelect));
% 
%     selectedFiles = [selectedFiles; files(idxSel)]; %#ok<AGROW>
% 
%     fprintf('Código %s -> %d disponibles, seleccionadas %d\n', ...
%             code, nAvailable, nSelect);
% end
% 
% files = selectedFiles;
% N     = numel(files);
% 
% fprintf('\nTotal de piezas seleccionadas para clasificar: %d\n\n', N);
% 
% if N == 0
%     error('No se ha seleccionado ninguna pieza (revisa nombres/códigos).');
% end
% 
% % 5) ABRIR ARCHIVO .TXT PARA GUARDAR RESULTADOS
% fid = fopen(outputTxt, 'w');
% if fid == -1
%     error('No se pudo crear el archivo de salida: %s', outputTxt);
% end
% 
% fprintf(fid, 'Resultados de clasificación de piezas segmentadas (una por imagen)\n');
% fprintf(fid, '=================================================================\n\n');
% 
% % 6) CLASIFICAR CADA PIEZA SELECCIONADA
% for n = 1:N
%     pieceName = files(n).name;
%     piecePath = fullfile(files(n).folder, pieceName);
% 
%     % Leer imagen (pieza)
%     Ipiece = imread(piecePath);
% 
%     % Extraer características
%     feat = extractColorFeatures(Ipiece);   % 1 x D
% 
%     % Convertir a tabla con los nombres que espera el modelo
%     featTable = array2table(feat, ...
%         'VariableNames', trainedModel.RequiredVariables);
% 
%     % Clasificación
%     predictedLabel = trainedModel.predictFcn(featTable);
% 
%     % Nombre base de la imagen original (sin _piece)
%     [~, baseName, ~] = fileparts(pieceName);   % p.ej. '07_225_40_003_piece01'
%     partes = split(baseName, '_piece');
%     if numel(partes) >= 2
%         imgBaseName = partes{1};               % '07_225_40_003'
%     else
%         imgBaseName = baseName;
%     end
% 
%     % Escribir resultado en el txt
%     fprintf(fid, 'Imagen %s -> clase predicha %s\n', ...
%             imgBaseName, string(predictedLabel));
% 
%     % Progreso
%     fprintf('Procesada pieza %d/%d\n', n, N);
% end
% 
% % 7) CERRAR ARCHIVO
% fclose(fid);
% 
% fprintf('\nClasificación finalizada.\nResultados guardados en:\n%s\n', outputTxt);

