clear all;close all;clc;
%load('legoFeatures_TRAIN_color_shape_22carac.mat'); % M todas carac
load('legoFeatures_TRAIN_color_shape_12carac.mat'); % M todas 12 carac



%% Seleccionar de forma aleatoria las Train y las Test
load('I_random.mat');
% I=randperm(1923);
M_train=M(I(1:1500),:);
M_test=M(I(1501:end),:);

%F_red = M(1:314,:);
%F_yellow = M(315:end,:);

% L_train = M.Label(I(1:1500),:);
% L_test = M.Label(I(1501:end),:);

% Quiero entrenar ahora con las train en classLearner. SVM funciona muy bien. Hacer crossvalidation y si quiero test tb, en el menu de new session. Y luego clasificar
% con las test con el predict.


%% ================================================================
%% EVALUAR TODO M_test: predecir, guardar a TXT y contar aciertos
%% Requiere: M_test en workspace + trainedModel cargado
%% ================================================================

% --- Ajusta rutas ---
segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Mtest_12carac.txt';

%load("TrainedModelWith_Mtrain.mat");
load("TrainedModelWith_Mtrain_12.mat");

% --- Predictor names (las 12 primeras columnas son features) ---
predictorNames = M_test.Properties.VariableNames(1:12);

% --- Abrir TXT ---
fid = fopen(outputTxt,'w');
if fid==-1
    error('No se pudo crear el archivo: %s', outputTxt);
end

fprintf(fid, 'EVALUACIÓN COMPLETA SOBRE M_test\n');
fprintf(fid, '================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

N = height(M_test);
nOK = 0;
nFail = 0;
nMissing = 0;

% (Opcional) almacenar fallos para resumen
failList = strings(0,1);

for i = 1:N
    imgName   = M_test.FileName{i};
    trueLabel = M_test.Label(i);

    imgPath = fullfile(segFolder, imgName);

    if ~isfile(imgPath)
        fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=--- | ERROR: NO FILE\n', ...
            i, N, imgName, string(trueLabel));
        nMissing = nMissing + 1;
        continue;
    end

    % Leer imagen y recalcular features (pipeline real)
    Ipiece = imread(imgPath);
    feat   = extractColorShapeFeatures(Ipiece);             % 1x8
    featTable = array2table(feat, 'VariableNames', predictorNames);

    predictedLabel = trainedModel.predictFcn(featTable);

    % Comparar
    isCorrect = (predictedLabel == trueLabel);

    if isCorrect
        nOK = nOK + 1;
    else
        nFail = nFail + 1;
        failList(end+1,1) = sprintf('%s | REAL=%s | PRED=%s', ...
                                    imgName, string(trueLabel), string(predictedLabel));
    end

    % Guardar línea en TXT
    if isCorrect
    resultStr = 'OK';
    else
        resultStr = 'FAIL';
    end
    
    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=%s | %s\n', ...
    i, N, imgName, string(trueLabel), string(predictedLabel), resultStr);

    % Progreso en consola
    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

% --- Resumen ---
totalEvaluated = nOK + nFail; % excluye missing
acc = 0;
if totalEvaluated > 0
    acc = 100 * (nOK / totalEvaluated);
end

fprintf(fid, '\n\nRESUMEN\n');
fprintf(fid, '------\n');
fprintf(fid, 'Total filas M_test        : %d\n', N);
fprintf(fid, 'Imágenes no encontradas   : %d\n', nMissing);
fprintf(fid, 'Evaluadas (con archivo)   : %d\n', totalEvaluated);
fprintf(fid, 'Aciertos                 : %d\n', nOK);
fprintf(fid, 'Fallos                   : %d\n', nFail);
fprintf(fid, 'Accuracy (sin missing)    : %.2f %%\n', acc);

% (Opcional) listar fallos al final
fprintf(fid, '\n\nLISTA DE FALLOS (si los hay)\n');
fprintf(fid, '----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

fclose(fid);

fprintf('\nHecho. TXT guardado en:\n%s\n', outputTxt);
fprintf('Aciertos: %d | Fallos: %d | Missing: %d | Acc: %.2f%%\n', ...
        nOK, nFail, nMissing, acc);


%% ================================================================
%% CLASIFICAR CARPETA: predecir, guardar a TXT y contar aciertos
%% (GT opcional: se extrae de los primeros dígitos del nombre)
%% Requiere: trainedModel cargado
%% ================================================================

segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_test2_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Mtest_test2.txt';

% S = load("TrainedModelWith_Mtrain.mat");
% trainedModel = S.trainedModel;
% clear S;

% OJO: como ya no usamos M_test, define aquí las 12 variables en el orden
% exacto que espera el modelo (mejor esto que depender de M_test).
predictorNames = { ...
        'Extent','Solidity','V_mean','Eccentricity','SkelLenNorm','Circularity', ...
        'H_mean_circ','S_mean','V_IQR','S_median','FD5','EulerNumber' ...
    };

% --- Listar imágenes reales en la carpeta ---
exts = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.webp'};
files = [];
for e = 1:numel(exts)
    files = [files; dir(fullfile(segFolder, exts{e}))]; %#ok<AGROW>
end
[~, idxSort] = sort({files.name});
files = files(idxSort);

N = numel(files);
if N == 0
    error('No se encontraron imágenes en: %s', segFolder);
end

% --- Abrir TXT ---
fid = fopen(outputTxt,'w');
if fid==-1
    error('No se pudo crear el archivo: %s', outputTxt);
end

fprintf(fid, 'EVALUACIÓN SOBRE CARPETA (modelo M)\n');
fprintf(fid, '==================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

nOK = 0;
nFail = 0;
nMissing = 0;   % aquí debería quedar 0 siempre
nNoGT = 0;

failList = strings(0,1);

for i = 1:N
    imgName = files(i).name;
    imgPath = fullfile(segFolder, imgName);

    % Ground truth opcional desde nombre
    [hasGT, trueLabelStr] = parseGTfromFilename(imgName);
    if ~hasGT
        trueLabelStr = "---";
        nNoGT = nNoGT + 1;
    end

    if ~isfile(imgPath)
        fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=--- | ERROR: NO FILE\n', ...
            i, N, string(imgName), trueLabelStr);
        nMissing = nMissing + 1;
        continue;
    end

    % Leer imagen y recalcular features (pipeline real)
    Ipiece = imread(imgPath);
    feat   = extractColorShapeFeatures(Ipiece); % 1x12 (asegúrate de que devuelve 12!)
    feat   = feat(:).';                         % fila
    featTable = array2table(feat, 'VariableNames', predictorNames);

    predictedLabel = trainedModel.predictFcn(featTable);

    % Comparar solo si hay GT
    if hasGT
        isCorrect = (string(predictedLabel) == string(trueLabelStr));
    else
        isCorrect = true; % no cuenta como fallo si no hay GT
    end

    if hasGT
        if isCorrect
            nOK = nOK + 1;
            resultStr = 'OK';
        else
            nFail = nFail + 1;
            resultStr = 'FAIL';
            failList(end+1,1) = sprintf('%s | REAL=%s | PRED=%s', ...
                string(imgName), trueLabelStr, string(predictedLabel));
        end
    else
        resultStr = 'NO_GT';
    end

    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=%s | %s\n', ...
        i, N, string(imgName), trueLabelStr, string(predictedLabel), resultStr);

    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

% --- Resumen ---
totalEvaluated = nOK + nFail; % solo las que tienen GT y fueron evaluadas
acc = 0;
if totalEvaluated > 0
    acc = 100 * (nOK / totalEvaluated);
end

fprintf(fid, '\n\nRESUMEN\n');
fprintf(fid, '------\n');
fprintf(fid, 'Total imágenes carpeta     : %d\n', N);
fprintf(fid, 'Imágenes no encontradas    : %d\n', nMissing);
fprintf(fid, 'Imágenes sin GT en nombre  : %d\n', nNoGT);
fprintf(fid, 'Evaluadas (con GT)         : %d\n', totalEvaluated);
fprintf(fid, 'Aciertos                   : %d\n', nOK);
fprintf(fid, 'Fallos                     : %d\n', nFail);
fprintf(fid, 'Accuracy (solo con GT)     : %.2f %%\n', acc);

fprintf(fid, '\n\nLISTA DE FALLOS (si los hay)\n');
fprintf(fid, '----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

fclose(fid);

fprintf('\nHecho. TXT guardado en:\n%s\n', outputTxt);
fprintf('Aciertos: %d | Fallos: %d | Missing: %d | Acc: %.2f%%\n', ...
        nOK, nFail, nMissing, acc);

% --- helper: GT desde los primeros dígitos ---
function [hasGT, gtStr] = parseGTfromFilename(fname)
    tok = regexp(fname, '^(\d{1,2})', 'tokens', 'once');
    if isempty(tok)
        hasGT = false;
        gtStr = "";
        return;
    end
    v = str2double(tok{1});
    if ~isfinite(v)
        hasGT = false;
        gtStr = "";
        return;
    end
    hasGT = true;
    gtStr = sprintf('%02d', v); % "3" -> "03"
end


