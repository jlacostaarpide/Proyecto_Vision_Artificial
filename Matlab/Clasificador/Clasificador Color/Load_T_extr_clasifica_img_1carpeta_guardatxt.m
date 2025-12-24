clear all;close all;clc;
load('legoFeatures_TRAIN_todas_8carac.mat'); % T todas 



%% Seleccionar de forma aleatoria las Train y las Test
%load('I_random.mat');
%I=randperm(1873);
T_train=T(I(1:1500),:);
T_test=T(I(1501:end),:);

% L_train = T.Label(I(1:1500),:);
% L_test = T.Label(I(1501:end),:);

% Quiero entrenar ahora con las train en classLearner. SVM funciona muy bien. Hacer crossvalidation y si quiero test tb, en el menu de new session. Y luego clasificar
% con las test con el predict.


%% ================================================================
%% EVALUAR T_test: base (color) + refinadores (forma) con variables distintas
%% + guardar en TXT: lista de FALLOS y lista de CAMBIOS (base->final)
%% ================================================================

segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Ttest_dobleClasificador.txt';

% --- Cargar modelos sin pisarlos ---
S = load("TrainedModelWith_Ttrain.mat");
model_base = S.trainedModel;

S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\TrainedModelWith_Ftrain_3_6_red.mat");
model_36 = S.trainedModel;

S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\TrainedModelWith_Ftrain_9_12_yellow.mat");
model_912 = S.trainedModel;

clear S;

% --- Nombres EXACTOS de variables por modelo ---
featNames_color = {'H_mean_circ','H_var_circ', ...
                   'S_median','S_IQR', ...
                   'V_median','V_IQR', ...
                   'S_mean','V_mean'};

featNames_shape = { ...
 'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
 'SkelLenNorm','SkelEndpoints','SkelBranchpoints', ...
 'FD2','FD3','FD4','FD5'};

% --- Abrir TXT ---
fid = fopen(outputTxt,'w');
if fid==-1
    error('No se pudo crear el archivo: %s', outputTxt);
end

fprintf(fid, 'EVALUACIÓN COMPLETA SOBRE T_test (CASCADA: COLOR->FORMA)\n');
fprintf(fid, '=======================================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

N = height(T_test);
nOK = 0; nFail = 0; nMissing = 0;

% Listas para resumen
failList   = strings(0,1);
changeList = strings(0,1);

% Contadores de uso/cambios del refinador
nRef36 = 0;  nFlip36 = 0;
nRef912 = 0; nFlip912 = 0;

for i = 1:N
    imgName   = T_test.FileName{i};
    trueLabel = T_test.Label(i);

    imgPath = fullfile(segFolder, imgName);

    if ~isfile(imgPath)
        fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=--- | ERROR: NO FILE\n', ...
            i, N, imgName, string(trueLabel));
        nMissing = nMissing + 1;
        continue;
    end

    % Leer imagen
    Ipiece = imread(imgPath);

    % 1) Predicción BASE (COLOR)
    featC = extractColorFeatures(Ipiece); % 1x8
    Xbase = array2table(featC, 'VariableNames', featNames_color);

    pred_base  = model_base.predictFcn(Xbase);
    pred_final = pred_base;
    refinador  = "none";

    % 2) Refinado condicional (FORMA)
    pb_num = str2double(string(pred_base)); % "03"->3, etc.

    if isfinite(pb_num) && (pb_num==3 || pb_num==6)
        nRef36 = nRef36 + 1;

        featS = extractShapeFeatures(Ipiece);  % 1x14
        featS = featS(:).';                    % asegurar fila
        Xref  = array2table(featS, 'VariableNames', featNames_shape);

        pred_ref = model_36.predictFcn(Xref);
        pred_final = pred_ref;
        refinador = "3-6";

        if string(pred_ref) ~= string(pred_base)
            nFlip36 = nFlip36 + 1;
        end

    elseif isfinite(pb_num) && (pb_num==9 || pb_num==12)
        nRef912 = nRef912 + 1;

        featS = extractShapeFeatures(Ipiece);  % 1x14
        featS = featS(:).';
        Xref  = array2table(featS, 'VariableNames', featNames_shape);

        pred_ref = model_912.predictFcn(Xref);
        pred_final = pred_ref;
        refinador = "9-12";

        if string(pred_ref) ~= string(pred_base)
            nFlip912 = nFlip912 + 1;
        end
    end

    % 3) Comparar contra etiqueta real (robusto a tipos)
    isCorrect = (string(pred_final) == string(trueLabel));

    if isCorrect
        nOK = nOK + 1;
        resultStr = "OK";
    else
        nFail = nFail + 1;
        resultStr = "FAIL";
        failList(end+1,1) = sprintf('%s | REAL=%s | PRED_BASE=%s | PRED_FINAL=%s | REF=%s', ...
            imgName, string(trueLabel), string(pred_base), string(pred_final), refinador);
    end

    % 4) Guardar cambios (si el refinador cambió la clase)
    if string(pred_final) ~= string(pred_base)
        changeList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s -> FINAL=%s | REF=%s | %s', ...
            imgName, string(trueLabel), string(pred_base), string(pred_final), refinador, resultStr);
    end

    % 5) Guardar línea en TXT
    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED_BASE=%s | PRED_FINAL=%s | REF=%s | %s\n', ...
        i, N, imgName, string(trueLabel), string(pred_base), string(pred_final), refinador, resultStr);

    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

% --- Resumen ---
totalEvaluated = nOK + nFail;
acc = 0;
if totalEvaluated > 0
    acc = 100 * (nOK / totalEvaluated);
end

fprintf(fid, '\n\nRESUMEN\n');
fprintf(fid, '------\n');
fprintf(fid, 'Total filas T_test        : %d\n', N);
fprintf(fid, 'Imágenes no encontradas   : %d\n', nMissing);
fprintf(fid, 'Evaluadas (con archivo)   : %d\n', totalEvaluated);
fprintf(fid, 'Aciertos                  : %d\n', nOK);
fprintf(fid, 'Fallos                    : %d\n', nFail);
fprintf(fid, 'Accuracy (sin missing)    : %.2f %%\n', acc);

fprintf(fid, '\nUSO DE REFINADORES\n');
fprintf(fid, '-----------------\n');
fprintf(fid, 'Ref 3-6  usado: %d | cambió decisión base: %d\n', nRef36, nFlip36);
fprintf(fid, 'Ref 9-12 usado: %d | cambió decisión base: %d\n', nRef912, nFlip912);

% --- Lista de fallos ---
fprintf(fid, '\n\nLISTA DE FALLOS (si los hay)\n');
fprintf(fid, '----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

% --- Lista de cambios ---
fprintf(fid, '\n\nLISTA DE CAMBIOS (BASE -> FINAL)\n');
fprintf(fid, '--------------------------------\n');
if isempty(changeList)
    fprintf(fid, 'Ninguno (el refinador nunca cambió la clase).\n');
else
    for k = 1:numel(changeList)
        fprintf(fid, '%s\n', changeList(k));
    end
end

fclose(fid);

fprintf('\nHecho. TXT guardado en:\n%s\n', outputTxt);
fprintf('Aciertos: %d | Fallos: %d | Missing: %d | Acc: %.2f%%\n', ...
        nOK, nFail, nMissing, acc);
fprintf('Cambios 3-6: %d/%d | Cambios 9-12: %d/%d\n', nFlip36, nRef36, nFlip912, nRef912);


%% ================================================================
%% CLASIFICAR carpeta "segFolder" con imágenes de cualquier nombre
%% (sin T_test): base (color) + refinadores (forma)
%% + TXT con: todas, fallos (si hay GT) y cambios (base->final)
%% + RESUMEN: aciertos, fallos, cambios por refinador y accuracy total
%% ================================================================

segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_test2_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_test2_dobleClasificador.txt';

% --- Cargar modelos sin pisarlos ---
S = load("TrainedModelWith_Ttrain.mat");
model_base = S.trainedModel;

S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\TrainedModelWith_Ftrain_3_6_red.mat");
model_36 = S.trainedModel;

S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\TrainedModelWith_Ftrain_9_12_yellow.mat");
model_912 = S.trainedModel;

clear S;

% --- Nombres EXACTOS de variables por modelo ---
featNames_color = {'H_mean_circ','H_var_circ', ...
                   'S_median','S_IQR', ...
                   'V_median','V_IQR', ...
                   'S_mean','V_mean'};

featNames_shape = { ...
 'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
 'SkelLenNorm','SkelEndpoints','SkelBranchpoints', ...
 'FD2','FD3','FD4','FD5'};

% --- Listar imágenes (cualquier nombre) ---
exts = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.webp'};
files = [];
for e = 1:numel(exts)
    files = [files; dir(fullfile(segFolder, exts{e}))]; %#ok<AGROW>
end

% (Opcional) ordenar por nombre
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

fprintf(fid, 'CLASIFICACIÓN EN CARPETA (CASCADA: COLOR->FORMA)\n');
fprintf(fid, '================================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n', segFolder);
fprintf(fid, 'Total imágenes: %d\n\n', N);

% Contadores (GT)
nOK = 0;            % NUEVO: aciertos (solo con GT)
nFail = 0;          % solo con GT
nHasGT = 0;
nNoGT  = 0;

% NUEVO: métricas para ver si el refinador ayuda o empeora
nBaseOK = 0;        % base acertaba (solo con GT)
nBaseFail = 0;      % base fallaba (solo con GT)
nImprove = 0;       % base fallaba -> final acierta
nWorsen  = 0;       % base acertaba -> final falla
nChangeTotal = 0;   % cuántas veces base != final (incluye NO_GT)

failList   = strings(0,1);
changeList = strings(0,1);

nRef36 = 0;  nFlip36 = 0;
nRef912 = 0; nFlip912 = 0;

for i = 1:N
    imgName = files(i).name;
    imgPath = fullfile(segFolder, imgName);

    % --- Leer imagen ---
    try
        Ipiece = imread(imgPath);
    catch ME
        fprintf(fid, '[%4d/%4d] %s | ERROR: imread (%s)\n', i, N, imgName, ME.message);
        continue;
    end

    % --- Ground truth opcional: si el nombre empieza por "03_" o "03-" o "03 " etc. ---
    [hasGT, trueLabelStr] = parseGTfromFilename(imgName);
    if hasGT
        nHasGT = nHasGT + 1;
    else
        nNoGT = nNoGT + 1;
        trueLabelStr = "---";
    end

    % 1) Predicción BASE (COLOR)
    featC = extractColorFeatures(Ipiece); % 1x8
    Xbase = array2table(featC, 'VariableNames', featNames_color);

    pred_base  = model_base.predictFcn(Xbase);
    pred_final = pred_base;
    refinador  = "none";

    % 2) Refinado condicional (FORMA)
    pb_num = str2double(string(pred_base)); % "03"->3, etc.

    if isfinite(pb_num) && (pb_num==3 || pb_num==6)
        nRef36 = nRef36 + 1;

        featS = extractShapeFeatures(Ipiece);  % 1x14
        featS = featS(:).';
        Xref  = array2table(featS, 'VariableNames', featNames_shape);

        pred_ref  = model_36.predictFcn(Xref);
        pred_final = pred_ref;
        refinador  = "3-6";

        if string(pred_ref) ~= string(pred_base)
            nFlip36 = nFlip36 + 1;
        end

    elseif isfinite(pb_num) && (pb_num==9 || pb_num==12)
        nRef912 = nRef912 + 1;

        featS = extractShapeFeatures(Ipiece);  % 1x14
        featS = featS(:).';
        Xref  = array2table(featS, 'VariableNames', featNames_shape);

        pred_ref  = model_912.predictFcn(Xref);
        pred_final = pred_ref;
        refinador  = "9-12";

        if string(pred_ref) ~= string(pred_base)
            nFlip912 = nFlip912 + 1;
        end
    end

    % NUEVO: contar cambios globales (aunque no haya GT)
    if string(pred_final) ~= string(pred_base)
        nChangeTotal = nChangeTotal + 1;
    end

    % 3) Evaluación (solo si hay GT)
    if hasGT
        baseCorrect  = (string(pred_base)  == string(trueLabelStr));
        finalCorrect = (string(pred_final) == string(trueLabelStr));

        % NUEVO: contadores base
        if baseCorrect
            nBaseOK = nBaseOK + 1;
        else
            nBaseFail = nBaseFail + 1;
        end

        % NUEVO: impacto del refinador
        if (~baseCorrect) && finalCorrect
            nImprove = nImprove + 1;   % arregló
        elseif baseCorrect && (~finalCorrect)
            nWorsen = nWorsen + 1;     % estropeó
        end

        % OK/FAIL final
        if finalCorrect
            nOK = nOK + 1;
            resultStr = "OK";
        else
            nFail = nFail + 1;
            resultStr = "FAIL";
            failList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s | FINAL=%s | REF=%s', ...
                imgName, trueLabelStr, string(pred_base), string(pred_final), refinador);
        end
    else
        resultStr = "NO_GT";
    end

    % 4) Guardar cambios (si el refinador cambió la clase)
    if string(pred_final) ~= string(pred_base)
        changeList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s -> FINAL=%s | REF=%s | %s', ...
            imgName, trueLabelStr, string(pred_base), string(pred_final), refinador, resultStr);
    end

    % 5) Línea por imagen
    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED_BASE=%s | PRED_FINAL=%s | REF=%s | %s\n', ...
        i, N, imgName, trueLabelStr, string(pred_base), string(pred_final), refinador, resultStr);

    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

% --- Accuracy total (solo con GT) ---
acc = 0;
if nHasGT > 0
    acc = 100 * (nOK / nHasGT);
end

% --- Resumen ---
fprintf(fid, '\n\nRESUMEN\n');
fprintf(fid, '------\n');
fprintf(fid, 'Total imágenes            : %d\n', N);
fprintf(fid, 'Con GT en nombre          : %d\n', nHasGT);
fprintf(fid, 'Sin GT en nombre          : %d\n', nNoGT);
fprintf(fid, 'Aciertos (FINAL, con GT)  : %d\n', nOK);
fprintf(fid, 'Fallos  (FINAL, con GT)   : %d\n', nFail);
fprintf(fid, 'Accuracy (FINAL, con GT)  : %.2f %%\n', acc);

% NUEVO: resumen extra para validar el segundo clasificador
fprintf(fid, '\nIMPACTO DEL REFINADOR (solo con GT)\n');
fprintf(fid, '----------------------------------\n');
fprintf(fid, 'Base acertaba (solo color)        : %d\n', nBaseOK);
fprintf(fid, 'Base fallaba  (solo color)        : %d\n', nBaseFail);
fprintf(fid, 'Mejoras (base FAIL -> final OK)   : %d\n', nImprove);
fprintf(fid, 'Empeoras (base OK -> final FAIL)  : %d\n', nWorsen);
fprintf(fid, 'Cambios totales (base != final)   : %d\n', nChangeTotal);

fprintf(fid, '\nUSO DE REFINADORES\n');
fprintf(fid, '-----------------\n');
fprintf(fid, 'Ref 3-6  usado: %d | cambió decisión base: %d\n', nRef36, nFlip36);
fprintf(fid, 'Ref 9-12 usado: %d | cambió decisión base: %d\n', nRef912, nFlip912);

% --- Lista de fallos ---
fprintf(fid, '\n\nLISTA DE FALLOS (si hay GT)\n');
fprintf(fid, '---------------------------\n');
if isempty(failList)
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

% --- Lista de cambios ---
fprintf(fid, '\n\nLISTA DE CAMBIOS (BASE -> FINAL)\n');
fprintf(fid, '--------------------------------\n');
if isempty(changeList)
    fprintf(fid, 'Ninguno (el refinador nunca cambió la clase).\n');
else
    for k = 1:numel(changeList)
        fprintf(fid, '%s\n', changeList(k));
    end
end

fclose(fid);

% --- Mostrar también en consola (lo que pedías) ---
fprintf('\nHecho. TXT guardado en:\n%s\n', outputTxt);
fprintf('GT=%d | OK=%d | FAIL=%d | ACC=%.2f%%\n', nHasGT, nOK, nFail, acc);
fprintf('Cambios totales (base!=final): %d\n', nChangeTotal);
fprintf('Ref 3-6: usado=%d, flips=%d | Ref 9-12: usado=%d, flips=%d\n', nRef36, nFlip36, nRef912, nFlip912);
fprintf('Mejoras: %d | Empeoras: %d\n', nImprove, nWorsen);

% ================================================================
% Función auxiliar: intenta extraer GT del nombre de archivo
% ================================================================
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
    gtStr = sprintf('%02d', v);
end

