%% ================================================================
%% CLASIFICAR CARPETA: base (modelo M 12 feats) + cascada amarillo (9-12)
%% ================================================================
clear; clc;

segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Mtest_dobleClassificador_6caracForma.txt';

validCodes = {'01','02','03','04','05','06','07','08','09','10','11','12'};

% --- Listar imágenes SOLO en la carpeta (jpg/png) y filtrar por 09/12 ---
files = [dir(fullfile(segFolder,'*.jpg')); dir(fullfile(segFolder,'*.png')); dir(fullfile(segFolder,'*.jpeg'))];
fprintf('Se han encontrado %d archivos de imagen en %s\n', numel(files), segFolder);

isValid = false(numel(files),1);
for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    partes = split(baseName, '_');
    if ~isempty(partes) && ismember(partes{1}, validCodes)
        isValid(i) = true;
    end
end
files = files(isValid);
[~,ix] = sort({files.name}); files = files(ix);
fprintf('Tras filtrar por código se han encontrado %d archivos de imagen en %s\n', numel(files), segFolder);

N = numel(files);
if N == 0
    error('No hay imágenes 09/12 en: %s', segFolder);
end

%--- Cargar modelo M (12 features) ---
S = load("TrainedModelWith_Mtrain_12.mat");
trainedModel = S.trainedModel;
clear S;

% --- Cargar refinador amarillo 9-12 (árbol) ---
S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\Clasificador Sencillo\tree_9v12_final.mat");
if ~isfield(S,"model")
    error("El .mat no contiene la variable 'model'.");
end
model_912 = S.model;          % struct con fields: tree, featNames
tree_912  = model_912.tree;   % ClassificationTree
featNames_shape = model_912.featNames;  % nombres EXACTOS esperados por el árbol
clear S;

% === RUTAS ORIENTACIÓN ===
orientRoot = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion";
templatesFolder = fullfile(orientRoot, "Templates_local");
addpath(genpath(orientRoot));

% --- PredictorNames del clasificador gordo (NO TOCAR) ---
predictorNames = { ...
    'Extent','Solidity','V_mean','Eccentricity','SkelLenNorm','Circularity', ...
    'H_mean_circ','S_mean','V_IQR','S_median','FD5','EulerNumber' ...
};

% --- Abrir TXT ---
fid = fopen(outputTxt,'w');
if fid==-1, error('No se pudo crear: %s', outputTxt); end

fprintf(fid, 'EVALUACIÓN SOBRE CARPETA (modelo M + cascada 9-12)\n');
fprintf(fid, '==================================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

nOK = 0; nFail = 0; nNoGT = 0;
nRef912 = 0; nFlip912 = 0;

failList   = strings(0,1);
changeList = strings(0,1);

% --- umbrales conservadores del refinador ---
TH_PMAX   = 0.70;
TH_MARGIN = 0.20;

for i = 1:N
    imgName = files(i).name;
    imgPath = fullfile(files(i).folder, imgName);

    [hasGT, trueLabelStr] = parseGTfromFilename(imgName);
    if ~hasGT
        trueLabelStr = "---";
        nNoGT = nNoGT + 1;
    end

    Ipiece = imread(imgPath);

    % =========================
    % 1) BASE: Modelo M (12 feats)  (NO TOCAR)
    % =========================
    feat12   = extractColorShapeFeatures(Ipiece);
    feat12   = feat12(:).';
    Xbase    = array2table(feat12, 'VariableNames', predictorNames);

    predictedLabel_base  = trainedModel.predictFcn(Xbase);
    predictedLabel_final = predictedLabel_base;
    refinador = "none";

    % =========================
    % 2) CASCADA AMARILLO 9-12 (árbol)
    % =========================
    pb_num = str2double(string(predictedLabel_base));

%     if isfinite(pb_num) && (pb_num==9 || pb_num==12)
%         nRef912 = nRef912 + 1;
% 
%         feat6 = extractShapeFeaturess(Ipiece);   % <- tu extractor de 6 feats
%         feat6 = feat6(:).';
% 
%         Xref = array2table(feat6, 'VariableNames', featNames_shape);
% 
%         [predictedLabel_ref, score912] = predict(tree_912, Xref);
%         predictedLabel_ref = string(predictedLabel_ref);
% 
%         pSort  = sort(score912,'descend');
%         pMax   = pSort(1);
%         margin = pSort(1) - pSort(2);
% 
%         if (pMax >= TH_PMAX) && (margin >= TH_MARGIN)
%             predictedLabel_final = predictedLabel_ref;
%             refinador = "9-12";
%         end
% 
%         if string(predictedLabel_final) ~= string(predictedLabel_base)
%             nFlip912 = nFlip912 + 1;
%             changeList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s -> FINAL=%s | REF=%s | pMax=%.3f margin=%.3f', ...
%                 string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador, pMax, margin);
%         end
%         
%     end

    % =========================
    % 3) ORIENTACIÓN (según código final)
    % =========================
    codeFinal = string(predictedLabel_final);
    [yaw, pitch, oScore, oGap] = predictOrientationFromTemplates(Ipiece, codeFinal, templatesFolder);

    % =========================
    % 4) Comparar
    % =========================
    if hasGT
        isCorrect = (string(predictedLabel_final) == string(trueLabelStr));
    else
        isCorrect = true;
    end

    if hasGT
        if isCorrect
            nOK = nOK + 1;
            resultStr = 'OK';
        else
            nFail = nFail + 1;
            resultStr = 'FAIL';
            failList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s | FINAL=%s | REF=%s', ...
                string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador);
        end
    else
        resultStr = 'NO_GT';
    end

    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED_BASE=%s | PRED_FINAL=%s | REF=%s | ORI=%03d/%02d (s=%.3f g=%.3f) | %s\n', ...
        i, N, string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador, ...
        yaw, pitch, oScore, oGap, resultStr);

    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

totalEvaluated = nOK + nFail;
acc = 0;
if totalEvaluated > 0
    acc = 100*(nOK/totalEvaluated);
end

fprintf(fid, '\n\nRESUMEN\n------\n');
fprintf(fid, 'Total imágenes carpeta     : %d\n', N);
fprintf(fid, 'Imágenes sin GT en nombre  : %d\n', nNoGT);
fprintf(fid, 'Evaluadas (con GT)         : %d\n', totalEvaluated);
fprintf(fid, 'Aciertos                   : %d\n', nOK);
fprintf(fid, 'Fallos                     : %d\n', nFail);
fprintf(fid, 'Accuracy (solo con GT)     : %.2f %%\n', acc);

fprintf(fid, '\nUSO REFINADOR 9-12\n-----------------\n');
fprintf(fid, 'Ref 9-12 usado            : %d\n', nRef912);
fprintf(fid, 'Cambios BASE->FINAL (flip): %d\n', nFlip912);

fprintf(fid, '\n\nLISTA DE FALLOS\n----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList), fprintf(fid, '%s\n', failList(k)); end
end

fprintf(fid, '\n\nLISTA DE CAMBIOS (BASE -> FINAL)\n--------------------------------\n');
if isempty(changeList)
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(changeList), fprintf(fid, '%s\n', changeList(k)); end
end

fclose(fid);
fprintf('\nHecho. TXT: %s\n', outputTxt);
fprintf('Aciertos: %d | Fallos: %d | Acc: %.2f%% | Ref9-12 usado: %d | flips: %d\n', ...
    nOK, nFail, acc, nRef912, nFlip912);

function [hasGT, gtStr] = parseGTfromFilename(fname)
    tok = regexp(fname, '^(\d{1,2})', 'tokens', 'once');
    if isempty(tok)
        hasGT = false; gtStr = ""; return;
    end
    v = str2double(tok{1});
    if ~isfinite(v)
        hasGT = false; gtStr = ""; return;
    end
    hasGT = true;
    gtStr = sprintf('%02d', v);
end

function [yaw, pitch, bestScore, gap] = predictOrientationFromTemplates(Ipiece, codeStr, templatesFolder)
% Devuelve yaw/pitch usando templates del código codeStr ("01".."12")
% Cachea las templates por código para no recargar en cada imagen.

    persistent cache
    if isempty(cache)
        cache = containers.Map('KeyType','char','ValueType','any');
    end

    codeKey = char(codeStr);

    % --- Cargar del cache o desde disco ---
    if isKey(cache, codeKey)
        templates = cache(codeKey);
    else
        anglesStr = ["000","045","090","135","180","225","270","315"];
        pitchStr  = ["10","40","70","90"];

        templatesCell = {};
        for a = 1:numel(anglesStr)
            for p = 1:numel(pitchStr)
                ang = anglesStr(a);
                pit = pitchStr(p);

                f = fullfile(templatesFolder, sprintf("tpl_%s_%s_%s.mat", codeKey, ang, pit));
                if ~isfile(f), continue; end
                S = load(f);
                if ~isfield(S,"tpl"), continue; end
                templatesCell{end+1} = S.tpl; %#ok<AGROW>
            end
        end

        if isempty(templatesCell)
            yaw = NaN; pitch = NaN; bestScore = NaN; gap = NaN;
            cache(codeKey) = []; % para no insistir
            return;
        end

        % Unificar fields (evita el “dissimilar structures”)
        allFields = {};
        for k = 1:numel(templatesCell)
            allFields = union(allFields, fieldnames(templatesCell{k}));
        end
        for k = 1:numel(templatesCell)
            for ff = 1:numel(allFields)
                fn = allFields{ff};
                if ~isfield(templatesCell{k}, fn)
                    templatesCell{k}.(fn) = [];
                end
            end
        end

        templates = [templatesCell{:}];
        cache(codeKey) = templates;
    end

    if isempty(templates)
        yaw = NaN; pitch = NaN; bestScore = NaN; gap = NaN;
        return;
    end

    % --- Predicción ---
    [yaw, pitch, scores, gap] = predictYawPitch_byTemplate(Ipiece, templates);
    bestScore = max(scores);
end

