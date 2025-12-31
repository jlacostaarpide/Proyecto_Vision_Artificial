%% ================================================================
%% CLASIFICAR CARPETA: base (modelo M 12 feats) + cascada amarillo (9-12)
%% (GT opcional: se extrae de los primeros dígitos del nombre)
%% Requiere: trainedModel (modelo M) cargado + model_912 cargado
%% ================================================================

segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_test3_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Mtest3_dobleClassificador.txt';

%--- Cargar modelo M (12 features) ---
S = load("TrainedModelWith_Mtrain_12.mat");
trainedModel = S.trainedModel;
clear S;

% --- Cargar refinador amarillo 9-12 (24 features de forma) ---
S = load("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Forma\TrainedModelWith_Ftrain_9_12_yellow_24carac.mat");
model_912 = S.trainedModel;
clear S;

% === RUTAS ORIENTACIÓN ===
orientRoot = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion";
templatesFolder = fullfile(orientRoot, "Templates_local");
addpath(genpath(orientRoot)); % para predictYawPitch_byTemplate, extractMaskLego, etc.

% OJO: como ya no usamos M_test, define aquí las 12 variables en el orden
% exacto que espera el modelo M (mejor esto que depender de M_test).
predictorNames = { ...
    'Extent','Solidity','V_mean','Eccentricity','SkelLenNorm','Circularity', ...
    'H_mean_circ','S_mean','V_IQR','S_median','FD5','EulerNumber' ...
};

% --- Variables EXACTAS para el refinador 9-12 (24 shape feats) ---
featNames_shape = { ...
 'AreaNorm','PerimNorm','Circularity','Extent','Solidity','Eccentricity','AspectRatio','EulerNumber', ...
 'HolesCount','HolesAreaFrac','SkelLenNorm','SkelEndpoints','SkelBranchpoints', ...
 'ProjV_peaks','ProjH_peaks','ProjV_entropy','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3', ...
 'StudsCount','StudsCountNormArea','StudsMeanRadius','StudsRadiusStd' ...
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

fprintf(fid, 'EVALUACIÓN SOBRE CARPETA (modelo M + cascada 9-12)\n');
fprintf(fid, '==================================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

nOK = 0;
nFail = 0;
nMissing = 0;   % aquí debería quedar 0 siempre
nNoGT = 0;

% contadores del refinador 9-12
nRef912  = 0;
nFlip912 = 0;

failList   = strings(0,1);
changeList = strings(0,1);

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

    % Leer imagen
    Ipiece = imread(imgPath);

    % =========================
    % 1) BASE: Modelo M (12 feats)
    % =========================
    feat12   = extractColorShapeFeatures(Ipiece); % 1x12 (tu función)
    feat12   = feat12(:).';                        % fila
    Xbase    = array2table(feat12, 'VariableNames', predictorNames);

    predictedLabel_base  = trainedModel.predictFcn(Xbase);
    predictedLabel_final = predictedLabel_base;
    refinador = "none";

   
    % =========================
    % 2) CASCADA AMARILLO 9-12 (si aplica)
    % =========================
    pb_num = str2double(string(predictedLabel_base));  % "09"->9, "12"->12

    if isfinite(pb_num) && (pb_num==9 || pb_num==12)
        nRef912 = nRef912 + 1;

        feat24 = extractShapeFeatures(Ipiece);   % 1x14
        feat24 = feat24(:).';                    % fila
        Xref   = array2table(feat24, 'VariableNames', featNames_shape);

        predictedLabel_ref = model_912.predictFcn(Xref);

        predictedLabel_final = predictedLabel_ref;
        refinador = "9-12";

        if string(predictedLabel_final) ~= string(predictedLabel_base)
            nFlip912 = nFlip912 + 1;
            changeList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s -> FINAL=%s | REF=%s', ...
                string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador);
        end
    end

    % =========================
    % 3) ORIENTACIÓN (según código final)
    % =========================
    codeFinal = string(predictedLabel_final); % "01".."12"
    [yaw, pitch, oScore, oGap] = predictOrientationFromTemplates(Ipiece, codeFinal, templatesFolder);


    % =========================
    % 4) Comparar (solo si hay GT)
    % =========================
    if hasGT
        isCorrect = (string(predictedLabel_final) == string(trueLabelStr));
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
            failList(end+1,1) = sprintf('%s | REAL=%s | BASE=%s | FINAL=%s | REF=%s', ...
                string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador);
        end
    else
        resultStr = 'NO_GT';
    end

    % =========================
    % 5) Log por imagen (CON ORIENTACIÓN)
    % =========================
    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED_BASE=%s | PRED_FINAL=%s | REF=%s | ORI=%03d/%02d (s=%.3f g=%.3f) | %s\n', ...
        i, N, string(imgName), string(trueLabelStr), string(predictedLabel_base), string(predictedLabel_final), refinador, ...
        yaw, pitch, oScore, oGap, resultStr);

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

fprintf(fid, '\nUSO REFINADOR 9-12\n');
fprintf(fid, '-----------------\n');
fprintf(fid, 'Ref 9-12 usado            : %d\n', nRef912);
fprintf(fid, 'Cambios BASE->FINAL (flip): %d\n', nFlip912);

fprintf(fid, '\n\nLISTA DE FALLOS (si los hay)\n');
fprintf(fid, '----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

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
fprintf('Aciertos: %d | Fallos: %d | Acc: %.2f%% | Ref9-12 usado: %d | flips: %d\n', ...
    nOK, nFail, acc, nRef912, nFlip912);

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

