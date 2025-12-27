%% run_orient_01_testFolder.m
clear; clc;

%% 1) Carpeta plantillas
tplFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion";

%% 2) Código y ángulos
code   = "01";
anglesStr = ["000","045","090","135","180","225","270","315"];
anglesNum = str2double(anglesStr);

%% 3) Cargar plantillas (cada .mat contiene templateXX_YYY)
templates = cell(numel(anglesStr),1);

for k = 1:numel(anglesStr)
    a = anglesStr(k);
    tplFile = fullfile(tplFolder, sprintf("template%s_%s.mat", code, a));
    if ~isfile(tplFile), error("No existe: %s", tplFile); end

    S = load(tplFile);
    varName = sprintf("template%s_%s", code, a);

    if ~isfield(S, varName)
        error("Dentro de %s no está la variable %s", tplFile, varName);
    end

    templates{k} = S.(varName);  % debe tener .mask
end

fprintf("Cargadas %d plantillas para pieza %s.\n\n", numel(templates), code);

%% 4) Carpeta test (segmentadas)
testFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local";

files = [dir(fullfile(testFolder,'*.jpg')); dir(fullfile(testFolder,'*.png')); dir(fullfile(testFolder,'*.jpeg'))];
if isempty(files), error("No hay imágenes en %s", testFolder); end

%% 5) Filtrar por código 01_...
isCode = false(numel(files),1);
for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    parts = split(baseName,'_');
    if numel(parts) >= 2 && string(parts{1}) == code
        isCode(i) = true;
    end
end
files = files(isCode);

fprintf("Probando %d imágenes del código %s...\n\n", numel(files), code);

%% 6) Métricas
nTotal = numel(files);
nEval = 0; nOK = 0; nFAIL = 0; nNoGT = 0;

for i = 1:numel(files)
    fname = files(i).name;
    imgPath = fullfile(files(i).folder, fname);
    I = imread(imgPath);

    % GT desde nombre: 01_XXX_...
    [~, baseName, ~] = fileparts(fname);
    parts = split(baseName,'_');

    if numel(parts) < 2
        gt = NaN;
        nNoGT = nNoGT + 1;
    else
        gt = str2double(string(parts{2})); % "000"->0
    end

    [pred, scores] = predictOrientation45(I, templates, anglesNum);

    % confianza: gap entre mejor y segundo
    scoresSorted = sort(scores,'descend');
    if numel(scoresSorted) >= 2
        gap = scoresSorted(1) - scoresSorted(2);
    else
        gap = NaN;
    end

    if ~isnan(gt)
        nEval = nEval + 1;
        ok = (pred == gt);
        if ok, nOK = nOK + 1; else, nFAIL = nFAIL + 1; end

        tag = "OK"; if ~ok, tag = "FAIL"; end
        fprintf("%-28s GT:%03dº -> Pred:%03dº (IoU=%.3f, gap=%.3f)  [%s]\n", ...
            fname, gt, pred, max(scores), gap, tag);
    else
        fprintf("%-28s GT:???  -> Pred:%03dº (IoU=%.3f, gap=%.3f)\n", ...
            fname, pred, max(scores), gap);
    end
end

acc = 100 * (nOK / max(nEval,1));

fprintf("\n==================== RESUMEN ====================\n");
fprintf("Total imágenes (código %s): %d\n", code, nTotal);
fprintf("Evaluadas (con GT):         %d\n", nEval);
fprintf("Aciertos:                   %d\n", nOK);
fprintf("Fallos:                     %d\n", nFAIL);
fprintf("Sin GT (nombre raro):       %d\n", nNoGT);
fprintf("Accuracy:                   %.2f %%\n", acc);
fprintf("=================================================\n");
