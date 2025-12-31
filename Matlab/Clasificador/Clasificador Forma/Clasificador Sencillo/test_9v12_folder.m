%% test_9v12_folder.m
clear; clc;

% ---------------------------
% 1) Cargar modelo
% ---------------------------
S = load("tree_9v12_final.mat");

% Intenta encontrar el modelo aunque no se llame "model"
if isfield(S,"model")
    model = S.model;
else
    % coge el primer campo que sea un modelo de clasificación
    f = fieldnames(S);
    model = [];
    for k=1:numel(f)
        if isa(S.(f{k}), "ClassificationTree") || isa(S.(f{k}), "CompactClassificationTree")
            model = S.(f{k});
            break;
        end
    end
    if isempty(model)
        error("No encuentro un ClassificationTree dentro de tree_9v12_final.mat");
    end
end

% Asegura orden fijo de clases
order = categorical(["09","12"]);

% ---------------------------
% 2) Carpeta de test (segmentadas)
% ---------------------------
segFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_test2_local";
validCodes = ["09","12"];

% ---------------------------
% 3) Listar imágenes
% ---------------------------
exts = {'*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.webp'};
files = [];
for e=1:numel(exts)
    files = [files; dir(fullfile(segFolder, exts{e}))]; %#ok<AGROW>
end
if isempty(files), error("No hay imágenes en %s", segFolder); end
[~,ix] = sort({files.name}); files = files(ix);

% ---------------------------
% 4) Filtrar solo 09_ y 12_
% ---------------------------
isValid = false(numel(files),1);
for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    parts = split(string(baseName), "_");
    if ~isempty(parts) && any(parts(1) == validCodes)
        isValid(i) = true;
    end
end
files = files(isValid);

if isempty(files)
    error("No hay archivos que empiecen por 09_ o 12_ en %s", segFolder);
end

fprintf("Imágenes válidas (09/12): %d\n\n", numel(files));

% ---------------------------
% 5) Evaluar
% ---------------------------
opts = struct(); % si quieres tocar umbrales del extractor, aquí

pred = strings(numel(files),1);
gt   = strings(numel(files),1);
hasGT = false(numel(files),1);

for i=1:numel(files)
    imgName = files(i).name;
    imgPath = fullfile(files(i).folder, imgName);
    I = imread(imgPath);

    [lab, post] = predict_9v12_fromImage(I, model, opts);
    pred(i) = lab;

    % GT desde nombre (09_... o 12_...)
    tok = regexp(imgName, '^(\d{2})', 'tokens', 'once');
    if ~isempty(tok) && any(string(tok{1}) == validCodes)
        gt(i) = string(tok{1});
        hasGT(i) = true;
    end

    % post = [P(09) P(12)] en el orden fijo
    fprintf("%-25s -> %s   (p09=%.3f  p12=%.3f)\n", imgName, lab, post(1), post(2));
end

% ---------------------------
% 6) Accuracy + Confusion (si hay GT)
% ---------------------------
if any(hasGT)
    Ytrue = categorical(gt(hasGT), ["09","12"]);
    Ypred = categorical(pred(hasGT), ["09","12"]);

    acc = mean(Ypred == Ytrue);
    C = confusionmat(Ytrue, Ypred, "Order", order);

    fprintf("\nAccuracy carpeta (solo con GT): %.2f %%  (%d imgs)\n", 100*acc, nnz(hasGT));
    disp("Confusion matrix (rows=real, cols=pred) [09 12]:");
    disp(C);
else
    fprintf("\nNo hay GT en nombres. Solo predicciones.\n");
end