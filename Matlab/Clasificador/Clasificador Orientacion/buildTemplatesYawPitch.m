%% buildTemplatesYawPitch_split_allcodes.m
clear; clc;

% ===== CONFIG =====
codes     = ["01","02","03","04","05","06","07","08","09","10","11","12"];
segFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local";
tplOut    = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion\Templates";

anglesStr = ["000","045","090","135","180","225","270","315"]; % yaw
pitchStr  = ["10","40","70","90"];                             % pitch

outSize   = 128;

% Regla deseada:
% N=4 -> 2 train 2 test
% N=5 -> 3 train 2 test
% N=6 -> 4 train 2 test
% En general: si N>=4 -> test=2, train=N-2
% si N==3 -> test=1, train=2
% si N<=2 -> no se crea plantilla (o crea con todo train y 0 test; aquí lo saltamos)
rng(1); % reproducible

if ~isfolder(tplOut), mkdir(tplOut); end

% Listar imágenes una sola vez (más eficiente)
allFiles = [dir(fullfile(segFolder,'*.jpg')); dir(fullfile(segFolder,'*.png')); dir(fullfile(segFolder,'*.jpeg'))];
if isempty(allFiles), error("No hay imágenes en %s", segFolder); end

for c = 1:numel(codes)
    code = codes(c);

    % ===== Filtrar por código una vez =====
    isCode = false(numel(allFiles),1);
    for i = 1:numel(allFiles)
        [~, baseName, ~] = fileparts(allFiles(i).name);
        parts = split(baseName,'_');
        if numel(parts) >= 3 && string(parts{1}) == code
            isCode(i) = true;
        end
    end
    files = allFiles(isCode);

    fprintf("\n====================================\n");
    fprintf("Código %s: %d imágenes\n", code, numel(files));
    fprintf("====================================\n");

    % Listado de test (se guarda)
    testItems = struct('path',{}, 'gtYaw',{}, 'gtPitch',{});

    % ===== LOOP combos yaw/pitch =====
    for a = 1:numel(anglesStr)
        for pp = 1:numel(pitchStr)
            ang = anglesStr(a);
            pit = pitchStr(pp);

            % Filtrar por (yaw,pitch)
            pick = false(numel(files),1);
            for i = 1:numel(files)
                [~, baseName, ~] = fileparts(files(i).name);
                parts = split(baseName,'_');
                if numel(parts) >= 3 && string(parts{2}) == ang && string(parts{3}) == pit
                    pick(i) = true;
                end
            end
            fsub = files(pick);

            N = numel(fsub);
            fprintf("Combo yaw=%s pitch=%s: %d imgs\n", ang, pit, N);

            if N == 0
                fprintf("  -> No hay imágenes. Salto.\n");
                continue;
            end

            % Barajar (para que el split sea aleatorio)
            fsub = fsub(randperm(N));

            % ===== Split según regla =====
            if N >= 4
                nTest = 2;
            elseif N == 3
                nTest = 1;
            else
                % N=1 o 2: no hay base para test + plantilla robusta
                fprintf("  -> AVISO: N=%d (muy pocas). Salto plantilla.\n", N);
                continue;
            end
            nTrain = N - nTest;

            trainFiles = fsub(1:nTrain);
            testFiles  = fsub(nTrain+1:end);

            fprintf("  -> Train: %d  Test: %d\n", numel(trainFiles), numel(testFiles));

            % Guardar test list
            for i = 1:numel(testFiles)
                testItems(end+1).path  = fullfile(testFiles(i).folder, testFiles(i).name); %#ok<SAGROW>
                testItems(end).gtYaw   = str2double(ang);
                testItems(end).gtPitch = str2double(pit);
            end

            % ===== Construir plantilla SOLO con trainFiles =====
            acc = zeros(outSize, outSize);
            nOK = 0;

            for i = 1:numel(trainFiles)
                I = imread(fullfile(trainFiles(i).folder, trainFiles(i).name));
                [J, ok] = normalizeMaskedPatch(I, outSize);
                if ~ok, continue; end
                acc = acc + J;
                nOK = nOK + 1;
            end

            if nOK < 2
                fprintf("  -> AVISO: nTrainOK=%d (poco). Plantilla puede ser mala. Salto.\n", nOK);
                continue;
            end

            T = acc / nOK;
            T = T - mean(T(:));
            T = T / (norm(T(:)) + 1e-12);

            tpl = struct();
            tpl.code        = code;
            tpl.orientation = str2double(ang);
            tpl.pitch       = str2double(pit);
            tpl.size        = outSize;
            tpl.T           = T;

            outFile = fullfile(tplOut, sprintf("tpl_%s_%s_%s.mat", code, ang, pit));
            save(outFile, "tpl", "-v7");

            fprintf("  -> Plantilla guardada (%s) con nTrainOK=%d\n", outFile, nOK);
        end
    end

    % ===== Guardar lista de test del código =====
    testListFile = fullfile(tplOut, sprintf("test_list_%s.mat", code));
    save(testListFile, "testItems", "-v7");
    fprintf("\nGuardado test list: %s (N=%d)\n", testListFile, numel(testItems));
end

fprintf("\nDONE.\n");

%% ===== FUNCIÓN LOCAL =====
function [J, ok] = normalizeMaskedPatch(I, outSize)
    ok = false;
    J  = zeros(outSize,outSize);

    try
        mask = extractMaskLego(I);

        % Limpieza mínima (te evitará máscaras “churras” como la que enseñaste)
        mask = bwareafilt(mask, 1);
        mask = imfill(mask, 'holes');
        mask = imclose(mask, strel('disk', 3));

        if ~any(mask(:)), return; end

        stats = regionprops(mask, 'BoundingBox', 'Area');
        if isempty(stats), return; end
        [~, idx] = max([stats.Area]);
        bb = stats(idx).BoundingBox;

        if size(I,3)==3
            G = im2double(rgb2gray(I));
        else
            G = im2double(I);
        end

        x = floor(bb(1)); y = floor(bb(2));
        w = ceil(bb(3));  h = ceil(bb(4));
        x = max(x,1); y = max(y,1);
        x2 = min(x+w-1, size(G,2));
        y2 = min(y+h-1, size(G,1));

        Gc = G(y:y2, x:x2);
        Mc = mask(y:y2, x:x2);
        Gc(~Mc) = 0;

        [H,W] = size(Gc);
        S = max(H,W);
        padY = floor((S-H)/2);
        padX = floor((S-W)/2);
        Gp = padarray(Gc, [padY padX], 0, 'both');
        Gp = padarray(Gp, [S-size(Gp,1) S-size(Gp,2)], 0, 'post');

        Jr = imresize(Gp, [outSize outSize], 'bilinear');
        Jr = Jr - mean(Jr(:));
        nrm = norm(Jr(:));
        if nrm < 1e-9, return; end
        J = Jr / nrm;

        ok = true;
    catch
        ok = false;
    end
end
