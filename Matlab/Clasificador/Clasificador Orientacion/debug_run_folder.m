%% debug_run_folder.m
clear; clc;

% ===== CONFIG =====
code = "09";

tplFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion\Templates_local";
segFolder = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED";

anglesStr = ["000","045","090","135","180","225","270","315"];
pitchStr  = ["10","40","70","90"];

% ===== UMBRALES PARA MARCAR "RARO" =====
TH_SCORE_LOW = 0.75;   % si max(score) < esto, mala coincidencia general
TH_GAP_LOW   = 0.05;   % si gap < esto, casi empate -> inestable

% Carpeta debug de salida
dbgOut = fullfile(tplFolder, sprintf("DBG_%s", code));
if ~isfolder(dbgOut), mkdir(dbgOut); end

%% ===== CARGAR PLANTILLAS =====
templatesCell = {};
for a = 1:numel(anglesStr)
    for p = 1:numel(pitchStr)
        ang = anglesStr(a);
        pit = pitchStr(p);

        f = fullfile(tplFolder, sprintf("tpl_%s_%s_%s.mat", code, ang, pit));
        if ~isfile(f), continue; end
        S = load(f);
        if ~isfield(S,"tpl"), continue; end
        templatesCell{end+1} = S.tpl; %#ok<SAGROW>
    end
end
if isempty(templatesCell)
    error("No hay plantillas tpl_%s_*.mat en %s", code, tplFolder);
end

% Unificar fields
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
fprintf("Cargadas %d plantillas.\n", numel(templates));

%% ===== LEER CARPETA Y FILTRAR POR CÓDIGO =====
files = [dir(fullfile(segFolder,'*.jpg')); dir(fullfile(segFolder,'*.png')); dir(fullfile(segFolder,'*.jpeg'))];
if isempty(files), error("No hay imágenes en %s", segFolder); end

idxOk = false(numel(files),1);
for i = 1:numel(files)
    [~,base,~] = fileparts(files(i).name);
    parts = split(base,'_');
    if numel(parts)>=1 && string(parts{1})==code
        idxOk(i)=true;
    end
end
files = files(idxOk);
fprintf("Imágenes de code=%s: %d\n\n", code, numel(files));

%% ===== TEST + DEBUG =====
for i = 1:numel(files)
    path = fullfile(files(i).folder, files(i).name);
    I = imread(path);

    [predYaw, predPitch, scores, gap] = predictYawPitch_byTemplate(I, templates);
    bestScore = max(scores);

    [~,name,ext] = fileparts(path);
    fname = string(name)+string(ext);

    fprintf("%-22s -> Pred:%03d/%02d (score=%.3f gap=%.3f)\n", fname, predYaw, predPitch, bestScore, gap);

    % ---- Si es "raro", imprime TOP5 y guarda patch/mask ----
    if (bestScore < TH_SCORE_LOW) || (gap < TH_GAP_LOW)
        fprintf("   [DEBUG] caso raro: score=%.3f, gap=%.3f\n", bestScore, gap);

        % TOP5
        [srt, idxs] = sort(scores, 'descend');
        fprintf("   TOP5:\n");
        for t = 1:min(5,numel(idxs))
            k = idxs(t);
            fprintf("     #%d: yaw=%03d pitch=%02d score=%.3f\n", ...
                t, templates(k).orientation, templates(k).pitch, srt(t));
        end

        % Guardar patch normalizado y máscara para inspección
        outSize = templates(1).size;
        [J, M, ok] = buildPatchAndMaskForDebug(I, outSize);
        if ok
            imwrite(mat2gray(J), fullfile(dbgOut, sprintf("%s_patch.png", name)));
            imwrite(uint8(M)*255,  fullfile(dbgOut, sprintf("%s_mask.png",  name)));

            % opcional: imagen original con máscara encima (simple)
            overlay = I;
            if size(overlay,3)==1, overlay = repmat(overlay,[1 1 3]); end
            R = overlay(:,:,1); G = overlay(:,:,2); B = overlay(:,:,3);
            R(M) = uint8(min(double(R(M)) + 60, 255));
            overlay(:,:,1)=R; overlay(:,:,2)=G; overlay(:,:,3)=B;
            imwrite(overlay, fullfile(dbgOut, sprintf("%s_overlay.png", name)));
        end
        fprintf("   [DEBUG] guardado en %s\n\n", dbgOut);
    end
end

fprintf("\nListo. Mira la carpeta: %s\n", dbgOut);

%% ===== FUNCIONES DEBUG (autocontenidas) =====
function [J, Mfull, ok] = buildPatchAndMaskForDebug(I, outSize)
    ok = false;
    J  = zeros(outSize,outSize);
    Mfull = false(size(I,1), size(I,2));

    try
        M = extractMaskLego(I);
        Mfull = M;
        if ~any(M(:)), return; end

        stats = regionprops(M, 'BoundingBox', 'Area');
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
        Mc = M(y:y2, x:x2);
        Gc(~Mc) = 0;

        S = max(size(Gc));
        pad = floor((S - size(Gc))/2);
        Gp = padarray(Gc, pad, 0, 'both');
        Gp = imresize(Gp, [outSize outSize], 'bilinear');

        Gp = Gp - mean(Gp(:));
        nrm = norm(Gp(:));
        if nrm < 1e-9, return; end
        J = Gp / nrm;

        ok = true;
    catch
        ok = false;
    end
end
