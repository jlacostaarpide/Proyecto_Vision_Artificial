function [predYaw, predPitch, scoresBest, gap] = predictYawPitch_byTemplate(I, templates)
% Matching robusto por correlación normalizada contra plantillas (yaw,pitch)
% -> prueba varios centrados (multi-crop) y se queda con el mejor.

    assert(nargin == 2, 'predictYawPitch_byTemplate requiere (I, templates)');
    assert(~isempty(templates), 'templates vacío');

    outSize = templates(1).size;

    % shifts en píxeles (en coordenadas del recorte original antes de resize)
    % si tus piezas son grandes, puedes subir a 6; si son pequeñas, baja a 2.
    shifts = [-4 0 4];

    K = numel(templates);

    bestGlobalScore = -inf;
    bestIdx = 1;
    bestScores = -inf(1,K);
    bestJ = [];
    okAny = false;

    for dy = 1:numel(shifts)
        for dx = 1:numel(shifts)
            [J, ok] = normalizeMaskedPatch_shift(I, outSize, shifts(dx), shifts(dy));
            if ~ok, continue; end
            okAny = true;

            v = J(:);
            scores = -inf(1,K);
            for k = 1:K
                scores(k) = dot(v, templates(k).T(:));
            end

            [smax, idx] = max(scores);
            if smax > bestGlobalScore
                bestGlobalScore = smax;
                bestIdx = idx;
                bestScores = scores;
                bestJ = J;
            end
        end
    end

    if ~okAny
        predYaw = NaN; predPitch = NaN;
        scoresBest = nan(1,K);
        gap = NaN;
        return;
    end

    scoresBest = bestScores;
    predYaw   = templates(bestIdx).orientation;
    predPitch = templates(bestIdx).pitch;

    % gap contra el segundo mejor
    s2 = scoresBest; s2(bestIdx) = -inf;
    gap = bestGlobalScore - max(s2);

    % (opcional) desempate 180° SOLO si el opuesto está cerca y quieres
    % flipYaw = mod(predYaw + 180, 360);
    % idxFlip = find([templates.orientation] == flipYaw & [templates.pitch] == predPitch);
    % if ~isempty(idxFlip)
    %     flipScore = max(scoresBest(idxFlip));
    %     if (bestGlobalScore - flipScore) < 0.03
    %         predYaw = breakTieFlip180(bestJ, predYaw);
    %     end
    % end
end

function [J, ok] = normalizeMaskedPatch_shift(I, outSize, shiftX, shiftY)
% Igual que tu normalización, pero permitiendo desplazar el recorte unos píxeles.

    ok = false;
    J  = zeros(outSize,outSize);

    try
        mask = extractMaskLego(I);
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

        x = floor(bb(1)) + shiftX;
        y = floor(bb(2)) + shiftY;
        w = ceil(bb(3));
        h = ceil(bb(4));

        x = max(x,1); y = max(y,1);
        x2 = min(x+w-1, size(G,2));
        y2 = min(y+h-1, size(G,1));

        if x2 <= x || y2 <= y, return; end

        Gc = G(y:y2, x:x2);
        Mc = mask(y:y2, x:x2);

        % OJO: como hemos desplazado el recorte, la máscara del recorte corresponde igual
        Gc(~Mc) = 0;

        % Cuadrar + resize
        H = size(Gc,1); W = size(Gc,2);
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

function yawFixed = breakTieFlip180(J, yawBest)
    Gx = imfilter(J, fspecial('sobel')'/8, 'replicate');
    Gy = imfilter(J, fspecial('sobel') /8, 'replicate');
    E  = Gx.^2 + Gy.^2;

    topEnergy    = sum(E(1:end/2,:), 'all');
    bottomEnergy = sum(E(end/2+1:end,:), 'all');

    if topEnergy >= bottomEnergy
        yawFixed = yawBest;
    else
        yawFixed = mod(yawBest + 180, 360);
    end
end
