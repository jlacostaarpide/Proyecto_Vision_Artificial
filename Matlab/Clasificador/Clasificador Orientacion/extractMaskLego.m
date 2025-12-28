function maskN = extractMaskLego(I)
% EXTRACTMASKLEGO  Extrae máscara binaria del LEGO (sin normalizar orientación)
% Entrada:  I RGB o gray (segmentada o no), fondo idealmente negro u oscuro
% Salida:   maskN logical (máscara limpia del objeto)

    if ~isfloat(I), I = im2double(I); end
    if size(I,3) == 3
        Ig = rgb2gray(I);
    else
        Ig = I;
    end

    % --- Umbral simple para separar fondo/objeto ---
    % Ajusta si tu fondo no es negro: 0.03..0.10 típico
    t = 0.03;
    mask = Ig > t;

    % --- Limpieza ---
    mask = bwareaopen(mask, 150);
    mask = imclose(mask, strel('disk', 2));
    mask = imopen(mask,  strel('disk', 1));
    mask = imfill(mask, 'holes');

    % --- Quedarse con el componente mayor ---
    CC = bwconncomp(mask, 8);
    if CC.NumObjects == 0
        maskN = false(size(mask));
        return;
    end
    if CC.NumObjects > 1
        areas = cellfun(@numel, CC.PixelIdxList);
        [~, imax] = max(areas);
        mask2 = false(size(mask));
        mask2(CC.PixelIdxList{imax}) = true;
        mask = mask2;
    end

    maskN = logical(mask);
end
