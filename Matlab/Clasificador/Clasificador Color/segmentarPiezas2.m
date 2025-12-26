function [piece, stats_final, num_final, I_corrected, mask_final] = segmentarPiezas2(nombre_imagen)
% SEGMENTARPIEZAS2  Segmenta UNA imagen con tu versión NUEVA (S+correcciones) y devuelve SOLO si hay 1 pieza.
%   [piece, stats_final, num_final, I_corrected, mask_final] = segmentarPiezas2(nombre_imagen)
%
%   - piece      : imagen RGB (double) de la pieza recortada, fondo negro ([] si se descarta)
%   - stats_final: regionprops del/los objetos tras filtrado
%   - num_final  : nº de piezas detectadas tras filtrado (0 si se descarta)
%   - I_corrected: imagen corregida (double) (balance RGB + boost S + normalización V)
%   - mask_final : máscara final binaria (útil para depurar)
%
%   NOTA: Esta versión replica tu script nuevo:
%         - Balance simple de blancos (R,B escalados a media de G)
%         - Boost de S (x1.55)
%         - Corrección de V con gauss grande + normalización
%         - Otsu en S con 1 umbral (multithresh(...,1))
%         - Morfología: close -> fill -> open -> clearborder -> close(R=14) -> fill -> open(R=4)
%         - Filtrado por área relativa + filtrado por saturación media del objeto
%         - DEVUELVE pieza solo si queda 1 objeto final

    %% 1) CARGA
    I = imread(nombre_imagen);
    I_double = im2double(I);

    %% 2) PRE-PROCESAMIENTO: corrección fondo / balance RGB + boost S + corrección V
    R = I_double(:,:,1);
    G = I_double(:,:,2);
    B = I_double(:,:,3);

    mean_R = mean(R(:));
    mean_G = mean(G(:));
    mean_B = mean(B(:));

    % Evitar divisiones por cero
    if mean_R < eps, mean_R = eps; end
    if mean_B < eps, mean_B = eps; end

    R_bal = R * (mean_G / mean_R);
    G_bal = G;
    B_bal = B * (mean_G / mean_B);

    I_balanced = cat(3, R_bal, G_bal, B_bal);
    I_balanced(I_balanced > 1) = 1;

    I_hsv_temp = rgb2hsv(I_balanced);

    % Boost de S
    S_raw = I_hsv_temp(:,:,2);
    S_boosted = S_raw * 1.55;
    S_boosted(S_boosted > 1) = 1;

    % Corrección de V (normalización con gauss grande)
    V_raw = I_hsv_temp(:,:,3);
    V_filt = imgaussfilt(V_raw, 120);
    denomV = max(V_filt(:));
    if denomV < eps, denomV = eps; end
    V_corrected = V_raw ./ denomV;
    V_corrected = max(0, min(1, V_corrected));

    I_hsv_temp(:,:,2) = S_boosted;
    I_hsv_temp(:,:,3) = V_corrected;

    I_corrected = hsv2rgb(I_hsv_temp);

    %% 3) HSV (para segmentación)
    I_hsv = rgb2hsv(I_corrected);
    S = I_hsv(:,:,2);

    %% 4) ANÁLISIS CANAL S (Otsu con 1 umbral)
    gamma_val = 1;
    S_proc = S .^ gamma_val;

    thresh_vals = multithresh(S_proc, 1);
    mask_S = S_proc > thresh_vals(1);

    %% 5) MORFOLOGÍA (igual que tu script)
    mask_combined = mask_S;

    se_suture   = strel('disk', 3);
    mask_sutured = imclose(mask_combined, se_suture);

    mask_filled = imfill(mask_sutured, 'holes');

    se_noise   = strel('disk', 3);
    mask_clean = imopen(mask_filled, se_noise);

    mask_noBorder = imclearborder(mask_clean);

    radio_disco = 14;
    se_merge = strel('disk', radio_disco);
    mask_merged = imclose(mask_noBorder, se_merge);

    mask_merged = imfill(mask_merged, 'holes');

    se_smooth = strel('disk', 4);
    mask_final = imopen(mask_merged, se_smooth);

    %% 6) OBJETOS + FILTROS (área relativa + saturación media)
    [L, ~] = bwlabel(mask_final, 8);
    stats = regionprops(L, 'Area','Centroid','BoundingBox','Perimeter','Circularity','Image','PixelIdxList');

    if isempty(stats)
        stats_final = struct([]);
        num_final   = 0;
        piece       = [];
        return;
    end

    all_areas = [stats.Area];
    max_area  = max(all_areas);

    umbral_area = 0.15 * max_area;
    candidates_idx = find(all_areas > umbral_area);

    % Filtro por saturación media en los píxeles del objeto (sobre I_corrected)
    I_hsv_check = rgb2hsv(I_corrected);
    S_channel   = I_hsv_check(:,:,2);

    umbral_saturacion = 0.25;
    valid_idx = [];

    for k = 1:numel(candidates_idx)
        idx_obj = candidates_idx(k);
        pixIdx  = stats(idx_obj).PixelIdxList;

        mean_sat = mean(S_channel(pixIdx));

        if mean_sat > umbral_saturacion
            valid_idx = [valid_idx; idx_obj]; %#ok<AGROW>
        end
    end

    mask_filtered = ismember(L, valid_idx);

    stats_final = regionprops(mask_filtered, 'Area','Centroid','BoundingBox','Circularity','Image');
    num_final   = numel(stats_final);

    %% 7) CONDICIÓN: SOLO 1 PIEZA
    if num_final ~= 1
        piece      = [];
        stats_final = struct([]);
        num_final  = 0;
        return;
    end

    %% 8) EXTRAER PIEZA (fondo negro) + MEJORA V local (percentiles) como en tu script
    bb = stats_final(1).BoundingBox;
    img_crop = imcrop(I_double, bb);

    mask_local = stats_final(1).Image;
    mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

    % Aplicar máscara (fondo negro)
    piece = img_crop;
    mask_3ch = cat(3, mask_local, mask_local, mask_local);
    piece(~mask_3ch) = 0;

    % Realce de V en el recorte (percentiles 1-95) sobre la pieza ya en negro
    hsv_crop = rgb2hsv(piece);
    V_crop = hsv_crop(:,:,3);

    p1  = prctile(V_crop(:), 1);
    p95 = prctile(V_crop(:), 95);

    denom = (p95 - p1);
    if denom < eps, denom = eps; end

    v_eq = (V_crop - p1) / denom;
    v_eq = max(0, min(1, v_eq));

    hsv_crop(:,:,3) = v_eq;
    piece = hsv2rgb(hsv_crop);
end
