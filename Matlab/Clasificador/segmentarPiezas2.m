function [piece, stats_final, num_final, I_corrected, mask_final] = segmentarPiezas2(nombre_imagen)
% SEGMENTARPIEZAS2  Segmenta UNA imagen y devuelve SOLO si hay 1 pieza.
%   [piece, stats_final, num_final, I_corrected, mask_final] = segmentarPiezas2(nombre_imagen)
%
%   - piece      : imagen RGB (double) de la pieza recortada, fondo negro ([])
%                  si se descarta.
%   - stats_final: regionprops del/los objetos tras filtrado
%   - num_final  : nº de piezas detectadas tras filtrado (0 si se descarta)
%   - I_corrected: imagen corregida (double)
%   - mask_final : máscara final binaria (útil para depurar)

    %% 1) CARGA
    I = imread(nombre_imagen);
    I_corrected = im2double(I);

    %% 2) HSV
    I_hsv = rgb2hsv(I_corrected);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);

    %% 3) CANAL S (Multi-Otsu + decisión por ratio/solidez)
    gamma_val = 1.4;
    S_proc = S .^ gamma_val;

    thresh_vals = multithresh(S_proc, 2);  % 2 umbrales -> 3 clases
    Lq = imquantize(S_proc, thresh_vals);
    mask_mid_temp = (Lq == 2);

    ratio_mid = nnz(mask_mid_temp) / numel(mask_mid_temp);

    umbral_inferior = 0.055;
    umbral_superior = 0.17;

    use_lower_thresh = false;

    if ratio_mid < umbral_inferior
        use_lower_thresh = true;
    elseif ratio_mid > umbral_superior
        use_lower_thresh = false;
    else
        stMid = regionprops(mask_mid_temp, 'Area', 'Solidity');
        if ~isempty(stMid)
            [~, idxMax] = max([stMid.Area]);
            solidez_mid = stMid(idxMax).Solidity;
            use_lower_thresh = (solidez_mid > 0.6);
        else
            use_lower_thresh = false;
        end
    end

    if use_lower_thresh
        level_otsu_S = thresh_vals(1);
    else
        level_otsu_S = thresh_vals(2);
    end

    mask_S = imbinarize(S_proc, level_otsu_S);

    %% 4) CANAL H (morado/rosa)
    min_sat_H_purple = 0.4 * level_otsu_S;
    min_sat_H_pink   = 1.0 * level_otsu_S;

    mask_H_purple = (H >= 0.58) & (H <= 0.92) & (S > min_sat_H_purple);
    mask_H_pink   = (H >= 0.01) & (H <= 0.065) & (S > min_sat_H_pink);

    mask_H = mask_H_purple | mask_H_pink;

    %% 5) CANAL V (desactivado como en tu código)
    mask_V_dark = false(size(V));

    %% 6) FUSIÓN + MORFOLOGÍA (igual que tu bloque)
    mask_combined = mask_S | mask_H | mask_V_dark;

    se_suture  = strel('disk', 3);
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

    %% 7) OBJETOS + FILTRADO POR ÁREA (tu criterio)
    [L, ~] = bwlabel(mask_final, 8);
    stats = regionprops(L, 'Area','Centroid','BoundingBox','Circularity','Image');

    if isempty(stats)
        stats_final = struct([]);
        num_final = 0;
        piece = [];
        return;
    end

    all_areas = [stats.Area];
    max_area  = max(all_areas);

    umbral_area = 0.15 * max_area;
    valid_idx = find(all_areas > umbral_area);

    mask_filtered = ismember(L, valid_idx);

    stats_final = regionprops(mask_filtered, 'Area','Centroid','BoundingBox','Circularity','Image');
    num_final = numel(stats_final);

    %% 8) CONDICIÓN CLAVE: SOLO 1 PIEZA
    if num_final ~= 1
        piece = [];
        stats_final = struct([]);
        num_final = 0;
        return;
    end

    %% 9) EXTRAER PIEZA (fondo negro)
    bb = stats_final(1).BoundingBox;
    img_crop = imcrop(I_corrected, bb);

    mask_local = stats_final(1).Image;
    mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

    piece = img_crop;
    mask_3ch = cat(3, mask_local, mask_local, mask_local);
    piece(~mask_3ch) = 0;
end
