function [pieces, stats_final, num_final, I_corrected, mask_filtered] = segmentarPiezas2(nombre_imagen, opts)
% SEGMENTARPIEZAS_SCRIPTG01
% Función equivalente al script largo "SEGMENTACIÓN DE PIEZAS LEGO"
% - MISMO pipeline: WB por medias RGB -> HSV -> S*1.55 -> V_corrected (Gauss sigma 120 / max global)
% - Otsu (multithresh) sobre S -> morfología (close/fill/open/clearborder/close/fill/open)
% - Filtrado por área relativa+absoluta, ratio, saturación media
% - Orden por área descendente
% - Recorte de cada pieza con fondo negro + realce V por percentiles (p1 y p95)
%
% Entradas:
%   nombre_imagen : ruta o nombre del archivo
%   opts          : struct opcional para override de parámetros
%
% Salidas:
%   pieces       : cell array {num_final x 1}, cada celda es RGB double [0..1] recortada con fondo negro y realce V
%   stats_final  : regionprops de piezas válidas (ordenadas por área desc)
%   num_final    : número de piezas válidas detectadas
%   I_corrected  : imagen corregida (RGB double)
%   mask_filtered: máscara final con piezas válidas (logical)

    if nargin < 2, opts = struct(); end

    % =======================
    % Defaults (idénticos al script)
    % =======================
    d.S_boost           = 1.55;
    d.V_gauss_sigma     = 120;
    d.gamma_val         = 1;

    d.umbral_area_rel   = 0.15;
    d.umbral_area_abs   = 1000;
    d.umbral_ratio_max  = 4.0;
    d.umbral_saturacion = 0.30;

    d.se_suture_r = 3;
    d.se_noise_r  = 3;
    d.se_merge_r  = 14;
    d.se_smooth_r = 4;

    % Realce final de V en el recorte (idéntico al script)
    d.v_prc_low  = 1;
    d.v_prc_high = 95;

    % Guardado opcional (por defecto apagado; el script lo controlaba con save_images)
    d.save_images    = false;
    d.output_folder  = "";

    % Override con opts
    fn = fieldnames(d);
    for k = 1:numel(fn)
        if isfield(opts, fn{k})
            d.(fn{k}) = opts.(fn{k});
        end
    end
    opts = d;

    % =======================
    % Init outputs
    % =======================
    pieces = {};
    stats_final = struct([]);
    num_final = 0;
    I_corrected = [];
    mask_filtered = [];

    % =======================
    % Load
    % =======================
    if ~isfile(nombre_imagen)
        return;
    end
    try
        I = imread(nombre_imagen);
    catch
        return;
    end
    if size(I,3) ~= 3
        I = repmat(I, [1 1 3]);
    end

    I_double = im2double(I);

    % =====================================================================
    % 1) PRE-PROCESAMIENTO: Corrección de fondo (idéntico al script)
    % =====================================================================
    R = I_double(:,:,1);
    G = I_double(:,:,2);
    B = I_double(:,:,3);

    mean_R = mean(R(:));
    mean_G = mean(G(:));
    mean_B = mean(B(:));

    % Para evitar NaN/Inf si algo raro:
    if abs(mean_R) < 1e-12, mean_R = 1e-12; end
    if abs(mean_B) < 1e-12, mean_B = 1e-12; end

    R_bal = R * (mean_G / mean_R);
    G_bal = G;
    B_bal = B * (mean_G / mean_B);

    I_balanced = cat(3, R_bal, G_bal, B_bal);
    I_balanced(I_balanced > 1) = 1;

    I_hsv_temp = rgb2hsv(I_balanced);

    % Multiplicar canal S por 1.55
    S_raw = I_hsv_temp(:,:,2);
    S_boosted = S_raw * opts.S_boost;
    S_boosted(S_boosted > 1) = 1;

    % Corrección de V: Gauss + normalización por max global (idéntico)
    V_raw = I_hsv_temp(:,:,3);
    V_filt = imgaussfilt(V_raw, opts.V_gauss_sigma);
    denom = max(V_filt(:));
    if denom < 1e-12, denom = 1e-12; end
    V_corrected = V_raw ./ denom;

    I_hsv_temp(:,:,3) = V_corrected;
    I_hsv_temp(:,:,2) = S_boosted;

    I_corrected = hsv2rgb(I_hsv_temp);

    % =====================================================================
    % 2) TRANSFORMACIÓN A HSV (idéntico)
    % =====================================================================
    I_hsv = rgb2hsv(I_corrected);
    S = I_hsv(:,:,2);

    % =====================================================================
    % 3) ANÁLISIS CANAL S (Otsu) (idéntico)
    % =====================================================================
    S_proc = S .^ opts.gamma_val;

    thresh_vals = multithresh(S_proc, 1);
    if isempty(thresh_vals) || ~isfinite(thresh_vals(1))
        mask_filtered = false(size(S_proc));
        return;
    end

    mask_S = S_proc > thresh_vals(1);

    % =====================================================================
    % 4) MORFOLOGÍA (idéntico)
    % =====================================================================
    mask_combined = mask_S;

    se_suture = strel('disk', opts.se_suture_r);
    mask_sutured = imclose(mask_combined, se_suture);
    mask_filled = imfill(mask_sutured, 'holes');

    se_noise = strel('disk', opts.se_noise_r);
    mask_clean = imopen(mask_filled, se_noise);
    mask_final = imclearborder(mask_clean);

    se_merge = strel('disk', opts.se_merge_r);
    mask_merged = imclose(mask_final, se_merge);
    mask_merged = imfill(mask_merged, 'holes');

    se_smooth = strel('disk', opts.se_smooth_r);
    mask_final_consolidated = imopen(mask_merged, se_smooth);

    mask_final = mask_final_consolidated;

    % =====================================================================
    % 5) RESULTADOS Y FILTRADO (idéntico)
    % =====================================================================
    [L, ~] = bwlabel(mask_final, 8);
    stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', 'Perimeter', ...
                           'Circularity', 'Image', 'PixelIdxList');

    if isempty(stats)
        mask_filtered = false(size(mask_final));
        stats_final = struct([]);
        num_final = 0;
        pieces = {};
        return;
    end

    all_areas = [stats.Area];
    max_area = max(all_areas);

    umbral_area_rel = opts.umbral_area_rel * max_area;

    candidates_idx = find((all_areas > umbral_area_rel) & (all_areas > opts.umbral_area_abs));

    % En el script se recalculaba hsv sobre I_corrected; aquí es equivalente:
    S_channel = I_hsv(:,:,2);

    valid_idx = [];

    for k = 1:length(candidates_idx)
        idx_obj = candidates_idx(k);

        % Aspect ratio
        bb = stats(idx_obj).BoundingBox; % [x, y, w, h]
        width = bb(3);
        height = bb(4);
        minwh = min(width, height);
        if minwh < 1e-12, minwh = 1e-12; end
        ratio = max(width, height) / minwh;

        if ratio > opts.umbral_ratio_max
            continue;
        end

        % Saturación promedio
        pixels_indices = stats(idx_obj).PixelIdxList;
        mean_sat = mean(S_channel(pixels_indices));

        if mean_sat > opts.umbral_saturacion
            valid_idx = [valid_idx; idx_obj]; %#ok<AGROW>
        end
    end

    mask_filtered = ismember(L, valid_idx);

    % OJO: en el script: stats_final = regionprops(mask_filtered, ...)
    % (regionprops acepta máscara lógica)
    stats_final = regionprops(mask_filtered, 'Area', 'Centroid', 'BoundingBox', ...
                              'Circularity', 'Image', 'PixelIdxList');

    if isempty(stats_final)
        num_final = 0;
        pieces = {};
        return;
    end

    % Ordenar por área descendente (idéntico)
    areas_finales = [stats_final.Area];
    [~, sort_idx] = sort(areas_finales, 'descend');
    stats_final = stats_final(sort_idx);

    num_final = length(stats_final);

    % =====================================================================
    % 6) RECORTE DE PIEZAS + FONDO NEGRO + REALCE V (idéntico)
    % =====================================================================
    pieces = cell(num_final, 1);

    % Para el guardado opcional, necesitamos nombre base
    [~, name_base, ext_orig] = fileparts(nombre_imagen);

    for k = 1:num_final
        bb = stats_final(k).BoundingBox;

        % Extraer la pieza con fondo negro
        img_crop = imcrop(I_double, bb);

        % Máscara local del objeto (Image) redimensionada al crop (idéntico)
        mask_local = stats_final(k).Image;
        mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

        % Aplicar máscara (fondo negro)
        img_crop_masked = img_crop;
        mask_3ch = cat(3, mask_local, mask_local, mask_local);
        img_crop_masked(~mask_3ch) = 0;

        % Realce V por percentiles (idéntico)
        hsv_crop = rgb2hsv(img_crop_masked);
        V_crop = hsv_crop(:,:,3);

        p1  = prctile(V_crop(:), opts.v_prc_low);
        p95 = prctile(V_crop(:), opts.v_prc_high);

        denomV = (p95 - p1);
        if abs(denomV) < 1e-12
            denomV = 1e-12;
        end

        v_eq = (V_crop - p1) / denomV;
        v_eq = max(0, min(1, v_eq));
        hsv_crop(:,:,3) = v_eq;

        img_crop_enhanced = hsv2rgb(hsv_crop);

        pieces{k} = img_crop_enhanced;

        % Guardado opcional (equivalente al script)
        if opts.save_images
            if strlength(opts.output_folder) == 0
                continue;
            end
            if ~exist(opts.output_folder, 'dir')
                mkdir(opts.output_folder);
            end

            if num_final > 1
                suffix = sprintf('_%c', char(96 + k)); % _a, _b, ...
            else
                suffix = '';
            end

            nombre_guardado = sprintf('segmented_%s%s%s', name_base, suffix, ext_orig);
            ruta_completa = fullfile(opts.output_folder, nombre_guardado);
            imwrite(img_crop_enhanced, ruta_completa);
        end
    end
end
