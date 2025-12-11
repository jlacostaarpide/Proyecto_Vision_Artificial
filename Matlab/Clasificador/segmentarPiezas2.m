function [images_final, stats_final, num_final, I_corrected] = segmentarPiezas(nombre_imagen)
% SEGMENTARPIEZAS  Segmenta las piezas LEGO de una imagen.
%   [images_final, stats_final, num_final, I_corrected] = segmentarPiezas(nombre_imagen)
%   - nombre_imagen: string con el nombre o ruta de la imagen.
%   - images_final : cell array (1x4) con las piezas recortadas, fondo negro.
%   - stats_final  : struct de regionprops de las piezas.
%   - num_final    : número de piezas detectadas.
%   - I_corrected  : imagen corregida (ahora igual que la original).

    % 1. CARGA DE LA IMAGEN
    I = imread(nombre_imagen);
    I_double = im2double(I);
    I_corrected = I_double;   % de momento sin corrección de iluminación adicional

    % 2. TRANSFORMACIÓN A HSV
    I_hsv = rgb2hsv(I_corrected);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);

    % --- ECUALIZACIÓN DEL CANAL V (Percentiles 1% y 95%) ---
    limits = stretchlim(V, [0.01 0.05]); 
    V_eq = imadjust(V, limits, []); 
    I_hsv(:,:,3) = V_eq;
    V = V_eq;                            % usamos el V ecualizado en lo que sigue
    % Si quisieras trabajar en RGB corregido, podrías hacer:
    % I_corrected = hsv2rgb(I_hsv);

    % 4. ANÁLISIS CANAL S (Multi-level Otsu)
    gamma_val = 1; 
    S_proc = S .^ gamma_val;
    thresh_vals = multithresh(S_proc, 2);   % 2 umbrales -> 3 clases
    L_quantized = imquantize(S_proc, thresh_vals);

    % Porcentaje de la clase intermedia
    num_pixels = numel(S_proc);
    count_mid = sum(L_quantized(:) == 2);
    ratio_mid = count_mid / num_pixels;

    % Si la clase media ocupa demasiado, la consideramos fondo
    umbral_area_max_mid = 0.15; 
    if ratio_mid > umbral_area_max_mid
        % Clase intermedia enorme -> fondo
        level_otsu_S = thresh_vals(2);
    else
        % Clase intermedia pequeña -> parte del LEGO
        level_otsu_S = thresh_vals(2); %se puede cambiar
    end

    mask_S = imbinarize(S_proc, level_otsu_S);

    % 5. ANÁLISIS CANAL H (rescate morado/rosa)
    % Rango Morado/Rosa: 0.68 a 0.88
    % Condición de seguridad: S > 40% del umbral Otsu de S
    min_sat_H = 0.4 * level_otsu_S;
    mask_H_purple = (H >= 0.68) & (H <= 0.88) & (S > min_sat_H);

    % 6. ANÁLISIS CANAL V (oscuro) – aquí lo dejamos desactivado como en tu script
    V_inv = imcomplement(V);
    level_otsu_V = graythresh(V_inv); %#ok<NASGU>
    mask_V_dark = false(size(V_inv));   % desactivado

    % 7. FUSIÓN Y MORFOLOGÍA BÁSICA
    mask_combined = mask_S | mask_H_purple | mask_V_dark;

    mask_filled = imfill(mask_combined, 'holes');
    se_noise = strel('disk', 3);
    mask_clean = imopen(mask_filled, se_noise);
    mask_final = imclearborder(mask_clean);

    % 8. PROCESADO MORFOLÓGICO AVANZADO (cierre + apertura)
    radio_pegamento = 12; 
    se_merge = strel('disk', radio_pegamento);
    mask_merged = imclose(mask_final, se_merge);
    mask_merged = imfill(mask_merged, 'holes');
    se_smooth = strel('disk', 5);
    mask_final_consolidated = imopen(mask_merged, se_smooth);

    mask_final = mask_final_consolidated;

    % 9. RESULTADOS Y FILTRADO POR ÁREA
    [L, num_inicial] = bwlabel(mask_final, 8); %#ok<NASGU>
    stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', ...
                           'Perimeter', 'Circularity', 'Eccentricity', 'Image');
    
    if ~isempty(stats)
        all_areas = [stats.Area];
        max_area = max(all_areas);
        
        umbral_area = 0.05 * max_area; 
        % all_circ = [stats.Circularity];
        % umbral_circ = 0.2; 
        
        % Aquí filtramos solo por área (como en tu versión final)
        valid_idx = find(all_areas > umbral_area);
        mask_filtered = ismember(L, valid_idx);
        
        stats_final = regionprops(mask_filtered, 'Area', 'Centroid', ...
                                  'BoundingBox', 'Circularity', 'Image');
        num_final = numel(stats_final);
    else
        mask_filtered = mask_final; %#ok<NASGU>
        stats_final = struct([]);
        num_final = 0;
    end

    % 10. EXTRAER LAS PIEZAS INDIVIDUALES CON FONDO NEGRO

images_final = cell(1, num_final);    % una celda por pieza detectada

if num_final > 0
    for k = 1:num_final
        bb = stats_final(k).BoundingBox;
        img_crop = imcrop(I_corrected, bb);

        mask_local = stats_final(k).Image;
        mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

        img_crop_masked = img_crop;
        mask_3ch = cat(3, mask_local, mask_local, mask_local);
        img_crop_masked(~mask_3ch) = 0;  % fondo negro

        images_final{k} = img_crop_masked;
    end
end
