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
    I_corrected = I_double;   % de momento sin corrección de iluminación

    % 2. TRANSFORMACIÓN A HSV
    I_hsv = rgb2hsv(I_corrected);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3); %#ok<NASGU> % V no se usa, pero lo dejamos por claridad

    % 3. ESTIMAR EL COLOR DEL FONDO (MESA)
    mask_fondo = S < 0.25;                     
    H_fondo = mean(H(mask_fondo), 'all');      

    dist_H = abs(H - H_fondo);                 
    umbral_distH = 0.06;                       
    mask_H = dist_H > umbral_distH;

    % Máscara por saturación (Otsu)
    level_otsu_S = graythresh(S);
    mask_S = imbinarize(S, level_otsu_S);

    % Máscara final
    mask = mask_H | mask_S;

    % 4. MORFOLOGÍA
    se_bridge = strel('disk', 4);
    mask = imclose(mask, se_bridge);
    mask = imfill(mask, 'holes');
    se_noise = strel('disk', 2);
    mask = imopen(mask, se_noise);
    mask = bwareaopen(mask, 500);
    mask_final = imclearborder(mask);

    % 5. COMPONENTES CONEXAS Y REGIONPROPS
    [L, num_inicial] = bwlabel(mask_final, 8); %#ok<NASGU>
    stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', ...
                           'Perimeter', 'Circularity', 'Eccentricity', 'Image');

    % 6. FILTRADO DE OBJETOS
    if ~isempty(stats)
        all_areas = [stats.Area];
        max_area = max(all_areas);
        umbral_area = 0.05 * max_area;

        all_circ = [stats.Circularity];
        umbral_circ = 0.05;

        valid_idx = find((all_areas > umbral_area) & (all_circ > umbral_circ));

        final_mask_filtered = ismember(L, valid_idx);
        stats_final = regionprops(final_mask_filtered, 'Area', 'Centroid', ...
                                  'BoundingBox', 'Circularity', 'Image');
        num_final = length(stats_final);
    else
        final_mask_filtered = mask_final; %#ok<NASGU>
        stats_final = struct([]);
        num_final = 0;
    end

    % 7. EXTRAER LAS PIEZAS INDIVIDUALES (FONDO NEGRO)
    images_final = cell(1,4);    % hasta 4 piezas (puedes cambiar a num_final si quieres todas)

    if num_final > 0
        for k = 1:min(num_final, 4)
            bb = stats_final(k).BoundingBox;
            img_crop = imcrop(I_corrected, bb);

            mask_local = stats_final(k).Image;
            mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

            img_crop_masked = img_crop;
            mask_3ch = cat(3, mask_local, mask_local, mask_local);
            img_crop_masked(~mask_3ch) = 0;

            images_final{k} = img_crop_masked;
        end
    end
end
