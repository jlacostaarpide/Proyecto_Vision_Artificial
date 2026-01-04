%% SEGMENTACIÓN DE PIEZAS LEGO

clear; close all; clc;

% addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\matlab_imagen\Matlab - Imagen")
% addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\legocodes")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database\DB_G01_COD123")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database\DB_G02_COD456")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database\DB_G03_COD789")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database\DB_G04_COD101112")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database\tests")
addpath("C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Database")
addpath("C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test2");

%% Procesamiento por lotes

imagenes = {
    'IMG_7647.jpg';
    'IMG_7643.jpg';
    '4_legos.jpg';
    '07_270_70_003.jpg';
    '07_315_10_005.jpg';
    '09_270_70_001.jpg';
    '09_270_70_003.jpg';

    '08_270_40_003.jpg';

    '01_270_70_003.jpg';
    '04_270_10_003.jpg';
    '04_045_10_003.jpg';
    '04_270_40_003.jpg';
    '04_045_40_003.jpg';
    '04_270_70_003.jpg';
    '04_045_70_003.jpg';
    '04_270_90_003.jpg';
    '10_270_70_003.jpg';
    '11_270_70_003.jpg';
    '11_135_70_003.jpg';
    '11_45_90_002.jpg';
    '12_270_70_003.jpg';

    '01_000_40_001.jpg';
    '02_090_40_001.jpg';
    '03_270_10_001.jpg';
    '05_315_10_001.jpg';
    '06_000_70_005.jpg';
    '06_135_90_001.jpg';
    '07_000_10_004.jpg';
    '08_000_40_001.jpg';
    '08_045_40_001.jpg';
    '08_180_40_004.jpg';
    '09_000_70_004.jpg';
    '10_135_10_001.jpg';
    '11_135_10_001.jpg';

    '06_000_70_001.jpg';
    '10_000_40_002.jpg';
    '10_000_70_005.jpg';
    '10_045_10_001.jpg';
    '10_045_40_002.jpg';
    '10_090_70_002.jpg';
    '10_270_70_001.jpg';
    '11_000_10_003.jpg';
    '11_000_70_002.jpg';
    '11_045_10_002.jpg';
    '11_180_40_003.jpg';
    '12_045_40_002.jpg';

    % --- Serie 02 ---
    'test2/02_000_10_001.jpg';
    'test2/02_000_10_002.jpg';
    'test2/02_000_10_003.jpg';
    'test2/02_000_10_004.jpg';

    % --- Serie 03 ---
    'test2/03_000_10_001.jpg';
    'test2/03_000_10_002.jpg';
    'test2/03_000_10_003.jpg';
    'test2/03_000_10_004.jpg';
    'test2/03_000_10_005.jpg';
    'test2/03_000_10_006.jpg';
    'test2/03_000_10_007.jpg';

    % --- Serie 04 ---
    'test2/04_000_10_001.jpg';
    'test2/04_000_10_002.jpg';
    'test2/04_000_10_003.jpg';

    % --- Serie 05 ---
    'test2/05_000_10_001.jpg';
    'test2/05_000_10_002.jpg';
    'test2/05_000_10_003.jpg';

    % --- Serie 06 ---
    'test2/06_000_10_001.jpg';
    'test2/06_000_10_002.jpg';
    'test2/06_000_10_003.jpg';
    'test2/06_000_10_004.jpg';
    'test2/06_000_10_005.jpg';
    'test2/06_000_10_006.jpg';
    'test2/06_000_10_007.jpg';

    % --- Serie 09 ---
    'test2/09_000_10_001.jpg';
    'test2/09_000_10_002.jpg';
    'test2/09_000_10_003.jpg';
    'test2/09_000_10_004.jpg';
    'test2/09_000_10_005.jpg';
    'test2/09_000_10_006.jpg';
    'test2/09_000_10_007.jpg';
    'test2/09_000_10_008.jpg';

    % --- Serie 12 ---
    'test2/12_000_10_001.jpg';
    'test2/12_000_10_002.jpg';
    'test2/12_000_10_003.jpg';
    'test2/12_000_10_004.jpg';
    'test2/12_000_10_005.jpg';
    'test2/12_000_10_006.jpg';
    'test2/12_000_10_007.jpg';
    'test2/12_000_10_008.jpg';
    'test2/12_000_10_009.jpg';

    'test2/42.jpg';
    'test2/43.jpg';
    'test2/44.jpg';
    'test2/45.jpg';
    'test2/46.jpg';
    'test2/47.jpg';
    'test2/48.jpg';
    'test2/49.jpg';
    'test2/50.jpg';
    'test2/51.jpg';
    'test2/52.jpg';
    'test2/53.jpg';
    'test2/54.jpg';
    'test2/55.jpg';
    'test2/56.jpg'
};

sweep_codes   = [1,2,3,4,5,6,7,8,9,10,11,12];      % Ej: [8] o [8, 9] (Código de pieza)
sweep_orient  = [0, 45, 90, 135, 180, 225, 270, 315];  % Ej: [0, 45, 90, 135...] (Orientación)
sweep_zenith  = [10, 40, 70, 90];        % Ej: [10, 40, 70, 90] (Ángulo Cenital)
sweep_seq     = [1,2,3,4,5];         % Ej: 1:5 o [1, 3, 5] (Número de secuencia)

for c = sweep_codes
    for o = sweep_orient
        for z = sweep_zenith
            for s = sweep_seq
                nombre_generado = sprintf('%02d_%03d_%02d_%03d.jpg', c, o, z, s);
                % imagenes{end+1} = nombre_generado;
            end
        end
    end
end

% SELECTOR DE FIGURAS (6 VENTANAS)
% 1: Canales HSV
% 2: Análisis S (Otsu + Histograma original)
% 3: Limpieza Morfológica
% 4: Resultado Final
% show_figures = [1, 1, 1, 1];
show_figures = [0, 0, 0, 1];

save_images = false;
output_folder = "C:\Users\Iñaki Janices\Documents\Github\ProyectoPSM\Matlab\temp_Segmented_final";

fprintf('Procesando %d imágenes\n', length(imagenes));

for i = 1:length(imagenes)
    nombre_imagen = imagenes{i};

    try
        I = imread(nombre_imagen);
    catch
        warning('No se pudo cargar %s. Saltando...', nombre_imagen);
        continue;
    end

    I_double = im2double(I);
    fprintf('\n--- Procesando: %s ---\n', nombre_imagen);

    % 1. PRE-PROCESAMIENTO: Correción de fondo
    R = I_double(:,:,1);
    G = I_double(:,:,2);
    B = I_double(:,:,3);
    mean_R = mean(R(:));
    mean_G = mean(G(:));
    mean_B = mean(B(:));

    R_bal = R * (mean_G / mean_R);
    G_bal = G; 
    B_bal = B * (mean_G / mean_B);

    I_balanced = cat(3, R_bal, G_bal, B_bal);
    I_balanced(I_balanced > 1) = 1;

    I_hsv_temp = rgb2hsv(I_balanced);

    % Multiplicar canal S por 1.55
    S_raw = I_hsv_temp(:,:,2);
    S_boosted = S_raw * 1.55;
    S_boosted(S_boosted > 1) = 1;

    V_raw = I_hsv_temp(:,:,3);
    
    V_filt = imgaussfilt(V_raw, 120);    
    V_corrected = V_raw ./ max(V_filt(:));
    
    I_hsv_temp(:,:,3) = V_corrected;
    I_hsv_temp(:,:,2) = S_boosted;
    I_corrected = hsv2rgb(I_hsv_temp);
    % I_corrected = I_double;

    % 2. TRANSFORMACIÓN A HSV
    I_hsv = rgb2hsv(I_corrected);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);

    % FIGURA 1: Correción de fondo y canales HSV
    if show_figures(1) == 1
        figure('Name', 'Análisis de Canales HSV', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        subplot(2,2,1); imshow(I_double); colormap(gca, 'hsv'); title('Imagen original');
        subplot(2,2,2); imshow(I_corrected); colormap(gca, 'hsv'); title('Imagen normalizada en V');
        subplot(2,3,4); imshow(H); colormap(gca, 'hsv'); title('Canal H (Matiz)');
        subplot(2,3,5); imshow(S); colormap(gca, 'jet'); title('Canal S (Saturación)');
        subplot(2,3,6); imshow(V); colormap(gca, 'gray'); title('Canal V (Valor)');
    end

    % 3. ANÁLISIS CANAL S (Otsu)
    gamma_val = 1;
    S_proc = S .^ gamma_val;
    
    % Calculamos 2 umbrales
    thresh_vals = multithresh(S_proc, 1);

    mask_S = S_proc > thresh_vals(1);

    
    if show_figures(2) == 1
        figure('Name', 'Canal S - Otsu', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        subplot(2, 2, 1);
        imshow(S_proc); colormap(gca, 'jet'); colorbar;
        title('Canal Saturación (Entrada)');

        subplot(2, 2, 2);
        imhist(S_proc); hold on;
        xline(thresh_vals(1), 'r', 'LineWidth', 2);

        text(thresh_vals(1), max(ylim)*0.8, sprintf(' Th: %.3f', thresh_vals(1)), 'Color', 'r', 'FontWeight', 'bold');
        title({'Histograma'});
        xlabel('Intensidad S'); ylabel('Píxeles');

        subplot(2, 2, [3 4]);
        imshow(mask_S);
        title(['Máscara Binaria Final S (Umbral Otsu: ' num2str(thresh_vals(1)) ')']);
    end

    % 4. FUSIÓN Y MORFOLOGÍA
    % UNIÓN LÓGICA (OR)
    mask_combined = mask_S;

    % PERÍMETRO Y LIMPIEZA
    se_suture = strel('disk', 3); 
    mask_sutured = imclose(mask_combined, se_suture);
    mask_filled = imfill(mask_sutured, 'holes');
    se_noise = strel('disk', 3);
    mask_clean = imopen(mask_filled, se_noise);
    mask_final = imclearborder(mask_clean);

    % OPERACIÓN DE CIERRE
    radio_disco = 14;
    se_merge = strel('disk', radio_disco);
    mask_merged = imclose(mask_final, se_merge);

    % RELLENO
    mask_merged = imfill(mask_merged, 'holes');

    % APERTURA FINAL (Suavizar contornos)
    se_smooth = strel('disk', 4);
    mask_final_consolidated = imopen(mask_merged, se_smooth);

    % FIGURA 5: Morfología
    if show_figures(3) == 1
        figure('Name', 'Limpieza Morfológica', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        subplot(2, 3, 1);
        imshow(mask_combined);
        title({'1: Binaria Original', '(Con agujeros y ruido, S + H)'});
        subplot(2, 3, 2);
        imshow(mask_filled);
        title({'2: Relleno de Huecos', '(imfill)'});
        subplot(2, 3, 3);
        imshow(mask_clean);
        title({'3: Eliminación de Ruido', '(imopen: borra puntos pequeños)'});
        subplot(2, 3, 4);
        imshow(mask_final);
        title({'4: Eliminación de bordes', '(imclearborder: quita bordes)'});
        subplot(2, 3, 5); imshow(mask_merged);
        title(sprintf('5. CIERRE (Disco R=%d)', radio_disco));
        subplot(2, 3, 6); imshow(mask_final_consolidated);
        title('6. APERTURA (Suavizado Final)');
    end

    mask_final = mask_final_consolidated;

    % 5. RESULTADOS Y FILTRADO
    [L, num_inicial] = bwlabel(mask_final, 8);
    stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', 'Perimeter', 'Circularity', 'Image', 'PixelIdxList');
    
    if ~isempty(stats)
        all_areas = [stats.Area];
        max_area = max(all_areas);

        umbral_area_rel = 0.15 * max_area;
        umbral_area_abs = 1000;      % Mínimo 1000 píxeles
        umbral_ratio_max = 4.0;      % El lado largo no puede ser más de 5 veces el corto
        umbral_saturacion = 0.30;    % Mínima saturación de color
        
        % Comprobación de area
        candidates_idx = find((all_areas > umbral_area_rel) & (all_areas > umbral_area_abs));

        I_hsv_check = rgb2hsv(I_corrected); 
        S_channel = I_hsv_check(:,:,2);
        
        valid_idx = [];
        
        for k = 1:length(candidates_idx)
            idx_obj = candidates_idx(k);

            % Comprobación de Aspect Ratio
            bb = stats(idx_obj).BoundingBox; % [x, y, w, h]
            width = bb(3);
            height = bb(4);
            ratio = max(width, height) / min(width, height);
            
            if ratio > umbral_ratio_max
                fprintf('  - Descartado obj #%d por forma alargada (Ratio: %.1f)\n', idx_obj, ratio);
                continue; % Salta al siguiente ciclo sin mirar color
            end

            % Comprobación de Saturación Promedio
            pixels_indices = stats(idx_obj).PixelIdxList;
            mean_sat = mean(S_channel(pixels_indices));
            
            if mean_sat > umbral_saturacion
                valid_idx = [valid_idx; idx_obj];
            else
                fprintf('  - Descartado obj #%d por baja saturación (%.2f)\n', idx_obj, mean_sat);
            end
        end
        
        mask_filtered = ismember(L, valid_idx);
        stats_final = regionprops(mask_filtered, 'Area', 'Centroid', 'BoundingBox', 'Circularity', 'Image');

        % Ordenar por Area
        if ~isempty(stats_final)
            areas_finales = [stats_final.Area];
            [~, sort_idx] = sort(areas_finales, 'descend'); % 'descend' = de grande a pequeño
            stats_final = stats_final(sort_idx);
        end

        num_final = length(stats_final);
    else
        mask_filtered = mask_final;
        stats_final = [];
        num_final = 0;
    end
    fprintf('  > Objetos Detectados: %d\n', num_final);

    % 6. VISUALIZACIÓN DE RESULTADOS
    if (show_figures(4) == 1 || save_images)
        if show_figures(4) == 1
            figure('Name', 'Resultados Finales de Segmentación', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

            % Imagen Original
            subplot(2, 2, 1);
            imshow(I);
            title(sprintf('Imagen Original: %s', nombre_imagen), 'Interpreter', 'none');

            % Imagen con Corrección y Superposiciones
            subplot(2, 2, 2);
            imshow(I_corrected); hold on;
            title(sprintf('Detección Final: %d Piezas (Corrección Brillo + HSV + Otsu + Morfología)', num_final));
        end

        for k = 1:num_final
            % Datos
            c = stats_final(k).Centroid;
            bb = stats_final(k).BoundingBox;
            circ = stats_final(k).Circularity;
            area = stats_final(k).Area;

            if show_figures(4) == 1
                % Dibujar Bounding Box
                rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 2);

                % Marcar Centroide
                plot(c(1), c(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);

                % Etiquetar
                str_label = sprintf('#%d\nC: %.2f\nA: %d', k, circ, round(area));
                text(bb(1), bb(2)-20, str_label, 'Color', 'yellow', 'FontWeight', 'bold', 'FontSize', 8, 'BackgroundColor', 'k');
            end
        end
        hold off;

        % Visualización de Objetos Individuales
        if num_final > 0
            for k = 1:min(num_final, 4)

                % Extraer la pieza con fondo negro
                bb = stats_final(k).BoundingBox;
                img_crop = imcrop(I_double, bb);

                % Recortar la máscara correspondiente a el objeto
                mask_local = stats_final(k).Image;
                mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');

                % Aplicar máscara para fondo negro
                img_crop_masked = img_crop;
                % Replicar máscara para 3 canales RGB
                mask_3ch = cat(3, mask_local, mask_local, mask_local);
                img_crop_masked(~mask_3ch) = 0;

                hsv_crop = rgb2hsv(img_crop_masked);
                V_crop = hsv_crop(:,:,3);

                p1 = prctile(V_crop(:), 1);
                p95 = prctile(V_crop(:), 95);
                v_eq = (V_crop - p1) / (p95 - p1);
                v_eq = max(0, min(1, v_eq));
                hsv_crop(:,:,3) = v_eq;

                img_crop_enhanced = hsv2rgb(hsv_crop);

                if show_figures(4) == 1
                    subplot(2, 4, 4+k);

                    imshow(img_crop_enhanced);
                    title(sprintf('Pieza #%d', k));
                end
                if save_images
                    % Crear carpeta si no existe
                    if ~exist(output_folder, 'dir')
                        mkdir(output_folder);
                    end

                    % Obtener nombre base y extensión
                    [~, name_base, ext_orig] = fileparts(nombre_imagen);

                    if num_final > 1
                        % char(97) es 'a'. Usamos 96 + k para sacar a, b, c...
                        suffix = sprintf('_%c', char(96 + k));
                    else
                        suffix = ''; % Sin sufijo si solo hay una pieza
                    end

                    % Construir nombre final: segmented_NombreOriginal_a.jpg
                    nombre_guardado = sprintf('segmented_%s%s%s', name_base, suffix, ext_orig);
                    ruta_completa = fullfile(output_folder, nombre_guardado);

                    imwrite(img_crop_enhanced, ruta_completa);
                    fprintf('   > Guardado: %s\n', nombre_guardado);
                end
            end
        end
    end
    pause(0.5);
    % pause;
    % close all;
end