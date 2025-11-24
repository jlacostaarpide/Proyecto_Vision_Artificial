%% SEGMENTACIÓN DE PIEZAS LEGO

clear; close all; clc;

% addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\matlab_imagen\Matlab - Imagen")
% addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\legocodes")
addpath("C:\Users\Iñaki Janices\Documentos\Github\ProyectoPSM\Database\DB_G03_COD789")
addpath("C:\Users\Iñaki Janices\Documentos\Github\ProyectoPSM\Database\tests")

%% 1. CARGA DE LA IMAGEN
imagenes = {
    'IMG_7647.jpg';
    'IMG_7643.jpg';
    '4_legos.jpg';
    '07_270_70_003.jpg';
    '07_315_10_005.jpg';
    '08_270_70_003.jpg';
};

% SELECTOR DE FIGURAS (6 VENTANAS)
% 1: Canales HSV
% 2: Análisis S (Otsu + Histograma original)
% 3: Análisis H (Detección de Morado)
% 4: Análisis V (Detección de Oscuros + Histograma Invertido)
% 5: Limpieza Morfológica
% 6: Resultado Final
% show_figures = [1, 1, 1, 1, 1, 1]; 
show_figures = [0, 0, 0, 0, 0, 1]; 

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
    
    % 2. PRE-PROCESAMIENTO
    I_corrected = I_double; 
    
    % 3. TRANSFORMACIÓN A HSV
    I_hsv = rgb2hsv(I_corrected);
    H = I_hsv(:,:,1);
    S = I_hsv(:,:,2);
    V = I_hsv(:,:,3);
    
    % FIGURA 1: Canales Originales
    if show_figures(1) == 1
        figure('Name', 'Análisis de Canales HSV', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
        subplot(1,3,1); imshow(H); colormap(gca, 'hsv'); title('Canal H (Matiz)');
        subplot(1,3,2); imshow(S); colormap(gca, 'jet'); title('Canal S (Saturación)');
        subplot(1,3,3); imshow(V); colormap(gca, 'gray'); title('Canal V (Valor)');
    end

    % 4. ANÁLISIS CANAL S (Original)
    gamma_val = 1; 
    S_proc = S .^ gamma_val;
    level_otsu_S = graythresh(S_proc);
    mask_S = imbinarize(S_proc, level_otsu_S);
    
    fprintf('  > S: Umbral Otsu = %.4f\n', level_otsu_S);
    
    % FIGURA 2: Lógica S
    if show_figures(2) == 1
        figure('Name', 'Canal S', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        subplot(2, 2, 1);
        imshow(S); colormap(gca, 'jet'); colorbar;
        title('Canal Saturación (Entrada)');
        
        subplot(2, 2, 2);
        imhist(S); hold on;
        line([level_otsu_S, level_otsu_S], ylim, 'Color', 'r', 'LineWidth', 2);
        text(level_otsu_S, max(ylim)*0.8, sprintf(' Umbral: %.3f', level_otsu_S), 'Color', 'r', 'FontWeight', 'bold');
        title('Histograma + Corte de Otsu');
        xlabel('Intensidad de Saturación'); ylabel('Cantidad de Píxeles');
        
        subplot(2, 2, [3, 4]);
        imshow(mask_S);
        title('Máscara Binaria (antes de limpiar)');
    end
    
    % 5. ANÁLISIS CANAL H (Rescate Morado)
    % Rango Morado/Rosa: 0.68 a 0.88 aprox.
    % Condición de seguridad: S debe ser > 40% del umbral de Otsu calculado antes
    % para no detectar ruido gris de fondo como morado.
    min_sat_H = 0.4 * level_otsu_S;
    mask_H_purple = (H >= 0.68) & (H <= 0.88) & (S > min_sat_H);
    
    fprintf('  > H: Detectando rango morado [0.68 - 0.88]\n');
    
    % FIGURA 3: Lógica H
    if show_figures(3) == 1
        figure('Name', 'Canal H', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        
        subplot(2, 2, 1);
        imshow(H); colormap(gca, 'hsv'); colorbar;
        title('Canal H Completo (Mucho ruido)');
        
        subplot(2, 2, 2);
        % Visualización: Mostramos solo los píxeles con saturación suficiente
        % para ver dónde busca realmente el algoritmo
        H_masked = H; 
        H_masked(S < min_sat_H) = NaN; % Lo ponemos transparente/negro
        imshow(H_masked); colormap(gca, 'hsv');
        title('H (Solo zonas con Sat > min)');
        
        subplot(2, 2, [3, 4]);
        imshow(mask_H_purple);
        title('Máscara H (Morado)');
    end

    % 6. ANÁLISIS CANAL V (Colores Oscuros)
    % Invertimos V para usar Otsu (Lo oscuro se vuelve pico blanco en histograma)
    V_inv = imcomplement(V);
    level_otsu_V = graythresh(V_inv);
    
    % Forzamos detección solo si hay contraste fuerte
    mask_V_dark = imbinarize(V_inv, level_otsu_V);
    
    % DESACTIVAR MASCARA EN CANAL V
    mask_V_dark(:) = 0;
    
    fprintf('  > V: Umbral Otsu (Invertido) = %.4f\n', level_otsu_V);
    
    % FIGURA 4: Lógica V
    if show_figures(4) == 1
        figure('Name', 'Canal V', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);        subplot(1, 3, 1);
        imshow(V_inv); title('V Invertido (Negro=Blanco)');
        
        subplot(1, 3, 2);
        imhist(V_inv); hold on;
        line([level_otsu_V, level_otsu_V], ylim, 'Color', 'r', 'LineWidth', 2);
        title('Hist V_inv + Otsu');
        
        subplot(1, 3, 3);
        imshow(mask_V_dark); title('Máscara V (Oscuros)');
    end

    % 7. FUSIÓN Y MORFOLOGÍA
    % UNIÓN LÓGICA (OR)
    mask_combined = mask_S | mask_H_purple | mask_V_dark;
    
    % LIMPIEZA
    mask_filled = imfill(mask_combined, 'holes');
    se_noise = strel('disk', 3);
    mask_clean = imopen(mask_filled, se_noise);
    mask_final = imclearborder(mask_clean);
    
    % FIGURA 5: Morfología
    if show_figures(5) == 1
        figure('Name', 'Limpieza Morfológica', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        subplot(2, 2, 1);
        imshow(mask_combined);
        title({'PASO 1: Binaria Original', '(Con agujeros y ruido, S + H + V)'});
        subplot(2, 2, 2);
        imshow(mask_filled);
        title({'PASO 2: Relleno de Huecos', '(imfill: recupera studs brillantes)'});
        subplot(2, 2, 3);
        imshow(mask_clean);
        title({'PASO 3: Eliminación de Ruido', '(imopen: borra puntos pequeños)'});
        subplot(2, 2, 4);
        imshow(mask_final);
        title({'PASO 4: Máscara Final', '(imclearborder: quita bordes)'});
    end

    % 8. RESULTADOS Y FILTRADO
    [L, num_inicial] = bwlabel(mask_final, 8);
    stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', 'Perimeter', 'Circularity', 'Eccentricity', 'Image');
    
    if ~isempty(stats)
        all_areas = [stats.Area];
        max_area = max(all_areas);
        
        umbral_area = 0.05 * max_area; 
        all_circ = [stats.Circularity];
        umbral_circ = 0.2; 
        
        valid_idx = find((all_areas > umbral_area) & (all_circ > umbral_circ));
        final_mask_filtered = ismember(L, valid_idx);
        
        stats_final = regionprops(final_mask_filtered, 'Area', 'Centroid', 'BoundingBox', 'Circularity', 'Image');
        num_final = length(stats_final);
    else
        final_mask_filtered = mask_final;
        stats_final = [];
        num_final = 0;
    end
    
    fprintf('  > Objetos Detectados: %d\n', num_final);

    % 9. VISUALIZACIÓN DE RESULTADOS
    if show_figures(6) == 1
        figure('Name', 'Resultados Finales de Segmentación', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        
        % Imagen Original
        subplot(2, 2, 1);
        imshow(I);
        title('Imagen Original');
        
        % Imagen con Corrección y Superposiciones
        subplot(2, 2, 2);
        imshow(I_corrected); hold on;
        title(sprintf('Detección Final: %d Piezas (Corrección Brillo + HSV + Otsu + Morfología)', num_final));
        
        for k = 1:num_final
            % Datos
            c = stats_final(k).Centroid;
            bb = stats_final(k).BoundingBox;
            circ = stats_final(k).Circularity;
            area = stats_final(k).Area;
            
            % Dibujar Bounding Box
            rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 2);
            
            % Marcar Centroide
            plot(c(1), c(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
            
            % Etiquetar
            str_label = sprintf('#%d\nC: %.2f\nA: %d', k, circ, round(area));
            text(bb(1), bb(2)-20, str_label, 'Color', 'yellow', 'FontWeight', 'bold', 'FontSize', 8, 'BackgroundColor', 'k');
        end
        hold off;
        
        % Visualización de Objetos Individuales
        if num_final > 0
            for k = 1:min(num_final, 4)
                subplot(2, 4, 4+k);
                
                % Extraer la pieza con fondo negro
                bb = stats_final(k).BoundingBox;
                img_crop = imcrop(I_corrected, bb);
                
                % Recortar la máscara correspondiente a el objeto
                mask_local = stats_final(k).Image;
                mask_local = imresize(mask_local, [size(img_crop,1), size(img_crop,2)], 'nearest');
                
                % Aplicar máscara para fondo negro
                img_crop_masked = img_crop;
                % Replicar máscara para 3 canales RGB
                mask_3ch = cat(3, mask_local, mask_local, mask_local);
                img_crop_masked(~mask_3ch) = 0; 
                
                imshow(img_crop_masked);
                title(sprintf('Pieza #%d', k));
            end
        end
    end
    pause(0.5);
end