%% SEGMENTACIÓN DE PIEZAS LEGO

clear; close all; clc;

addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\matlab_imagen\Matlab - Imagen")
addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\legocodes")

%% 1. CARGA DE LA IMAGEN
% nombre_imagen = 'IMG_7647.jpg';
nombre_imagen = 'IMG_7643.jpg';
nombre_imagen = '4_legos.jpg';

I = imread(nombre_imagen);
I_double = im2double(I);
fprintf('Imagen cargada correctamente: %s\n', nombre_imagen);

%% 2. PRE-PROCESAMIENTO: CORRECCIÓN DE BRILLO
% Pendiente de implementar
I_corrected = I_double;

% figure('Name', 'Pre-procesamiento', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
% subplot(1,3,1); imshow(I); title('Imagen Original');
% subplot(1,3,2); imshow(background); title('Estimación del Fondo (Iluminación)');
% subplot(1,3,3); imshow(I_corrected); title('Imagen con Iluminación Corregida');

%% 3. TRANSFORMACIÓN A HSV
I_hsv = rgb2hsv(I_corrected);

H = I_hsv(:,:,1);
S = I_hsv(:,:,2);
V = I_hsv(:,:,3);

figure('Name', 'Análisis de Canales HSV', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
subplot(1,3,1); imshow(H); colormap(gca, 'hsv'); title('Canal H (Matiz)');
subplot(1,3,2); imshow(S); colormap(gca, 'jet'); title('Canal S (Saturación)');
subplot(1,3,3); imshow(V); colormap(gca, 'gray'); title('Canal V (Valor)');

%% 4. SEGMENTACIÓN OTSU EN CANAL S
% Maximizar la varianza inter-clase sobre el canal S.
level_otsu = graythresh(S);
fprintf('Umbral de Otsu calculado para Saturación: %.4f\n', level_otsu);

% Binarización
mask = imbinarize(S, level_otsu);

%% 5. PROCESAMIENTO MORFOLÓGICO (LIMPIEZA Y RECONSTRUCCIÓN)
% Relleno de huecos:
%    Corregir los brillos especulares (blancos) que tienen S=0
mask_filled = imfill(mask, 'holes');

% Apertura Morfológica:
%    Elimina ruido "sal" (pequeños puntos blancos) 
se_noise = strel('disk', 3);
mask_clean = imopen(mask_filled, se_noise);

% Eliminación de bordes:
%    Elimina objetos que tocan el borde de la imagen
mask_final = imclearborder(mask_clean);

%% 6. ANÁLISIS DE COMPONENTES CONEXAS Y EXTRACCIÓN DE CARACTERÍSTICAS
% Etiquetado con conectividad-8 (para agrupar píxeles diagonales)
[L, num_inicial] = bwlabel(mask_final, 8);

% Extracción de propiedades geométricas
stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', 'Perimeter', 'Circularity', 'Eccentricity');

%% 7. FILTRADO DE OBJETOS
if ~isempty(stats)
    % Convertir áreas a vector
    all_areas = [stats.Area];
    max_area = max(all_areas);
    
    % Criterio 1: Filtrado por Área Relativa
    % Se conservan objetos con al menos el 5% del área del objeto más grande.
    % Esto hace el filtro robusto al zoom/resolución de la imagen.
    umbral_area = 0.05 * max_area; 
    
    % Criterio 2: Filtrado por Circularidad
    % Elimina formas muy irregulares.
    % Lego cuadrado ~0.78.
    all_circ = [stats.Circularity];
    umbral_circ = 0.2; % Valor conservador para no borrar piezas complejas
    
    valid_idx = find((all_areas > umbral_area) & (all_circ > umbral_circ));
    final_mask_filtered = ismember(L, valid_idx);
    
    % Recalcular estadísticas
    stats_final = regionprops(final_mask_filtered, 'Area', 'Centroid', 'BoundingBox', 'Circularity', 'Image');
    num_final = length(stats_final);
else
    final_mask_filtered = mask_final;
    stats_final ="";
    num_final = 0;
end

fprintf('Objetos detectados inicialmente: %d\n', num_inicial);
fprintf('Objetos finales tras filtrado: %d\n', num_final);

%% 8. VISUALIZACIÓN DE RESULTADOS
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