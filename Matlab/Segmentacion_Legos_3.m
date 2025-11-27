%% SEGMENTACIÓN DE PIEZAS LEGO

clear; close all; clc;


% Iñaki:
addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\matlab_imagen\Matlab - Imagen")
addpath("C:\Nextcloud\Escritorio\UPNA\Doble Master - 1º Semestre (Septiembre 2025)\Procesado de Señales Multimedia\Matlab\legocodes")
addpath("C:\Users\Iñaki Janices\Documentos\Github\ProyectoPSM\Database")

% Juan:
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\tests");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456");
addpath("C:\Users\jlaco\OneDrive\Escritorio\Académico\UPNA\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789");

%% 1. CARGA DE LA IMAGEN
nombre_imagen = 'IMG_7650.jpg';
nombre_imagen = '08_270_70_004.jpg';

I = imread(nombre_imagen);
I_double = im2double(I);
fprintf('Imagen cargada correctamente:\n');
figure; imshow(I_double);

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
subplot(1,3,1); imshow(H,[]); title('Canal H (Matiz)');
subplot(1,3,2); imshow(S,[]); title('Canal S (Saturación)');
subplot(1,3,3); imshow(V); title('Canal V (Valor)');

%% 4. SEGMENTACIÓN COMBINANDO H Y S

% --- 4.1. Estimar el color del fondo (mesa) ---
% Consideramos "fondo" los píxeles de saturación baja
mask_fondo = S < 0.25;          % umbral bajo de saturación
H_fondo = mean(H(mask_fondo), 'all');   % tono medio del fondo

% Distancia de cada píxel al tono del fondo
dist_H = abs(H - H_fondo);

% Máscara 1: píxeles cuyo tono es distinto del fondo
% (las piezas tienen un matiz muy distinto al de la mesa)
umbral_distH = 0.06;            % puedes probar 0.05–0.10
mask_H = dist_H > umbral_distH;
%figure; imshow(mask_H);

% --- 4.2. Máscara basada en saturación (como antes) ---
level_otsu_S = graythresh(S);
fprintf('Umbral de Otsu para S: %.4f\n', level_otsu_S);
mask_S = imbinarize(S, level_otsu_S);
%figure; imshow(mask_S);

% --- 4.3. Máscara final: unión de ambas ---
% Si un píxel tiene suficiente saturación O es cromáticamente distinto
% del fondo, lo consideramos pieza.
mask = mask_H | mask_S;

%% 5. PROCESAMIENTO MORFOLÓGICO (LIMPIEZA Y RECONSTRUCCIÓN)

% Cerrar pequeños huecos y “puentes” entre bloques
se_bridge = strel('disk', 4);
mask = imclose(mask, se_bridge);

% Rellenar huecos interiores
mask = imfill(mask, 'holes');

% Pequeña apertura para eliminar ruido muy pequeño
se_noise = strel('disk', 2);
mask = imopen(mask, se_noise);

% Eliminar objetos diminutos
mask = bwareaopen(mask, 500);

% Si en tus imágenes no esperas piezas tocando el borde, puedes mantenerlo.
% Si sí pueden tocar el borde, mejor comentar esta línea.
mask_final = imclearborder(mask);

%% 6. ANÁLISIS DE COMPONENTES CONEXAS Y EXTRACCIÓN DE CARACTERÍSTICAS
[L, num_inicial] = bwlabel(mask_final, 8);
stats = regionprops(L, 'Area', 'Centroid', 'BoundingBox', ...
                       'Perimeter', 'Circularity', 'Eccentricity', 'Image');

%% 7. FILTRADO DE OBJETOS
if ~isempty(stats)
    all_areas = [stats.Area];
    max_area = max(all_areas);

    % Filtro por área relativa
    umbral_area = 0.05 * max_area;
    all_circ = [stats.Circularity];

    % OJO: las piezas LEGO tienen formas bastante irregulares
    % Baja el umbral de circularidad o elimínalo.
    umbral_circ = 0.05;
    valid_idx = find((all_areas > umbral_area) & (all_circ > umbral_circ));

    final_mask_filtered = ismember(L, valid_idx);

    stats_final = regionprops(final_mask_filtered, 'Area', 'Centroid', ...
        'BoundingBox', 'Circularity', 'Image');
    num_final = length(stats_final);
else
    final_mask_filtered = mask_final;
    stats_final = "";
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