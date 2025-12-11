%% SEGMENTAR TODAS LAS IMÁGENES DE angulos225 Y GUARDAR PIEZAS EN SEGMENTED
clear; clc;

% Carpeta origen (imágenes completas)
inputFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\angulos225';

% Carpeta destino (piezas segmentadas)
basePath   = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database';
saveFolder = fullfile(basePath, 'SEGMENTED');   % misma carpeta que ya usabas

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% Listar imágenes (ajusta extensiones si hace falta)
filesJPG = dir(fullfile(inputFolder, '*.jpg'));
filesPNG = dir(fullfile(inputFolder, '*.png'));
files    = [filesJPG; filesPNG];

fprintf('Se han encontrado %d imágenes en %s\n', numel(files), inputFolder);

for n = 1:numel(files)
    imgPath = fullfile(files(n).folder, files(n).name);
    fprintf('\n[%d/%d] Segmentando: %s\n', n, numel(files), imgPath);

    % Nombre base sin extensión: ej. '09_225_40_003'
    [~, baseName, ~] = fileparts(files(n).name);

    % SEGMENTAR
    try
        [images_final, stats_final, num_final, Icorr] = segmentarPiezas2(imgPath); %#ok<ASGLU>
    catch ME
        warning('  >> Error en segmentarPiezas2: %s. Saltando imagen.', ME.message);
        continue;
    end

    if num_final == 0 || isempty(images_final)
        warning('  >> No se detectaron piezas en %s. Saltando.', imgPath);
        continue;
    end

    num_final = min(num_final, numel(images_final));

    % GUARDAR CADA PIEZA
    for k = 1:num_final
        piece = images_final{k};
        if isempty(piece)
            continue;
        end

        % Nombre: 09_225_40_003_piece01.png (mismo formato que antes)
        outName = sprintf('%s_piece%02d.png', baseName, k);
        outPath = fullfile(saveFolder, outName);

        imwrite(piece, outPath);
        fprintf('   -> Guardada pieza %02d: %s\n', k, outName);
    end
end

fprintf('\nProceso de segmentación completado. Piezas guardadas en:\n%s\n', saveFolder);
