%% SEGMENTAR TODAS LAS IMÁGENES Y GUARDAR SÓLO LAS QUE TIENEN 1 PIEZA
clear; clc;

% Carpeta origen
inputFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\Clasificador.bak';

% Carpeta destino
saveFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED';
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% Listar imágenes
filesJPG = dir(fullfile(inputFolder, '*.jpg'));
filesPNG = dir(fullfile(inputFolder, '*.png'));
files    = [filesJPG; filesPNG];

fprintf('Se han encontrado %d imágenes en %s\n', numel(files), inputFolder);

for n = 1:numel(files)

    imgPath = fullfile(files(n).folder, files(n).name);
    [~, baseName, ~] = fileparts(files(n).name);

    fprintf('\n[%d/%d] Segmentando: %s\n', n, numel(files), imgPath);

    % === SEGMENTAR ===
    try
        [images_final, stats_final, num_final, Icorr] = segmentarPiezas2(imgPath);
    catch ME
        warning('  >> Error segmentando %s: %s', imgPath, ME.message);
        continue;
    end

    % === FILTRO: sólo guardar si hay 1 pieza ===
    if num_final ~= 1
        fprintf('  >> Imagen descartada (tiene %d piezas)\n', num_final);
        continue;
    end

    % Extraer la pieza
    piece = images_final{1};

    % Guardar con el NOMBRE ORIGINAL (pero como PNG)
    outName = sprintf('%s.png', baseName);   % sin piece01
    outPath = fullfile(saveFolder, outName);

    imwrite(piece, outPath);

    fprintf('  -> Guardada: %s\n', outName);

end

fprintf('\nProceso finalizado. Archivos guardados en:\n%s\n', saveFolder);
