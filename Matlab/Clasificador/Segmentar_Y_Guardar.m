%% SEGMENTAR UNA CARPETA ENTERA Y GUARDAR SOLO LAS QUE TIENEN 1 PIEZA (n==1)
clear; clc;

% --- Carpetas de origen (donde están las imágenes originales) ---
folders = { ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G04_COD101112" ...
};

% --- Carpeta destino ---
basePath   = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database";
saveFolder = fullfile(basePath, "SEGMENTED");

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% --- Reunir todas las imágenes de todas las carpetas ---
filesAll = [];
for f = 1:numel(folders)
    filesJPG = dir(fullfile(folders{f}, "*.jpg"));
    filesPNG = dir(fullfile(folders{f}, "*.png"));
    filesAll = [filesAll; filesJPG; filesPNG]; %#ok<AGROW>
end

fprintf("Se han encontrado %d imágenes en total.\n", numel(filesAll));
if isempty(filesAll)
    error("No se han encontrado imágenes en las carpetas indicadas.");
end

% --- Procesar una por una ---
nSaved = 0;
nDiscarded = 0;

for i = 1:numel(filesAll)
    imgPath = fullfile(filesAll(i).folder, filesAll(i).name);
    fprintf("\n[%d/%d] Procesando: %s\n", i, numel(filesAll), imgPath);

    % Segmentar (tu función ya descarta internamente si no hay 1 pieza,
    % pero aquí lo comprobamos con n)
    try
        [piece, stats, n] = segmentarPiezas2(imgPath); %#ok<ASGLU>
    catch ME
        warning("  >> Error segmentando %s: %s. Saltando.", imgPath, ME.message);
        continue;
    end

    if n ~= 1 || isempty(piece)
        fprintf("  >> Descartada (n=%d)\n", n);
        nDiscarded = nDiscarded + 1;
        continue;
    end

    % Guardar con el MISMO nombre original (misma extensión)
    [~, baseName, ext] = fileparts(filesAll(i).name);
    outName = sprintf("%s%s", baseName, ext);
    outPath = fullfile(saveFolder, outName);

    % piece es double [0..1], imwrite lo acepta; si prefieres, puedes convertir:
    % imwrite(im2uint8(piece), outPath);
    imwrite(piece, outPath);

    fprintf("  -> Guardada: %s\n", outName);
    nSaved = nSaved + 1;
end

fprintf("\nProceso finalizado.\n");
fprintf("Guardadas   : %d\n", nSaved);
fprintf("Descartadas : %d\n", nDiscarded);
fprintf("Carpeta destino:\n%s\n", saveFolder);
