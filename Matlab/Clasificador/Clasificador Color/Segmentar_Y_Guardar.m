%% SEGMENTAR UNA CARPETA ENTERA Y GUARDAR LA PIEZA (MAYOR) SI EXISTE
% clear; clc;


folders = { ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G01_COD123", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G02_COD456", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G03_COD789", ...
    "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\DB_G04_COD101112" ...
    %"C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\test2"
};

basePath   = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database";
saveFolder = fullfile(basePath, "SEGMENTED_local");

if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

filesAll = [];
for f = 1:numel(folders)
    filesAll = [filesAll; ...
        dir(fullfile(folders{f}, "*.jpg")); ...
        dir(fullfile(folders{f}, "*.png"))]; %#ok<AGROW>
end

fprintf("Se han encontrado %d imágenes en total.\n", numel(filesAll));
if isempty(filesAll)
    error("No se han encontrado imágenes en las carpetas indicadas.");
end

nSaved = 0;
nNoPiece = 0;
nErrors = 0;

for i = 1:numel(filesAll)
    imgPath = fullfile(filesAll(i).folder, filesAll(i).name);
    fprintf("\n[%d/%d] Procesando: %s\n", i, numel(filesAll), imgPath);

    try
        [piece, stats_final, num_final] = segmentarPiezas2(imgPath); %#ok<ASGLU>
    catch ME
        warning("  >> Error segmentando %s: %s. Saltando.", imgPath, ME.message);
        nErrors = nErrors + 1;
        continue;
    end

    if num_final < 1 || isempty(piece)
        fprintf("  >> No se encontró pieza válida (num_final=%d)\n", num_final);
        nNoPiece = nNoPiece + 1;
        continue;
    end

    [~, baseName, ext] = fileparts(filesAll(i).name);
    outPath = fullfile(saveFolder, [baseName ext]);

    piece = piece{1};
    imwrite(piece, outPath);

    fprintf("  -> Guardada: %s%s\n", baseName, ext);
    nSaved = nSaved + 1;
end

fprintf("\nProceso finalizado.\n");
fprintf("Guardadas         : %d\n", nSaved);
fprintf("Sin pieza válida  : %d\n", nNoPiece);
fprintf("Errores           : %d\n", nErrors);
fprintf("Carpeta destino:\n%s\n", saveFolder);
