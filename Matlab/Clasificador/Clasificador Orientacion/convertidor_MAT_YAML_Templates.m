%% convert_templates_to_opencv_yaml.m
% Convierte tpl_XX_YYY_PP.mat (con struct "tpl") a YAML OpenCV FileStorage.
% Resultado: tpl_XX_YYY_PP.yml con la matriz T como !!opencv-matrix.

clear; clc;

inFolder  = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador Orientacion\Templates";
outFolder = inFolder;   % puedes poner otra carpeta si quieres

if ~isfolder(inFolder)
    error("No existe la carpeta: %s", inFolder);
end
if ~isfolder(outFolder)
    mkdir(outFolder);
end

files = dir(fullfile(inFolder, "tpl_*.mat"));
if isempty(files)
    error("No hay archivos tpl_*.mat en: %s", inFolder);
end

fprintf("Encontrados %d .mat\n", numel(files));

for i = 1:numel(files)
    matPath = fullfile(files(i).folder, files(i).name);

    S = load(matPath);
    if ~isfield(S, "tpl")
        fprintf("SKIP (no existe 'tpl'): %s\n", files(i).name);
        continue;
    end

    tpl = S.tpl;

    % --- Validaciones mínimas ---
    if ~isfield(tpl, "T") || isempty(tpl.T)
        fprintf("SKIP (tpl.T vacío): %s\n", files(i).name);
        continue;
    end
    T = double(tpl.T);

    if ~isfield(tpl, "size") || isempty(tpl.size)
        tpl.size = size(T,1);
    end
    if ~isfield(tpl, "orientation"), tpl.orientation = NaN; end
    if ~isfield(tpl, "pitch"),       tpl.pitch       = NaN; end
    if ~isfield(tpl, "code"),        tpl.code        = "";  end

    % Nombre de salida: mismo que .mat pero .yml
    [~, baseName, ~] = fileparts(files(i).name);
    ymlPath = fullfile(outFolder, baseName + ".yml");

    % OpenCV espera data en orden fila-major.
    % MATLAB es columna-major, así que hacemos reshape(T',1,[]) para linealizar por filas.
    dataRowMajor = reshape(T.', 1, []);

    % --- Escribir YAML OpenCV ---
    fid = fopen(ymlPath, "w");
    if fid < 0
        error("No puedo escribir: %s", ymlPath);
    end

    fprintf(fid, "%%YAML:1.0\n");
    fprintf(fid, "---\n");

    % code: lo guardo como string (si en tu tpl.code es string/char)
    codeStr = tpl.code;
    if isstring(codeStr), codeStr = char(codeStr); end
    if isnumeric(codeStr), codeStr = sprintf("%02d", codeStr); end
    if isempty(codeStr), codeStr = ""; end

    fprintf(fid, 'code: \"%s\"\n', codeStr);
    fprintf(fid, "yaw: %d\n", round(tpl.orientation));
    fprintf(fid, "pitch: %d\n", round(tpl.pitch));
    fprintf(fid, "size: %d\n", round(tpl.size));

    fprintf(fid, "T: !!opencv-matrix\n");
    fprintf(fid, "   rows: %d\n", size(T,1));
    fprintf(fid, "   cols: %d\n", size(T,2));
    fprintf(fid, "   dt: d\n");
    fprintf(fid, "   data: [");

    % Volcado con buena precisión
    % (usa %.17g para no perder precisión en doubles)
    for k = 1:numel(dataRowMajor)
        if k == numel(dataRowMajor)
            fprintf(fid, "%.17g", dataRowMajor(k));
        else
            fprintf(fid, "%.17g, ", dataRowMajor(k));
        end
    end

    fprintf(fid, "]\n");
    fclose(fid);

    fprintf("OK -> %s\n", ymlPath);
end

fprintf("\nDONE. YAMLs creados en: %s\n", outFolder);
