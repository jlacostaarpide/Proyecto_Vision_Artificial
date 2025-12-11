%% LIMPIAR SEGMENTADAS: QUEDARME SOLO CON ORIGINALES DE 1 PIEZA

segFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_Una_Pieza';

% Listar imágenes segmentadas (ajusta extensiones si hace falta)
filesPNG = dir(fullfile(segFolder, '*.png'));
filesJPG = dir(fullfile(segFolder, '*.jpg'));
files    = [filesPNG; filesJPG];

fprintf('Se han encontrado %d archivos en %s\n', numel(files), segFolder);

if isempty(files)
    error('No hay archivos en la carpeta SEGMENTED.');
end

% 1) Obtener el "baseName" original (antes de "_piece") de cada archivo
names      = {files.name};
baseNames  = cell(size(names));

for i = 1:numel(names)
    [~, bn, ~] = fileparts(names{i});      % ej: '09_225_40_003_piece01'
    partes = split(bn, '_piece');
    baseNames{i} = partes{1};             % ej: '09_225_40_003'
end

% 2) Contar cuántas piezas tiene cada baseName
[uniqueBases, ~, idx] = unique(baseNames);   % idx indica a qué uniqueBases pertenece cada archivo
counts = accumarray(idx(:), 1);             % nº de piezas por baseName

% 3) Recorrer cada baseName y borrar las que tienen más de 1 pieza
numDeleted = 0;
numKept    = 0;

for u = 1:numel(uniqueBases)
    thisBase = uniqueBases{u};
    c        = counts(u);

    % Archivos asociados a este baseName
    mask = (idx == u);
    theseFiles = files(mask);

    if c > 1
        % Borrar TODAS las piezas de este original
        fprintf('Base %s tiene %d piezas -> BORRANDO todas\n', thisBase, c);
        for k = 1:numel(theseFiles)
            fileToDelete = fullfile(theseFiles(k).folder, theseFiles(k).name);
            delete(fileToDelete);
            numDeleted = numDeleted + 1;
        end
    else
        % Se queda tal cual (1 sola pieza)
        fprintf('Base %s tiene 1 pieza -> SE MANTIENE\n', thisBase);
        numKept = numKept + 1;
    end
end

fprintf('\nResumen:\n');
fprintf('  Imágenes originales con 1 sola pieza (se mantienen): %d\n', numKept);
fprintf('  Archivos de piezas borrados (originales con >1 pieza): %d\n', numDeleted);
