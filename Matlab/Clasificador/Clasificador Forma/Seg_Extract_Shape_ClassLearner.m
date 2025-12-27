%% EXTRACCIÓN DE CARACTERÍSTICAS DE FORMA DE PIEZAS YA SEGMENTADAS (PARA CLASSIFICATION LEARNER)
clear; clc;

numcarac = 24;  % nº de características que devuelve extractShapeFeatures (NUEVA versión sin FD)
% Códigos de clase válidos
validCodes = {'09','12'};

% Carpeta donde tienes las IMÁGENES YA SEGMENTADAS (una pieza por archivo)
testFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';

% Listar imágenes
filesJPG = dir(fullfile(testFolder, '*.jpg'));
filesPNG = dir(fullfile(testFolder, '*.png'));
files    = [filesJPG; filesPNG];

fprintf('Se han encontrado %d archivos de imagen en %s\n', numel(files), testFolder);

% Filtrar solo archivos cuyo nombre empiece por 09 o 12
isValid = false(numel(files),1);

for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    partes = split(baseName, '_');
    if ~isempty(partes) && ismember(partes{1}, validCodes)
        isValid(i) = true;
    end
end

files = files(isValid);

fprintf('Tras filtrar por código (09,12): %d imágenes válidas\n', numel(files));

% Inicializamos contenedores
Xtest      = zeros(0, numcarac);   % prealocado "vacío" con numcarac columnas
names_cell = {};
piece_idx  = [];

for n = 1:numel(files)
    imgPath = fullfile(files(n).folder, files(n).name);
    fprintf('Procesando pieza %d/%d: %s\n', n, numel(files), files(n).name);

    % Leer la pieza (ya segmentada)
    Ipiece = imread(imgPath);

    if isempty(Ipiece)
        warning('  >> Imagen vacía: %s. Saltando.', imgPath);
        continue;
    end

    % (Opcional) filtro por si algún archivo está “casi negro”
    if max(Ipiece(:)) < 0.02
        warning('  >> Pieza casi negra en %s. Saltando.', imgPath);
        continue;
    end

    % Extraer características de forma (1 x 9)  <-- usa TU extractShapeFeatures actualizado
    feat = extractShapeFeatures(Ipiece);

    % Acumular
    Xtest(end+1, :) = feat;                 %#ok<AGROW>
    names_cell{end+1} = files(n).name;      %#ok<AGROW>
    piece_idx(end+1)   = 1;                 %#ok<AGROW>
end

fprintf('\nTotal de piezas analizadas: %d\n', size(Xtest,1));

%% 2) CREAR LABEL A PARTIR DEL NOMBRE DE FICHERO
% Suponiendo nombres tipo: 12_225_40_003.png -> clase = '12'

numSamples = numel(names_cell);
labels_str = cell(numSamples,1);

for i = 1:numSamples
    baseName = erase(names_cell{i}, {'.jpg','.png','.bmp','.jpeg'});
    partes   = split(baseName, '_');

    if numel(partes) < 1
        warning('Nombre raro: %s. Pongo label "unknown".', names_cell{i});
        labels_str{i} = 'unknown';
    else
        labels_str{i} = partes{1};   % primer bloque -> código de clase
    end
end

Label = categorical(labels_str);

%% 3) MONTAR TABLA PARA CLASSIFICATION LEARNER

% --- ANTES (10 feats) ---
% featNames = { ...
%  'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
%  'SkelLenNorm','SkelEndpoints','SkelBranchpoints'};

featNames = { ...
 'AreaNorm','PerimNorm','Circularity','Extent','Solidity','Eccentricity','AspectRatio','EulerNumber', ...
 'HolesCount','HolesAreaFrac','SkelLenNorm','SkelEndpoints','SkelBranchpoints', ...
 'ProjV_peaks','ProjH_peaks','ProjV_entropy','ProjH_entropy', ...
 'GridOccFrac_3x3','GridOccGini_3x3','GridOccDiagDiff_3x3', ...
 'StudsCount','StudsCountNormArea','StudsMeanRadius','StudsRadiusStd' ...
};

F_a = array2table(Xtest, 'VariableNames', featNames);
F_a.Label     = Label;
F_a.FileName  = names_cell(:);
F_a.PieceIdx  = piece_idx(:);

disp('Ejemplo de primeras filas de F_a:');
disp(F_a(1:min(5,height(F_a)), :));

%% 4) GUARDAR A .MAT PARA USAR EN CLASSIFICATION LEARNER
save('legoFeatures_TRAIN_shape_24carac_F_amarillas.mat', 'F_a');
disp('✔ Archivo guardado: legoFeatures_TRAIN_shape_24carac_F_amarillas.mat');
