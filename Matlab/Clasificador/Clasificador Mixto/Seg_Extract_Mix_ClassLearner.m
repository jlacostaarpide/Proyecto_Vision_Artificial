%% EXTRACCIÓN DE CARACTERÍSTICAS DE FORMA DE PIEZAS YA SEGMENTADAS (PARA CLASSIFICATION LEARNER)
clear; clc;

numcarac = 22;  % nº de características que devuelve extractShapeFeatures
% Códigos de clase válidos
validCodes = {'01','02','03','04','05','06','07','08','09','10','11','12'};

% Carpeta donde tienes las IMÁGENES YA SEGMENTADAS (una pieza por archivo)
testFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';

% Listar imágenes
filesJPG = dir(fullfile(testFolder, '*.jpg'));
filesPNG = dir(fullfile(testFolder, '*.png'));
files    = [filesJPG; filesPNG];

fprintf('Se han encontrado %d archivos de imagen en %s\n', numel(files), testFolder);

% Filtrar solo archivos cuyo nombre empiece por 03, 06, 09 o 12
isValid = false(numel(files),1);

for i = 1:numel(files)
    [~, baseName, ~] = fileparts(files(i).name);
    partes = split(baseName, '_');
    if ~isempty(partes) && ismember(partes{1}, validCodes)
        isValid(i) = true;
    end
end

files = files(isValid);

fprintf('Tras filtrar por código (03,06,09,12): %d imágenes válidas\n', numel(files));

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

    % Extraer características de forma (1 x 15)
    feat = extractColorShapeFeatures(Ipiece);

    % Acumular
    Xtest(end+1, :) = feat;                 %#ok<AGROW>
    names_cell{end+1} = files(n).name;      %#ok<AGROW>
    piece_idx(end+1)   = 1;                 %#ok<AGROW>
end

fprintf('\nTotal de piezas analizadas: %d\n', size(Xtest,1));

%% 2) CREAR LABEL A PARTIR DEL NOMBRE DE FICHERO
% Suponiendo nombres tipo: 07_225_40_003.png -> clase = '07'

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

featNames = { ...
 'H_mean_circ','H_var_circ','S_median','S_IQR','V_median','V_IQR','S_mean','V_mean', ...
 'Circularity','AspectRatio','Extent','Solidity','Convexity','Eccentricity','EulerNumber', ...
 'SkelLenNorm','SkelEndpoints','SkelBranchpoints','FD2','FD3','FD4','FD5' };


M = array2table(Xtest, 'VariableNames', featNames);
M.Label     = Label;
M.FileName  = names_cell(:);
M.PieceIdx  = piece_idx(:);

disp('Ejemplo de primeras filas de M:');
disp(M(1:min(5,height(M)), :));

%% 4) GUARDAR A .MAT PARA USAR EN CLASSIFICATION LEARNER
save('legoFeatures_TRAIN_color_shape_22carac.mat', 'M');
disp('✔ Archivo guardado: legoFeatures_TRAIN__color_shape_22carac.mat');



