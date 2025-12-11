%% EXTRACCIÓN DE CARACTERÍSTICAS DE PIEZAS YA SEGMENTADAS (PARA CLASSIFICATION LEARNER)
clear; clc;

numcarac = 8;  % mismo nº de características que en extractColorFeatures

% Carpeta donde tienes las IMÁGENES YA SEGMENTADAS (una pieza por archivo)
testFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED';

% Listar imágenes (ajusta extensiones si hace falta)
filesJPG = dir(fullfile(testFolder, '*.jpg'));
filesPNG = dir(fullfile(testFolder, '*.png'));
files    = [filesJPG; filesPNG];

fprintf('Se han encontrado %d archivos de imagen en %s\n', numel(files), testFolder);

% Inicializamos contenedores
Xtest      = [];   % matriz de características (N_piezas x numcarac)
names_cell = {};   % nombre de archivo para cada fila de Xtest
piece_idx  = [];   % índice de pieza (aquí siempre 1, pero lo dejamos por compatibilidad)

for n = 1:numel(files)
    imgPath = fullfile(files(n).folder, files(n).name);
    fprintf('Procesando pieza %d/%d: %s\n', n, numel(files), imgPath);

    % Leer la pieza (ya segmentada)
    Ipiece = imread(imgPath);

    if isempty(Ipiece)
        warning('  >> Imagen vacía: %s. Saltando.', imgPath);
        continue;
    end

    % Opcional: filtrar piezas casi negras
    if max(Ipiece(:)) < 0.05
        warning('  >> Pieza casi negra en %s. Saltando.', imgPath);
        continue;
    end

    % Extraer características (1 x numcarac)
    feat = extractColorFeatures(Ipiece);

    % Acumular
    Xtest = [Xtest; feat];            %#ok<AGROW>
    names_cell{end+1} = files(n).name; %#ok<AGROW>
    piece_idx(end+1)   = 1;           %#ok<AGROW>  % siempre 1, una pieza por archivo
end

fprintf('\nTotal de piezas analizadas: %d\n', size(Xtest,1));

%% 2) CREAR LABEL A PARTIR DEL NOMBRE DE FICHERO
% Suponiendo nombres tipo: 07_225_40_003.png -> clase = '07'

numSamples = numel(names_cell);
labels_str = cell(numSamples,1);

for i = 1:numSamples
    % Quitar extensión por si acaso
    baseName = erase(names_cell{i}, {'.jpg','.png','.bmp','.jpeg'});
    partes   = split(baseName, '_');
    if numel(partes) < 1
        warning('Nombre raro: %s. Pongo label "unknown".', names_cell{i});
        labels_str{i} = 'unknown';
    else
        labels_str{i} = partes{1};   % primer bloque -> código de clase ('01','02',...)
    end
end

Label = categorical(labels_str);

%% 3) MONTAR TABLA PARA CLASSIFICATION LEARNER

featNames = {'H_mean_circ', 'H_var_circ', ...
             'S_median', 'S_IQR', ...
             'V_median', 'V_IQR', ...
             'S_mean', 'V_mean'};

T = array2table(Xtest, 'VariableNames', featNames);
T.Label     = Label;              % columna respuesta
T.FileName  = names_cell(:);      % opcional, meta
T.PieceIdx  = piece_idx(:);       % opcional, meta (aquí siempre 1)

disp('Ejemplo de primeras filas de T:');
disp(T(1:min(5,height(T)), :));

%% 4) GUARDAR A .MAT PARA USAR EN CLASSIFICATION LEARNER

save('legoFeatures_TRAIN_todas_8carac.mat', 'T'); 
disp('✔ Archivo guardado');
