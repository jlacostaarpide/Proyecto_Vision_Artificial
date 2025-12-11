%% EXTRACCIÓN DE CARACTERÍSTICAS DE IMÁGENES PARA CLASSIFICATION LEARNER
clear; clc;

numcarac = 8;  % mismo nº de características que en extractColorFeatures

% Carpeta donde tienes las imágenes nuevas
testFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\Clasificador_sin225';

% Listar imágenes (ajusta extensiones si hace falta)
filesJPG = dir(fullfile(testFolder, '*.jpg'));
filesPNG = dir(fullfile(testFolder, '*.png'));
files = [filesJPG; filesPNG];

fprintf('Se han encontrado %d archivos de imagen en %s\n', numel(files), testFolder);

% Inicializamos contenedores
Xtest      = [];   % matriz de características (N_piezas x numcarac)
names_cell = {};   % nombre de archivo para cada fila de Xtest
piece_idx  = [];   % índice de pieza dentro de la imagen (1,2,...)

for n = 1:numel(files)
    imgPath = fullfile(files(n).folder, files(n).name);
    fprintf('Procesando imagen %d/%d: %s\n', n, numel(files), imgPath);

    % 1) Segmentar la imagen (puede devolver varias piezas)
    try
        [pieces, stats_test, num_pieces, Icorr] = segmentarPiezas2(imgPath); %#ok<ASGLU>
    catch ME
        warning('  >> Error en segmentarPiezas2: %s. Saltando imagen.', ME.message);
        continue;
    end

    if num_pieces == 0 || isempty(pieces)
        warning('  >> No se detectaron piezas en %s. Saltando.', imgPath);
        continue;
    end

    num_pieces = min(num_pieces, numel(pieces));

    % 2) Extraer características de cada pieza
    for k = 1:num_pieces
        Ipiece = pieces{k};
        if isempty(Ipiece)
            continue;
        end

        % Opcional: filtrar piezas casi negras
        if max(Ipiece(:)) < 0.05
            continue;
        end

        feat = extractColorFeatures(Ipiece);   % 1 x numcarac

        % Acumular
        Xtest = [Xtest; feat];          %#ok<AGROW>
        names_cell{end+1} = files(n).name; %#ok<AGROW>
        piece_idx(end+1)   = k;         %#ok<AGROW>
    end
end

fprintf('\nTotal de piezas analizadas: %d\n', size(Xtest,1));

%% 2) CREAR LABEL A PARTIR DEL NOMBRE DE FICHERO
% Suponiendo nombres tipo: 07_225_40_003.jpg -> clase = '07'

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
        labels_str{i} = partes{1};   % primer bloque -> código de clase
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
T.PieceIdx  = piece_idx(:);       % opcional, meta

disp('Ejemplo de primeras filas de T:');
disp(T(1:min(5,height(T)), :));

%% 4) GUARDAR A .MAT PARA USAR EN CLASSIFICATION LEARNER

save('legoFeatures_TEST_todas_sin225_8carac.mat', 'T3');
disp('✔ Archivo legoFeatures_TEST_8carac.mat guardado con tabla T3');



%%
T3 = T;