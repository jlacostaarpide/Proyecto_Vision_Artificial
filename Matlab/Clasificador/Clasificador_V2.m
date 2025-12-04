%% CLASIFICADOR PIEZAS LEGO - ENTRENAMIENTO MASIVO (CLASE = PRIMEROS 2 DÍGITOS)
% Usa segmentarPiezas2(nombre_imagen) y extractColorFeatures(I)
% Clase = los dos primeros dígitos del nombre: 01_xxx_xx_001.jpg -> clase "01"

clear; close all; clc;

%% 0) RUTAS BASE (AJUSTA ESTO A TU PC)
basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\";

% Carpetas donde están las imágenes (G01, G02, G03...)
folders = { ...
    %fullfile(basePath, "DB_G01_COD123"), ...
    %fullfile(basePath, "DB_G02_COD456"), ...
    fullfile(basePath, "DB_G03_COD789") ...
    };

% Códigos de clase que aparecen como PRIMER bloque en el nombre de fichero
% Ejemplo: 01_270_70_003.jpg -> código de clase = '01'
% ADAPTA esta lista a los códigos que realmente tengas:
codigoClases = { ...
    %'01','02','03' ...
    %,'04','05','06', ...
    '07','08','09', ...
    %,'10','11','12' ...
    };

% Usamos directamente los códigos como nombres de clase (más natural)
classNames = codigoClases;
numClases  = numel(codigoClases);

%% 1) RECORRER TODAS LAS IMÁGENES, SEGMENTAR (Y no HACER DATA AUGMENTATION)

%angles = 0:45:315;     % 8 rotaciones: 0,45,...,315
images_train = {};
labels_cell  = {};

for fidx = 1:numel(folders)
    thisFolder = folders{fidx};
    filesJPG = dir(fullfile(thisFolder, '*.jpg'));
    filesPNG = dir(fullfile(thisFolder, '*.png'));
    files = [filesJPG; filesPNG];   % por si mezclas formatos
    
    fprintf('Procesando carpeta %s: %d imágenes\n', thisFolder, numel(files));
    
    for n = 1:numel(files)
        nombre_imagen = fullfile(files(n).folder, files(n).name);
        fprintf('  - %s\n', nombre_imagen);
        
        % ====== 1. OBTENER CÓDIGO DE CLASE DEL NOMBRE DE FICHERO ======
        [~, baseName, ~] = fileparts(files(n).name);   % ej: '01_270_70_003'
        partes = split(baseName, '_');
        if numel(partes) < 2
            warning('    >> Nombre raro (sin "_"): %s. Saltando.', baseName);
            continue;
        end
        
        codStr = partes{1};   % *** AQUÍ EL CAMBIO CLAVE: PRIMER BLOQUE ***
        % codStr debería ser '01', '02', '03', etc.
        
        % Buscar ese código en la tabla de códigos
        idxClase = find(strcmp(codigoClases, codStr));
        if isempty(idxClase)
            warning('    >> Código de clase %s no está en codigoClases. Saltando imagen.', codStr);
            continue;
        end
        
        etiquetaClase = classNames{idxClase};   % p.ej. '01'
        
        % ====== 2. SEGMENTAR ESTA IMAGEN ======
        try
            [images_final, stats_final, num_final, I_corrected] = segmentarPiezas2(nombre_imagen); %#ok<ASGLU>
        catch ME
            warning('    >> Error en segmentarPiezas2: %s. Saltando imagen.', ME.message);
            continue;
        end
        
        if num_final == 0 || isempty(images_final)
            warning('    >> No se detectaron piezas en %s. Saltando.', nombre_imagen);
            continue;
        end
        
        % Asegurar que images_final tiene num_final celdas
        num_final = min(num_final, numel(images_final));


         %% === GUARDAR PIEZAS SEGMENTADAS ===
    saveFolder = fullfile(basePath, "SEGMENTED");
    if ~exist(saveFolder, 'dir')
        mkdir(saveFolder);
    end

    for k = 1:num_final
        piece = images_final{k};
        if isempty(piece), continue; end

        outName = sprintf('%s_piece%02d.png', baseName, k);
        outPath = fullfile(saveFolder, outName);

        imwrite(piece, outPath);
    end
    %% =============================================================
        
  % ====== 3. PARA CADA PIEZA DETECTADA: SIN DATA AUGMENTATION ======
        for k = 1:num_final
            Ibase = images_final{k};
            if isempty(Ibase)
                continue;
            end
            
            % Opcional: eliminar piezas que hayan quedado casi negras
            if max(Ibase(:)) < 0.05
                continue;
            end
            
            images_train{end+1} = Ibase;          %#ok<SAGROW>
            labels_cell{end+1}  = etiquetaClase;  %#ok<SAGROW>
        end
    end
end

fprintf('\nTotal piezas (para entrenamiento): %d\n', numel(images_train));

% Convertimos etiquetas a categorical
labels_train = categorical(labels_cell(:));

disp('Distribución de clases en el entrenamiento:');
tabulate(labels_train)

%% 2) EXTRAER CARACTERÍSTICAS PARA TODAS LAS IMÁGENES DE ENTRENAMIENTO

numTrain = numel(images_train);
Xtrain   = zeros(numTrain, 6);   % 6: mean/std de H,S,V

for i = 1:numTrain
    I = images_train{i};
    Xtrain(i,:) = extractColorFeatures(I);
end
Ytrain = labels_train;

% %% 3) ENTRENAR CLASIFICADOR k-NN
% 
% Mdl = fitcknn(Xtrain, Ytrain, ...
%               'NumNeighbors', 3, ...
%               'Standardize', true);

%% 3) CREAR TABLA PARA CLASSIFICATION LEARNER

featureNames = {'H_mean','H_std','S_mean','S_std','V_mean','V_std'};

T = array2table(Xtrain, 'VariableNames', featureNames);
T.Label = Ytrain;   % columna de respuesta (categorical)

% Guardar por si quieres cargarla otro día
save('legoFeatures_TrainingSet.mat','T');

disp('✔ Tabla T creada en workspace con features y Label');

% %% 4) GUARDAR MODELO ENTRENADO
% 
% save('legoModel_porCodigoTercerBloque.mat','Mdl','codigoClases','classNames');
% disp("✔ Modelo entrenado y guardado como legoModel_porCodigoTercerBloque.mat");

