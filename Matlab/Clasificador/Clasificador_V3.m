%% CLASIFICADOR PIEZAS LEGO - ENTRENAMIENTO MASIVO (CLASE = PRIMEROS 2 DÍGITOS)
% Usa segmentarPiezas2(nombre_imagen) y extractColorFeatures(I)
% Clase = los dos primeros dígitos del nombre: 01_xxx_xx_001.jpg -> clase "01"

clear; close all; clc;

numcarac = 8; % tiene q coincidir con el de extractColorFeatures

%% 0) RUTAS BASE (AJUSTA ESTO A TU PC)
basePath = "C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\";

% Carpetas donde están las imágenes (G01, G02, G03...)
folders = { ...
    fullfile(basePath, "DB_G01_COD123"), ...
    fullfile(basePath, "DB_G02_COD456"), ...
    fullfile(basePath, "DB_G03_COD789") ...
    };

% Códigos de clase que aparecen como PRIMER bloque en el nombre de fichero
% Ejemplo: 01_270_70_003.jpg -> código de clase = '01'
% ADAPTA esta lista a los códigos que realmente tengas:
codigoClases = { ...
    '01','02','03', ...
    '04','05','06', ...
    '07','08','09', ...
    %'10','11','12' ...
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
Xtrain   = zeros(numTrain, numcarac); %Tiene que coincidir con extractColorFeatures  

for i = 1:numTrain
    I = images_train{i};
    Xtrain(i,:) = extractColorFeatures(I);
end
Ytrain = labels_train;

%% 3) PARTIR EN 80% TRAIN / 20% TEST (ESTRATIFICADO POR CLASE)

rng(1);   % para reproducibilidad

cv = cvpartition(Ytrain,'HoldOut',0.2);   % 20% test

idxTrain = training(cv);   % índices lógicos de train
idxTest  = test(cv);       % índices lógicos de test

X_tr = Xtrain(idxTrain,:);   % características train
Y_tr = Ytrain(idxTrain);     % etiquetas train

X_te = Xtrain(idxTest,:);    % características test
Y_te = Ytrain(idxTest);      % etiquetas test

fprintf('Tamaño train: %d muestras\n', size(X_tr,1));
fprintf('Tamaño test : %d muestras\n', size(X_te,1));

%% 4) ENTRENAR CLASIFICADOR k-NN SOLO CON EL 80% TRAIN

Mdl = fitcknn(X_tr, Y_tr, ...
              'NumNeighbors', 5, ...   % o 3, lo que te haya ido mejor
              'Standardize', true);

%% 5) EVALUAR EN EL 20% TEST

Y_pred = predict(Mdl, X_te);

% Matriz de confusión
[cm, order] = confusionmat(Y_te, Y_pred);

figure;
confusionchart(cm, order);
title('Matriz de confusión (20% test)');

% Exactitud global
acc = sum(diag(cm)) / sum(cm(:));
fprintf('Accuracy global en test: %.2f %%\n', 100*acc);

% Exactitud por clase
acc_por_clase = diag(cm) ./ sum(cm,2);
tabla_acc = table(order, acc_por_clase, ...
                  'VariableNames', {'Clase','Accuracy'});
disp(tabla_acc);

%% 6) (OPCIONAL) CREAR TABLA PARA CLASSIFICATION LEARNER CON TODO EL CONJUNTO

feat = {'H_mean_circ', 'H_var_circ', ...
        'S_median', 'S_IQR', ...
        'V_median', 'V_IQR', ...
        'S_mean', 'V_mean'};

T = array2table(Xtrain, 'VariableNames', feat);
T.Label = Ytrain;   % columna de respuesta (categorical)

save('legoFeatures_TrainingSet8carac.mat','T');
disp('✔ Tabla T creada en workspace con features y Label');
