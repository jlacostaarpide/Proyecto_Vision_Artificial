clear all;close all;clc;
load('legoFeatures_TRAIN_shape_24carac_F_amarillas.mat'); % F todas 




%% Seleccionar de forma aleatoria las Train y las Test
%load('I_random.mat');
I=randperm(294);
F_train=F(I(1:230),:);
F_test=F(I(231:end),:);

%F_red = F(1:314,:);
%F_yellow = F(315:end,:);

% L_train = F.Label(I(1:1500),:);
% L_test = F.Label(I(1501:end),:);

% Quiero entrenar ahora con las train en classLearner. SVM funciona muy bien. Hacer crossvalidation y si quiero test tb, en el menu de new session. Y luego clasificar
% con las test con el predict.

%% TEST 1 IMAGEN ALEATORIA DE F_test (Real vs Pred)
% Requiere: F_test en workspace + trainedModel cargado + extractColorFeatures.m

% 1) Elegir fila aleatoria
rng('shuffle');
idx = randi(height(F_test));
row = F_test(idx,:);

fprintf('Fila seleccionada: %d de %d\n', idx, height(F_test));

% 2) Label real (de F_test)
trueLabel = row.Label(1);

% 3) Nombre de fichero (de F_test)
imgName = row.FileName{1};

% 4) Ruta a la carpeta SEGMENTED
segFolder = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';
imgPath = fullfile(segFolder, imgName);

if ~isfile(imgPath)
    error('No se encuentra la imagen: %s', imgPath);
end

% 5) Leer imagen
Ipiece = imread(imgPath);

% 6) Extraer features desde la imagen
feat = extractShapeFeatures(Ipiece);   % 1x14 double

% 7) Crear tabla de predictores con los MISMOS nombres y orden que F_test(:,1:14)
predictorNames = F_test.Properties.VariableNames(1:14);
featTable = array2table(feat, 'VariableNames', predictorNames);

% (Opcional) Reordenar a lo que espera el modelo, si lo necesitas:
% featTable = featTable(:, trainedModel.RequiredVariables);

% 8) Predicción
predictedLabel = trainedModel.predictFcn(featTable);

% 9) Mostrar resultado por consola
fprintf('\nImagen: %s\n', imgName);
fprintf('Clase REAL     : %s\n', string(trueLabel));
fprintf('Clase PREDICHA : %s\n', string(predictedLabel));

% % 10) Mostrar imagen
% figure('Name','Test sobre F_test','NumberTitle','off');
% imshow(Ipiece);
% title(sprintf('Real: %s | Pred: %s', string(trueLabel), string(predictedLabel)), 'FontSize', 14);


%% ================================================================
%% EVALUAR TODO F_test: predecir, guardar a TXT y contar aciertos
%% Requiere: F_test en workspace + trainedModel cargado
%% ================================================================

% --- Ajusta rutas ---
segFolder  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\SEGMENTED_local';
outputTxt  = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\resultados_Ftest_yellow.txt';

%load("TrainedModelWith_Ftrain.mat");

% --- Predictor names (las 14 primeras columnas son features) ---
predictorNames = F_test.Properties.VariableNames(1:14);

% --- Abrir TXT ---
fid = fopen(outputTxt,'w');
if fid==-1
    error('No se pudo crear el archivo: %s', outputTxt);
end

fprintf(fid, 'EVALUACIÓN COMPLETA SOBRE F_test\n');
fprintf(fid, '================================\n\n');
fprintf(fid, 'Carpeta imágenes: %s\n\n', segFolder);

N = height(F_test);
nOK = 0;
nFail = 0;
nMissing = 0;

% (Opcional) almacenar fallos para resumen
failList = strings(0,1);

for i = 1:N
    imgName   = F_test.FileName{i};
    trueLabel = F_test.Label(i);

    imgPath = fullfile(segFolder, imgName);

    if ~isfile(imgPath)
        fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=--- | ERROR: NO FILE\n', ...
            i, N, imgName, string(trueLabel));
        nMissing = nMissing + 1;
        continue;
    end

    % Leer imagen y recalcular features (pipeline real)
    Ipiece = imread(imgPath);
    feat   = extractShapeFeatures(Ipiece);             % 1x14
    featTable = array2table(feat, 'VariableNames', predictorNames);

    predictedLabel = trainedModel.predictFcn(featTable);

    % Comparar
    isCorrect = (predictedLabel == trueLabel);

    if isCorrect
        nOK = nOK + 1;
    else
        nFail = nFail + 1;
        failList(end+1,1) = sprintf('%s | REAL=%s | PRED=%s', ...
                                    imgName, string(trueLabel), string(predictedLabel));
    end

    % Guardar línea en TXT
    if isCorrect
    resultStr = 'OK';
    else
        resultStr = 'FAIL';
    end
    
    fprintf(fid, '[%4d/%4d] %s | REAL=%s | PRED=%s | %s\n', ...
    i, N, imgName, string(trueLabel), string(predictedLabel), resultStr);

    % Progreso en consola
    if mod(i,50)==0 || i==N
        fprintf('Procesadas %d/%d\n', i, N);
    end
end

% --- Resumen ---
totalEvaluated = nOK + nFail; % excluye missing
acc = 0;
if totalEvaluated > 0
    acc = 100 * (nOK / totalEvaluated);
end

fprintf(fid, '\n\nRESUMEN\n');
fprintf(fid, '------\n');
fprintf(fid, 'Total filas F_test        : %d\n', N);
fprintf(fid, 'Imágenes no encontradas   : %d\n', nMissing);
fprintf(fid, 'Evaluadas (con archivo)   : %d\n', totalEvaluated);
fprintf(fid, 'Aciertos                 : %d\n', nOK);
fprintf(fid, 'Fallos                   : %d\n', nFail);
fprintf(fid, 'Accuracy (sin missing)    : %.2f %%\n', acc);

% (Opcional) listar fallos al final
fprintf(fid, '\n\nLISTA DE FALLOS (si los hay)\n');
fprintf(fid, '----------------------------\n');
if nFail == 0
    fprintf(fid, 'Ninguno.\n');
else
    for k = 1:numel(failList)
        fprintf(fid, '%s\n', failList(k));
    end
end

fclose(fid);

fprintf('\nHecho. TXT guardado en:\n%s\n', outputTxt);
fprintf('Aciertos: %d | Fallos: %d | Missing: %d | Acc: %.2f%%\n', ...
        nOK, nFail, nMissing, acc);




