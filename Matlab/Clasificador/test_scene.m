%% TEST CLASIFICADOR PIEZAS LEGO


%% === 1) CARGAR MODELO ENTRENADO ===

load('legoFeatures_TrainingSet8carac.mat');   % el modelo exportado desde Classification Learner
% Alternativa si usaste fitcknn:
% load('legoModel_porCodigo.mat'); % variable Mdl

%% === 2) SELECCIONAR IMAGEN A TESTEAR ===

testImage = 'C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Database\tests\amarillas_camara.jpg';

fprintf("\n--- Clasificando imagen: %s ---\n", testImage);

%% === 3) SEGMENTAR LA IMAGEN ===

[pieces_test, stats_test, num_test, Icorr] = segmentarPiezas2(testImage);

if num_test == 0
    error("❌ No se detectaron piezas en la imagen de test");
end

%% === 4) CLASIFICAR CADA PIEZA DETECTADA ===

figure('Name','Clasificación Test','NumberTitle','off');

for k = 1:num_test
    Ipiece = pieces_test{k};

    % 1) Extraer features (fila 1×D)
    feat = extractColorFeatures(Ipiece);   % p.ej. 1x8 double
    
    % 2) Convertir a tabla con los mismos nombres que en el entrenamiento
    featTable = array2table(feat, ...
        'VariableNames', trainedModel.RequiredVariables);
    
    % 3) Predecir usando el modelo exportado
    predictedLabel = trainedModel.predictFcn(featTable);

    % Si usas un KNN manual:
    % predictedLabel = predict(Mdl, feat);

    % === MOSTRAR RESULTADO ===
    subplot(1, num_test, k);
    imshow(Ipiece);
    title(sprintf('Pred: %s', string(predictedLabel)), 'FontSize',14);
end




