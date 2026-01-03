#include "TrainingWorker.h"
#include "Segmentacion.h"
#include "FeatureCache.h"
#include "ExtractCaracteristicas.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QFileInfo>
#include <QRegularExpression>
#include <opencv2/ml.hpp>
#include <random>
#include <numeric>


void TrainingWorker::process()
{
    // Solo ejecutamos segmentación
    runStepSegmentation();
    runStepExtraction();
    runStepTraining();

    if (stopRequested.load()) {
        emit logMessage("Proceso cancelado por el usuario.");
    }
    else {
        emit logMessage("Proceso de entrenamiento finalizado.");
    }

    emit finished();
}

void TrainingWorker::runStepSegmentation()
{
    if (cfg.skipSegmentation) {
        emit logMessage("Saltando paso de segmentacion...");
        emit progressSeg(100);
        return;
    }

    emit logMessage("--- INICIANDO SEGMENTACION POR LOTES ---");

    QDir sourceDir(cfg.rawFolder);
    QDir destDir(cfg.segFolder);

    // 1. Validaciones básicas
    if (!sourceDir.exists()) {
        emit logMessage("ERROR: La carpeta Raw no existe: " + cfg.rawFolder);
        return;
    }
    if (!destDir.exists()) {
        // Intentar crearla si no existe
        if (destDir.mkpath(".")) {
            emit logMessage("Carpeta destino creada: " + cfg.segFolder);
        }
        else {
            emit logMessage("ERROR: No se pudo crear carpeta destino: " + cfg.segFolder);
            return;
        }
    }

    // 2. Filtrar archivos de imagen
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    sourceDir.setNameFilters(filters);

    QFileInfoList files = sourceDir.entryInfoList(QDir::Files);
    int totalFiles = files.size();

    if (totalFiles == 0) {
        emit logMessage("AVISO: No se encontraron imágenes en la carpeta Raw.");
        emit progressSeg(100);
        return;
    }

    emit logMessage(QString("Procesando %1 imagenes...").arg(totalFiles));

    // 3. Bucle de procesamiento
    for (int i = 0; i < totalFiles; ++i) {
        // Verificar cancelación
        if (stopRequested.load()) break;

        QFileInfo fileInfo = files[i];
        QString filePath = fileInfo.absoluteFilePath();
        QString fileName = fileInfo.fileName(); // Nombre original (ej: 01_001.jpg)

        // A. Cargar Imagen con QFile
        cv::Mat rawImg;

        QFile f(filePath);
        if (f.open(QIODevice::ReadOnly)) {
            // 1. Leer bytes con Qt
            QByteArray fileData = f.readAll();
            f.close();

            // 2. Convertir a vector para OpenCV
            std::vector<uchar> buf(fileData.begin(), fileData.end());

            // 3. Decodificar imagen desde memoria
            rawImg = cv::imdecode(buf, cv::IMREAD_COLOR);
        }

        if (rawImg.empty()) {
            emit logMessage("Error al leer (o decodificar): " + fileName);
            continue;
        }

        // B. Llamar a TU función de segmentación existente
        // No necesitamos pasarle &debugInfo porque no queremos pintar gráficos, solo resultados
        std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(rawImg, nullptr);

        // C. Guardar resultados
        if (!resultados.empty()) {
            // REGLA: Nos quedamos solo con la primera (resultados[0])
            // Segmentacion::Segmentar ya ordena por área descendente, así que la 0 es la más grande.
            const ResultadoPieza& piezaPrincipal = resultados[0];

            if (!piezaPrincipal.imagenRecortada.empty()) {
                QString outPath = destDir.filePath(fileName);

                // 1. Codificar la imagen en memoria (buffer) en formato JPG
                std::vector<uchar> buffer;
                std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 }; // Calidad 95%
                bool encoded = cv::imencode(".jpg", piezaPrincipal.imagenRecortada, buffer, params);

                if (encoded) {
                    // 2. Usar QFile para escribir los bytes en disco
                    QFile file(outPath);
                    if (file.open(QIODevice::WriteOnly)) {
                        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                        file.close();
                    }
                    else {
                        emit logMessage("Error: No se pudo crear el archivo (¿Permisos?): " + fileName);
                    }
                }
                else {
                    emit logMessage("Error: Falló la codificación JPG en memoria: " + fileName);
                }
            }
        }
        else {
            emit logMessage("No se detectan objetos en: " + fileName);
        }

        // D. Actualizar barra de progreso
        // Calculamos porcentaje
        int percent = static_cast<int>((static_cast<float>(i + 1) / totalFiles) * 100.0f);
        emit progressSeg(percent);
    }
}

void TrainingWorker::runStepExtraction()
{
    // 1. Verificar si el usuario quiere saltar este paso
    if (cfg.skipExtraction) {
        emit logMessage("Saltando paso de extracción (Feature Extraction)...");
        emit progressExtract(100);
        return;
    }

    emit logMessage("--- INICIANDO EXTRACCIÓN DE CARACTERÍSTICAS ---");

    QDir inputDir(cfg.segFolder);
    if (!inputDir.exists()) {
        emit logMessage("ERROR: La carpeta de imágenes segmentadas no existe: " + cfg.segFolder);
        return;
    }

    // 2. Listar imágenes
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";
    inputDir.setNameFilters(filters);

    // Ordenar por nombre para consistencia
    QFileInfoList files = inputDir.entryInfoList(QDir::Files, QDir::Name);
    int totalFiles = files.size();

    if (totalFiles == 0) {
        emit logMessage("ERROR: No hay imágenes en la carpeta segmentada.");
        return;
    }

    // Matrices para acumular datos (Formato OpenCV ML)
    cv::Mat trainingSamples;   // Matriz de features (N x D) tipo CV_32F
    cv::Mat trainingResponses; // Matriz de etiquetas (N x 1) tipo CV_32S

    int processed = 0;
    int skipped = 0;

    emit logMessage(QString("Extrayendo características de %1 imágenes...").arg(totalFiles));

    // 3. Bucle de procesamiento
    for (int i = 0; i < totalFiles; ++i) {
        if (stopRequested.load()) {
            emit logMessage("Extracción cancelada por el usuario.");
            return;
        }

        QFileInfo fileInfo = files[i];
        QString fileName = fileInfo.fileName();

        // A. Obtener etiqueta del nombre del archivo (Ej: "02_005.jpg" -> 2)
        // Usamos una expresión regular para buscar el número al principio
        QRegularExpression re("^(\\d+)");
        QRegularExpressionMatch match = re.match(fileName);

        int label = -1;
        if (match.hasMatch()) {
            label = match.captured(1).toInt();
        }
        else {
            skipped++;
            continue;
        }

        // B. Cargar Imagen (Usando QFile para robustez en rutas con tildes/ñ)
        cv::Mat img;
        QFile f(fileInfo.absoluteFilePath());
        if (f.open(QIODevice::ReadOnly)) {
            QByteArray fileData = f.readAll();
            f.close();
            std::vector<uchar> buf(fileData.begin(), fileData.end());
            img = cv::imdecode(buf, cv::IMREAD_COLOR);
        }

        if (img.empty()) {
            skipped++;
            continue;
        }

        // C. Extraer Características
        std::vector<double> feat;
        std::vector<std::string> names;

        FeatureExtractor::ExtractColorShapeFeatures(img, feat, names);

        if (feat.empty()) {
            skipped++;
            continue;
        }

        // D. Convertir a fila de Matriz
        cv::Mat row(1, static_cast<int>(feat.size()), CV_32F);
        for (size_t k = 0; k < feat.size(); ++k) {
            row.at<float>(0, static_cast<int>(k)) = static_cast<float>(feat[k]);
        }

        trainingSamples.push_back(row);
        trainingResponses.push_back(label);

        processed++;

        if (i % 10 == 0) {
            int percent = static_cast<int>((static_cast<float>(i + 1) / totalFiles) * 100.0f);
            emit progressExtract(percent);
        }
    }

    emit progressExtract(100);

    if (trainingSamples.empty()) {
        emit logMessage("ERROR: No se pudieron extraer características válidas.");
        return;
    }

    emit logMessage(QString("Extracción completada. Muestras: %1. Saltadas: %2").arg(processed).arg(skipped));

    // 4. Guardar a Archivo
    // IMPORTANTE: Convertimos la ruta a Local8Bit para que Windows acepte la "ñ" en OpenCV
    // Si la ruta del archivo features tiene directorios que no existen, hay que crearlos antes.
    QFileInfo featureFileInfo(cfg.featuresFile);
    QDir featureDir = featureFileInfo.absoluteDir();
    if (!featureDir.exists()) {
        featureDir.mkpath(".");
    }

    emit logMessage("Guardando archivo de características: " + cfg.featuresFile);

    try {
        // --- CAMBIO CLAVE AQUÍ: .toLocal8Bit().constData() ---
        // Esto convierte "Iñaki" a la codificación de Windows que espera fopen()
        cv::FileStorage fs(cfg.featuresFile.toLocal8Bit().constData(), cv::FileStorage::WRITE);

        if (fs.isOpened()) {
            fs << "samples" << trainingSamples;
            fs << "responses" << trainingResponses;
            fs.release();
            emit logMessage("Archivo guardado correctamente.");
        }
        else {
            emit logMessage("ERROR: No se pudo abrir el archivo para escritura (¿Ruta o permisos?).");
        }
    }
    catch (const cv::Exception& e) {
        emit logMessage("Excepción OpenCV al guardar: " + QString::fromStdString(e.what()));
    }
}

void TrainingWorker::runStepTraining()
{
    // 1. Verificar si saltamos el paso
    if (cfg.skipTraining) {
        emit logMessage("Saltando paso de entrenamiento...");
        emit progressTrain(100);
        return;
    }

    if (cfg.modelFile.isEmpty()) {
        emit logMessage("ERROR: No se ha especificado una ruta para guardar el modelo (.yml).");
        emit finished();
        return;
    }

    // Asegurar que la carpeta de destino existe
    QFileInfo modelInfo(cfg.modelFile);
    QDir modelDir = modelInfo.absoluteDir();
    if (!modelDir.exists()) {
        if (modelDir.mkpath(".")) {
            emit logMessage("Carpeta creada para el modelo: " + modelDir.absolutePath());
        }
        else {
            emit logMessage("ERROR: No se pudo crear la carpeta para el modelo: " + modelDir.absolutePath());
            return;
        }
    }

    emit logMessage("--- INICIANDO ENTRENAMIENTO SVM (K-Fold Grid Search) ---");

    // 2. Cargar Datos
    if (!QFile::exists(cfg.featuresFile)) {
        emit logMessage("ERROR: No existe el archivo de características: " + cfg.featuresFile);
        return;
    }

    cv::Mat samples, responses;
    try {
        // Usamos toLocal8Bit para rutas con caracteres especiales en Windows
        cv::FileStorage fs(cfg.featuresFile.toLocal8Bit().constData(), cv::FileStorage::READ);
        fs["samples"] >> samples;
        fs["responses"] >> responses;
        fs.release();
    }
    catch (...) {
        emit logMessage("ERROR: Fallo al leer el archivo de características.");
        return;
    }

    if (samples.empty() || responses.empty()) {
        emit logMessage("ERROR: Datos de entrenamiento vacíos.");
        return;
    }

    // Convertir a float (necesario para SVM)
    samples.convertTo(samples, CV_32F);
    responses.convertTo(responses, CV_32S); // Etiquetas a entero

    // 3. Normalización (Scaling) - LÓGICA IDÉNTICA A RUNTRAIN
    // Calculamos media y desviación típica
    cv::Mat meanVec = cv::Mat::zeros(1, samples.cols, CV_64F);
    cv::Mat stdVec = cv::Mat::zeros(1, samples.cols, CV_64F);

    for (int c = 0; c < samples.cols; ++c) {
        cv::Scalar mu, sigma;
        cv::meanStdDev(samples.col(c), mu, sigma);

        double s = sigma[0];
        if (s <= 1e-12) s = 1.0; // Evitar división por cero

        meanVec.at<double>(0, c) = mu[0];
        stdVec.at<double>(0, c) = s;

        // Aplicar normalización Z-score a la columna
        for (int r = 0; r < samples.rows; ++r) {
            float val = samples.at<float>(r, c);
            samples.at<float>(r, c) = static_cast<float>((val - mu[0]) / s);
        }
    }

    // Guardar el Scaler (necesario para usar el modelo luego)
    QString scalerPath = modelInfo.absolutePath() + "/" + modelInfo.baseName() + "_scaler.yml";

    try {
        cv::FileStorage fsSc(scalerPath.toLocal8Bit().constData(), cv::FileStorage::WRITE);
        fsSc << "mean" << meanVec;
        fsSc << "std" << stdVec;
        fsSc.release();
        emit logMessage("Scaler guardado: " + scalerPath);
    }
    catch (...) {
        emit logMessage("ERROR al guardar scaler.");
    }

    // 4. Grid Search con K-Fold (LÓGICA RUNTRAIN REPLICADA)

    // Parámetros a probar
    std::vector<double> C_vals = { 0.1, 1, 10, 100 };
    std::vector<double> Gamma_vals = { 0.001, 0.01, 0.1, 1 };

    int K = 5; // 5-Fold Cross Validation
    int N = samples.rows;
    if (K > N) K = N; // Seguridad para datasets muy pequeños

    // Preparar índices aleatorios para K-Fold
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    // Semilla fija para reproducibilidad (igual que RunTrain usaba 1234)
    std::mt19937 rng(1234);
    std::shuffle(indices.begin(), indices.end(), rng);

    double bestAcc = -1.0;
    double bestC = C_vals[0];
    double bestGamma = Gamma_vals[0];

    int totalCombinations = C_vals.size() * Gamma_vals.size();
    int currentStep = 0;

    emit logMessage(QString("Evaluando %1 combinaciones con %2-Fold CV...").arg(totalCombinations).arg(K));

    for (double C : C_vals) {
        for (double gamma : Gamma_vals) {

            if (stopRequested.load()) {
                emit logMessage("Entrenamiento cancelado.");
                return;
            }

            // --- INICIO K-FOLD MANUAL ---
            int correctTotal = 0;
            int totalSamples = 0;

            for (int k = 0; k < K; ++k) {
                cv::Mat trainS, trainR, testS, testR;

                // Separar datos en Train vs Test según el fold k
                for (int i = 0; i < N; ++i) {
                    int idx = indices[i];
                    if ((i % K) == k) {
                        // Pertenece al fold de test
                        testS.push_back(samples.row(idx));
                        testR.push_back(responses.row(idx));
                    }
                    else {
                        // Pertenece al fold de entrenamiento
                        trainS.push_back(samples.row(idx));
                        trainR.push_back(responses.row(idx));
                    }
                }

                // Entrenar SVM temporal en este fold
                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
                svm->setType(cv::ml::SVM::C_SVC);
                svm->setKernel(cv::ml::SVM::RBF);
                svm->setC(C);
                svm->setGamma(gamma);
                // Criterio de terminación estándar
                svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-6));

                bool ok = svm->train(trainS, cv::ml::ROW_SAMPLE, trainR);
                if (!ok) continue;

                // Evaluar en el fold de test
                for (int t = 0; t < testS.rows; ++t) {
                    float pred = svm->predict(testS.row(t));
                    int gt = testR.at<int>(t, 0);
                    if (static_cast<int>(pred) == gt) {
                        correctTotal++;
                    }
                    totalSamples++;
                }
            }
            // --- FIN K-FOLD ---

            double acc = (totalSamples > 0) ? (100.0 * correctTotal / totalSamples) : 0.0;

            // Log detallado (opcional, puede saturar si hay muchos)
            // emit logMessage(QString("C=%1 Gamma=%2 -> Acc=%3%").arg(C).arg(gamma).arg(acc, 0, 'f', 2));

            if (acc > bestAcc) {
                bestAcc = acc;
                bestC = C;
                bestGamma = gamma;
            }

            // Actualizar barra de progreso
            currentStep++;
            int progress = static_cast<int>((static_cast<float>(currentStep) / totalCombinations) * 100.0f);
            emit progressTrain(progress);
        }
    }

    emit logMessage(QString("MEJORES PARÁMETROS: C=%1 Gamma=%2 (Precisión CV: %3%)")
        .arg(bestC).arg(bestGamma).arg(bestAcc, 0, 'f', 2));

    // 5. Entrenamiento Final
    // Re-entrenamos con TODOS los datos usando los mejores parámetros encontrados
    emit logMessage("Entrenando modelo final con todos los datos...");

    cv::Ptr<cv::ml::SVM> finalSvm = cv::ml::SVM::create();
    finalSvm->setType(cv::ml::SVM::C_SVC);
    finalSvm->setKernel(cv::ml::SVM::RBF);
    finalSvm->setC(bestC);
    finalSvm->setGamma(bestGamma);
    finalSvm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-6));

    bool finalOk = finalSvm->train(samples, cv::ml::ROW_SAMPLE, responses);

    if (finalOk) {
        try {
            // Guardar modelo final
            finalSvm->save(cfg.modelFile.toLocal8Bit().constData());
            emit logMessage("Modelo guardado correctamente en: " + cfg.modelFile);
        }
        catch (const std::exception& e) {
            emit logMessage("ERROR al guardar el archivo .yml: " + QString(e.what()));
        }
    }
    else {
        emit logMessage("ERROR: Falló el entrenamiento final del SVM.");
    }

    emit progressTrain(100);
}