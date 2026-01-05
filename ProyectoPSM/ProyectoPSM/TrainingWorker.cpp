#include "TrainingWorker.h"
#include "Segmentacion.h"
#include "ExtractCaracteristicas.h"
#include "TemplateGenerator.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QFileInfo>
#include <QRegularExpression>
#include <opencv2/ml.hpp>
#include <random>
#include <numeric>
#include <iomanip>
#include <fstream>


void TrainingWorker::process()
{
    // Solo ejecutamos segmentación
	runStepSegmentation();
    if (stopRequested.load()) { emit finished(); return; }
    runStepExtraction();
    if (stopRequested.load()) { emit finished(); return; }
    runStepTemplates();
    if (stopRequested.load()) { emit finished(); return; }
    runStepTraining();
    if (stopRequested.load()) { emit finished(); return; }
	runStepEvaluation();
    emit finished();
}

void TrainingWorker::runStepSegmentation()
{
    if (cfg.skipSegmentation) {
        emit logMessage("Saltando paso de segmentacion...");
        emit progressSeg(100);
        return;
    }

    emit logMessage("");
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
        emit logMessage("Saltando paso de extraccion...");
        emit progressExtract(100);
        return;
    }

    emit logMessage("");
    emit logMessage("--- INICIANDO EXTRACCION DE CARACTERISTICAS ---");

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

    emit logMessage(QString("Extrayendo caracteristicas de %1 imagenes...").arg(totalFiles));

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

    emit logMessage(QString("Extraccion completada. Muestras: %1. Saltadas: %2").arg(processed).arg(skipped));

    // 4. Guardar a Archivo
    // IMPORTANTE: Convertimos la ruta a Local8Bit para que Windows acepte la "ñ" en OpenCV
    // Si la ruta del archivo features tiene directorios que no existen, hay que crearlos antes.
    QFileInfo featureFileInfo(cfg.featuresFile);
    QDir featureDir = featureFileInfo.absoluteDir();
    if (!featureDir.exists()) {
        featureDir.mkpath(".");
    }

    emit logMessage("Guardando archivo de caracteristicas: " + cfg.featuresFile);

    try {
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

void TrainingWorker::runStepTemplates()
{
    if (cfg.skipTemplates) {
        emit logMessage("Saltando generacion de plantillas...");
        emit progressTemplates(100);
        return;
    }

    emit logMessage("");
    emit logMessage("--- INICIANDO GENERACION DE PLANTILLAS DE ORIENTACION ---");

    TemplateConfig tplCfg;
    tplCfg.inputFolder = cfg.segFolder;       // Usa las imágenes segmentadas
    tplCfg.outputFolder = cfg.templatesFolder; // Carpeta destino
    tplCfg.templateSize = 128; // Tamaño estándar

    // Llamada estática, pasamos lambdas para conectar los logs y progreso con las señales del worker
    TemplateGenerator::Generate(tplCfg,
        [this](QString msg) { emit logMessage(msg); },
        [this](int p) { /* Podrias emitir una señal progressTemplates(p) si la creas */ }
    );
    emit progressTemplates(100);
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

    emit logMessage("");
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

    // 4. Grid Search con K-Fold

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

    emit logMessage(QString("MEJORES PARAMETROS: C=%1 Gamma=%2 (Precision CV: %3%)")
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

void TrainingWorker::runStepEvaluation()
{
    // 1. Verificar si saltamos el paso
    if (cfg.skipEvaluation) {
        emit logMessage("Saltando paso de evaluacion...");
        emit progressEval(100);
        return;
    }

    emit logMessage("");
    emit logMessage("--- INICIANDO EVALUACION DEL MODELO ---");

    // 2. Validaciones de carpetas
    if (!QFileInfo::exists(cfg.evaluationFolder)) {
        emit logMessage("ERROR: La carpeta de Test no existe: " + cfg.evaluationFolder);
        return;
    }
    if (!QFileInfo::exists(cfg.modelFile)) {
        emit logMessage("ERROR: No existe el modelo para evaluar: " + cfg.modelFile);
        return;
    }

    // 3. Cargar Modelo
    cv::Ptr<cv::ml::SVM> svm;
    try {
        svm = cv::Algorithm::load<cv::ml::SVM>(cfg.modelFile.toLocal8Bit().constData());
        if (!svm) throw std::runtime_error("Puntero nulo tras carga");
    }
    catch (const cv::Exception& e) {
        emit logMessage("ERROR CRITICO: Fallo al cargar SVM: " + QString(e.what()));
        return;
    }

    int expectedFeatures = svm->getVarCount();

    // 4. Cargar Scaler
    QFileInfo modelInfo(cfg.modelFile);
    QString scalerPath = modelInfo.absolutePath() + "/" + modelInfo.baseName() + "_scaler.yml";

    cv::Mat meanVec, stdVec;
    bool hasScaler = false;

    if (QFile::exists(scalerPath)) {
        try {
            cv::FileStorage fs(scalerPath.toLocal8Bit().constData(), cv::FileStorage::READ);
            fs["mean"] >> meanVec;
            fs["std"] >> stdVec;
            fs.release();

            // Verificar integridad
            if (!meanVec.empty() && !stdVec.empty() && meanVec.cols == stdVec.cols) {
                hasScaler = true;
                emit logMessage("Scaler cargado. (Esperando " + QString::number(expectedFeatures) + " features)");
            }
            else {
                emit logMessage("AVISO: Scaler corrupto o vacio.");
            }
        }
        catch (...) {
            emit logMessage("AVISO: Excepcion leyendo scaler.");
        }
    }
    else {
        emit logMessage("AVISO: No se encontro scaler. La precision puede ser baja.");
    }

    // 5. Preparar Reporte
    QString reportPath = modelInfo.absolutePath() + "/eval_report.txt";
    std::ofstream out(reportPath.toLocal8Bit().constData());
    if (!out.is_open()) {
        emit logMessage("ERROR: No se pudo crear el archivo de reporte.");
        return;
    }
    out << "filename,gt,pred\n";

    // 6. Bucle de Evaluacion
    QDir testDir(cfg.evaluationFolder);
    QStringList filters; filters << "*.jpg" << "*.png" << "*.bmp";
    testDir.setNameFilters(filters);
    QFileInfoList files = testDir.entryInfoList(QDir::Files, QDir::Name);

    int total = 0;
    int correct = 0;
    int skipped = 0;
    int nFiles = files.size();

    emit logMessage(QString("Evaluando %1 imagenes de: %2").arg(nFiles).arg(cfg.evaluationFolder));

    for (int i = 0; i < nFiles; ++i) {
        if (stopRequested.load()) {
            emit logMessage("Evaluacion cancelada.");
            out.close();
            return;
        }

        QString fileName = files[i].fileName();

        // Extraer GT (Ground Truth) del nombre
        QRegularExpression re("^(\\d+)");
        QRegularExpressionMatch match = re.match(fileName);
        int gt = -1;
        if (match.hasMatch()) {
            gt = match.captured(1).toInt();
        }
        else {
            skipped++;
            continue;
        }

        // Cargar Imagen (Robusto con QFile)
        cv::Mat img;
        QFile f(files[i].absoluteFilePath());
        if (f.open(QIODevice::ReadOnly)) {
            // Leemos todo en un QByteArray y lo pasamos a vector
            QByteArray bytes = f.readAll();
            std::vector<uchar> buf(bytes.begin(), bytes.end());
            // Decodificamos forzando COLOR para evitar errores en extractor
            img = cv::imdecode(buf, cv::IMREAD_COLOR);
            f.close();
        }

        if (img.empty()) { skipped++; continue; }

        // Extraer caracteristicas
        std::vector<double> feat;
        std::vector<std::string> dummy;
        try {
            FeatureExtractor::ExtractColorShapeFeatures(img, feat, dummy);
        }
        catch (...) {
            skipped++; continue;
        }

        if (feat.empty()) { skipped++; continue; }

        if ((int)feat.size() != expectedFeatures) {
            // Solo logueamos el primer error para no saturar
            if (skipped == 0) {
                emit logMessage(QString("ERROR DIMENSIONES: Imagen da %1 features, Modelo pide %2.")
                    .arg(feat.size()).arg(expectedFeatures));
            }
            skipped++;
            continue;
        }

        // Normalizar
        cv::Mat sample(1, (int)feat.size(), CV_32F);
        for (size_t k = 0; k < feat.size(); ++k) {
            float val = (float)feat[k];
            if (hasScaler) {
                double m = meanVec.at<double>(0, k);
                double s = stdVec.at<double>(0, k);
                val = (float)((val - m) / s);
            }
            sample.at<float>(0, k) = val;
        }

        // Predecir
        int pred = -1;
        try {
            float predFloat = svm->predict(sample);
            pred = static_cast<int>(predFloat);
        }
        catch (const cv::Exception& e) {
            emit logMessage("Error predict: " + QString(e.what())); 
            skipped++;
            continue;
        }

        out << fileName.toStdString() << "," << gt << "," << pred << "\n";

        if (pred == gt) correct++;
        total++;

        // Actualizar barra de progreso (porcentaje)
        int percent = static_cast<int>((static_cast<float>(i + 1) / nFiles) * 100.0f);
        emit progressEval(percent);
    }

    double accuracy = (total > 0) ? (100.0 * correct / total) : 0.0;

    out << "\n# RESUMEN\n";
    out << "Total: " << total << "\n";
    out << "Correctos: " << correct << "\n";
    out << "Precision: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    out.close();

    emit logMessage("Fin de la evaluacion. ");
    emit logMessage(QString("Procesados: %1 | Aciertos: %2").arg(total).arg(correct));
    emit logMessage(QString("PRECISION: %1%").arg(accuracy, 0, 'f', 2));
    emit logMessage("Reporte guardado en: " + reportPath);
    emit logMessage("");

    // Asegurar barra al 100% al terminar
    emit progressEval(100);
}