#include "TrainingWorker.h"
#include "Segmentacion.h"
#include "FeatureCache.h"
#include "ExtractCaracteristicas.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QFileInfo>
#include <QRegularExpression>

void TrainingWorker::process()
{
    // Solo ejecutamos segmentación
    runStepSegmentation();
    runStepExtraction();

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