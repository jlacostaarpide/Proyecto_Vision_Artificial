#include "TrainingWorker.h"
#include "Segmentacion.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QFileInfo>

void TrainingWorker::process()
{
    // Solo ejecutamos segmentación
    runStepSegmentation();

    if (stopRequested.load()) {
        emit logMessage("Proceso cancelado por el usuario.");
    }
    else {
        emit logMessage("Proceso de segmentación finalizado.");
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
            emit logMessage("No se detectó ningún objeto en: " + fileName);
        }

        // D. Actualizar barra de progreso
        // Calculamos porcentaje
        int percent = static_cast<int>((static_cast<float>(i + 1) / totalFiles) * 100.0f);
        emit progressSeg(percent);
    }
}