#include "TemplateGenerator.h"
#include <QRegularExpression>
#include <QFileInfo>
#include <QDebug>

// Función equivalente a 'normalizeMaskedPatch' de Matlab
bool TemplateGenerator::PreprocessImage(const cv::Mat& input, cv::Mat& output, int size) {
    if (input.empty()) return false;

    // --- 1. Preparación (MATLAB: im2double + rgb2gray) ---
    cv::Mat gray;
    if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();

    // --- 2. Umbral Fijo (MATLAB: t = 0.03; mask = Ig > t) ---
    // En MATLAB 0.03 es sobre 1.0. En OpenCV (0-255): 0.03 * 255 ≈ 7.65 -> Usamos 8.
    // Esto asume fondo muy oscuro/negro.
    cv::Mat mask;
    cv::threshold(gray, mask, 8, 255, cv::THRESH_BINARY);

    // --- 3. Limpieza Morfológica ---

    // A. MATLAB: imclose(mask, strel('disk', 2));
    // Disk radio 2 = diametro 5x5 (aprox)
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose);

    // B. MATLAB: imopen(mask,  strel('disk', 1));
    // Disk radio 1 = diametro 3x3
    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelOpen);

    // --- 4. Componente Mayor y Relleno (MATLAB: bwareaopen, imfill, max area) ---
    // En OpenCV esto se hace con findContours y dibujando solo el más grande RELLENO.

    std::vector<std::vector<cv::Point>> contours;
    // RETR_EXTERNAL solo busca el contorno exterior (equivale a ignorar agujeros internos)
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    int maxIdx = -1;
    double maxArea = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);

        // MATLAB: bwareaopen(mask, 150) -> Descartar menores de 150px
        if (area < 150) continue;

        // MATLAB: Quedarse con el componente mayor
        if (area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx == -1) return false; // No se encontró nada válido

    // --- 5. Generar Máscara Final Limpia ---
    // Creamos una máscara negra nueva y pintamos SOLO el contorno ganador
    // FILLED (-1) equivale a MATLAB: imfill(mask, 'holes') implícito al pintar el interior
    cv::Mat finalMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(finalMask, contours, maxIdx, cv::Scalar(255), cv::FILLED);

    // --- 6. Aplicar Máscara a la Imagen (MATLAB: Gc(~Mc) = 0) ---
    cv::Mat maskedGray;
    // Pone a negro todo lo que no esté en finalMask
    cv::bitwise_and(gray, gray, maskedGray, finalMask);

    // --- 7. Recorte (Bounding Box) ---
    cv::Rect maxRect = cv::boundingRect(contours[maxIdx]);
    cv::Mat cropped = maskedGray(maxRect);

    // --- 8. Padding (Hacer cuadrada) ---
    int h = cropped.rows;
    int w = cropped.cols;
    int dim = std::max(h, w);

    int top = (dim - h) / 2;
    int bottom = dim - h - top;
    int left = (dim - w) / 2;
    int right = dim - w - left;

    cv::Mat padded;
    cv::copyMakeBorder(cropped, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));

    // --- 9. Resize (128x128) ---
    cv::resize(padded, output, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

    // --- 10. Normalización Final (Standardization) ---
    // (Igual que en tu código anterior y Matlab)
    output.convertTo(output, CV_32F);
    cv::Scalar meanVal = cv::mean(output);
    output -= meanVal;
    double normVal = cv::norm(output);
    if (normVal > 1e-6) output /= normVal;

    return true;
}

void TemplateGenerator::Generate(const TemplateConfig& config, std::function<void(QString)> logCallback, std::function<void(int)> progressCallback) {
    QDir inDir(config.inputFolder);
    if (!inDir.exists()) {
        logCallback("ERROR: Carpeta de entrada no existe.");
        return;
    }

    QDir outDir(config.outputFolder);
    if (!outDir.exists()) outDir.mkpath(".");

    // Filtros de imagen
    QStringList filters; filters << "*.jpg" << "*.png" << "*.bmp";
    inDir.setNameFilters(filters);
    QFileInfoList files = inDir.entryInfoList(QDir::Files);

    // 1. Agrupar archivos por (Code, Yaw, Pitch)
    // Usamos un mapa donde la clave es la combinación y el valor es la lista de rutas
    std::map<GroupKey, std::vector<QString>> groups;

    // Regex espera formato: 01_000_90_xx.jpg
    QRegularExpression re("^(\\d+)_(\\d+)_(\\d+)");

    logCallback("Agrupando imagenes...");

    int totalFiles = files.size();
    for (int i = 0; i < totalFiles; ++i) {
        QString fname = files[i].fileName();
        QRegularExpressionMatch match = re.match(fname);

        if (match.hasMatch()) {
            GroupKey key;
            key.code = match.captured(1).toInt();
            key.yaw = match.captured(2).toInt();
            key.pitch = match.captured(3).toInt();

            groups[key].push_back(files[i].absoluteFilePath());
        }

        if (i % 100 == 0) progressCallback((int)(i * 20.0 / totalFiles)); // Progreso fase 1 (0-20%)
    }

    logCallback(QString("Detectados %1 grupos unicos (plantillas a generar).").arg(groups.size()));

    // 2. Procesar cada grupo
    int groupIdx = 0;
    int totalGroups = groups.size();

    for (auto const& [key, filePaths] : groups) {
        if (filePaths.empty()) continue;

        cv::Mat accumulator = cv::Mat::zeros(config.templateSize, config.templateSize, CV_32F);
        int count = 0;

        for (const QString& path : filePaths) {
            // Cargar imagen (usando QFile para rutas con caracteres especiales)
            cv::Mat img;
            QFile f(path);
            if (f.open(QIODevice::ReadOnly)) {
                std::vector<uchar> buf(f.readAll().begin(), f.readAll().end());
                img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
            }

            if (img.empty()) continue;

            cv::Mat processed;
            if (PreprocessImage(img, processed, config.templateSize)) {
                accumulator += processed;
                count++;
            }
        }

        // 3. Promediar y guardar si hay suficientes muestras
        if (count > 0) {
            // Promedio
            cv::Mat templateFinal = accumulator / count;

            // Normalizar de nuevo el resultado final (como en Matlab)
            cv::Scalar m = cv::mean(templateFinal);
            templateFinal -= m;
            double n = cv::norm(templateFinal);
            if (n > 1e-6) templateFinal /= n;

            // Guardar YAML con OpenCV
            QString outName = QString("tpl_%1_%2_%3.yml")
                .arg(key.code, 2, 10, QChar('0'))
                .arg(key.yaw, 3, 10, QChar('0'))
                .arg(key.pitch, 2, 10, QChar('0'));

            QString outPath = outDir.filePath(outName);

            try {
                // toLocal8Bit para Windows paths
                cv::FileStorage fs(outPath.toLocal8Bit().constData(), cv::FileStorage::WRITE);

                // Escribir metadatos y matriz
                fs << "code" << QString::number(key.code).toStdString();
                fs << "yaw" << key.yaw;
                fs << "pitch" << key.pitch;
                fs << "size" << config.templateSize;
                fs << "T" << templateFinal;

                fs.release();
            }
            catch (...) {
                logCallback("Error guardando: " + outName);
            }
        }

        groupIdx++;
        // Progreso fase 2 (20-100%)
        int p = 20 + (int)(groupIdx * 80.0 / totalGroups);
        progressCallback(p);
    }

    progressCallback(100);
    logCallback("Generacion de plantillas finalizada.");
}