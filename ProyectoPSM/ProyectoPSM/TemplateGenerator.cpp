//-----------------------------------------------------------------------------
// Script para generar plantillas a partir de imagenes segmentadas
//-----------------------------------------------------------------------------

#include "TemplateGenerator.h"

// ----------------------------------------------------------------------------
// Método de preprocesamiento de imagenes
// ----------------------------------------------------------------------------
bool TemplateGenerator::PreprocessImage(const cv::Mat& input, cv::Mat& output, int size) {
    if (input.empty()) return false;

    //  1. Preparación 
    cv::Mat gray;
    if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();

    //  2. Umbral Fijo 
    cv::Mat mask;
    cv::threshold(gray, mask, 8, 255, cv::THRESH_BINARY);

    //  3. Limpieza Morfológica 
    cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelClose);

    cv::Mat kernelOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelOpen);

    //  4. Componente Mayor y Relleno 
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return false;

    int maxIdx = -1;
    double maxArea = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 150) continue; // Filtro de ruido

        if (area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx == -1) return false;

    //  5. Generar Máscara Final Limpia 
    cv::Mat finalMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(finalMask, contours, maxIdx, cv::Scalar(255), cv::FILLED);

    //  6. Aplicar Máscara 
    cv::Mat maskedGray;
    cv::bitwise_and(gray, gray, maskedGray, finalMask);

    //  7. Recorte 
    cv::Rect maxRect = cv::boundingRect(contours[maxIdx]);
    cv::Mat cropped = maskedGray(maxRect);

    //  8. Padding 
    int h = cropped.rows;
    int w = cropped.cols;
    int dim = std::max(h, w);

    int top = (dim - h) / 2;
    int bottom = dim - h - top;
    int left = (dim - w) / 2;
    int right = dim - w - left;

    cv::Mat padded;
    cv::copyMakeBorder(cropped, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));

    //  9. Resize 
    cv::resize(padded, output, cv::Size(size, size), 0, 0, cv::INTER_LINEAR);

    //  10. Normalización 
    output.convertTo(output, CV_32F);
    cv::Scalar meanVal = cv::mean(output);
    output -= meanVal;
    double normVal = cv::norm(output);
    if (normVal > 1e-6) output /= normVal;

    return true;
}

// ----------------------------------------------------------------------------
// Método principal de generación de plantillas
// ----------------------------------------------------------------------------
void TemplateGenerator::Generate(const TemplateConfig& config, std::function<void(QString)> logCallback, std::function<void(int)> progressCallback) {
    QDir inDir(config.inputFolder);
    if (!inDir.exists()) {
        logCallback("ERROR: Carpeta de entrada no existe.");
        return;
    }

    QDir outDir(config.outputFolder);
    if (!outDir.exists()) outDir.mkpath(".");

    QStringList filters; filters << "*.jpg" << "*.png" << "*.bmp" << "*.jpeg";
    inDir.setNameFilters(filters);
    QFileInfoList files = inDir.entryInfoList(QDir::Files);

    if (files.isEmpty()) {
        logCallback("ERROR: No hay imágenes en la carpeta de entrada.");
        return;
    }

    // 1. Agrupar
    std::map<GroupKey, std::vector<QString>> groups;
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

        if (i % 50 == 0) progressCallback((int)(i * 20.0 / totalFiles));
    }

    logCallback(QString("Detectados %1 grupos unicos (plantillas a generar).").arg(groups.size()));

    // 2. Procesar
    int groupIdx = 0;
    int totalGroups = groups.size();

    for (auto const& [key, filePaths] : groups) {
        if (filePaths.empty()) continue;

        cv::Mat accumulator = cv::Mat::zeros(config.templateSize, config.templateSize, CV_32F);
        int count = 0;

        for (const QString& path : filePaths) {
            cv::Mat img;
            QFile f(path);
            if (f.open(QIODevice::ReadOnly)) {
                QByteArray fileBytes = f.readAll();
                std::vector<uchar> buf(fileBytes.begin(), fileBytes.end());
                img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
                f.close();
            }

            if (img.empty()) continue;

            cv::Mat processed;
            if (PreprocessImage(img, processed, config.templateSize)) {
                accumulator += processed;
                count++;
            }
        }

        // 3. Promediar y guardar
        if (count > 0) {
            cv::Mat templateFinal = accumulator / count;

            cv::Scalar m = cv::mean(templateFinal);
            templateFinal -= m;
            double n = cv::norm(templateFinal);
            if (n > 1e-6) templateFinal /= n;

            // Nombre de archivo: tpl_08_000_90.yml
            QString outName = QString("tpl_%1_%2_%3.yml")
                .arg(key.code, 2, 10, QChar('0'))
                .arg(key.yaw, 3, 10, QChar('0'))
                .arg(key.pitch, 2, 10, QChar('0'));

            QString outPath = outDir.filePath(outName);

            try {
                cv::FileStorage fs(outPath.toLocal8Bit().constData(), cv::FileStorage::WRITE);

                QString codeStr = QString("%1").arg(key.code, 2, 10, QChar('0'));

                fs << "code" << codeStr.toStdString();
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
        int p = 20 + (int)(groupIdx * 80.0 / totalGroups);
        progressCallback(p);
    }

    progressCallback(100);
    logCallback("Generacion de plantillas finalizada.");
}