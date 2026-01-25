#pragma once
#include <opencv2/opencv.hpp>
#include <QString>
#include <QDir>
#include <QMap>
#include <vector>
#include <QRegularExpression>
#include <QFileInfo>
#include <QDebug>
#include <QFile>

struct TemplateConfig {
    QString inputFolder;    // Carpeta con imágenes segmentadas (crops)
    QString outputFolder;   // Carpeta donde guardar los .yml
    int templateSize = 128; // Tamaño final (ej: 128x128)
};

class TemplateGenerator {
public:
    static void Generate(const TemplateConfig& config, std::function<void(QString)> logCallback, std::function<void(int)> progressCallback);

private:
    static bool PreprocessImage(const cv::Mat& input, cv::Mat& output, int size);

    // Estructura para agrupar imágenes
    struct GroupKey {
        int code;
        int yaw;
        int pitch;

        // Necesario para usarlo como key en QMap/std::map
        bool operator<(const GroupKey& other) const {
            if (code != other.code) return code < other.code;
            if (yaw != other.yaw) return yaw < other.yaw;
            return pitch < other.pitch;
        }
    };
};