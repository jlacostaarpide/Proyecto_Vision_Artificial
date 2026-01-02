#include "ClasificadorSVM.h"
#include "ExtractCaracteristicas.h" // Tu extractor arreglado
#include <fstream>

using namespace cv;

ClasificadorSVM::ClasificadorSVM() {}

bool ClasificadorSVM::Load(const std::string& modelPath, const std::string& scalerPath) {
    loaded = false;
    hasScaler = false;

    // 1. Cargar SVM
    try {
        svm = cv::ml::SVM::load(modelPath);
        if (svm.empty()) return false;
    }
    catch (...) {
        return false;
    }

    // 2. Cargar Scaler (Media y Desviación estándar) si existe
    // Esto es CRÍTICO: El SVM necesita los datos en la misma escala que el entrenamiento
    if (!scalerPath.empty()) {
        cv::FileStorage fs(scalerPath, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["mean"] >> mean;
            fs["std"] >> stdv;

            // Asegurar tipos compatibles para operaciones matemáticas
            if (!mean.empty() && !stdv.empty()) {
                mean.convertTo(mean, CV_64F);
                stdv.convertTo(stdv, CV_64F);
                hasScaler = true;
            }
            fs.release();
        }
    }

    loaded = true;
    return true;
}

int ClasificadorSVM::Predict(const cv::Mat& img) {
    if (!loaded || img.empty()) return -1;

    // 1. Extraer Características (usando tu función corregida que devuelve bool)
    std::vector<double> feats;
    std::vector<std::string> dummyNames;
    FeatureExtractor::ExtractColorShapeFeatures(img, feats, dummyNames);

    //if (!ok || feats.empty()) return -1; // Imagen no válida

    // 2. Preparar matriz de fila para OpenCV
    cv::Mat rowD(1, static_cast<int>(feats.size()), CV_64F);
    for (size_t i = 0; i < feats.size(); ++i) {
        rowD.at<double>(0, (int)i) = feats[i];
    }

    // 3. Aplicar Normalización (Scaler) si existe
    // Fórmula: (Valor - Media) / Desviación
    if (hasScaler && mean.cols == rowD.cols) {
        for (int c = 0; c < rowD.cols; ++c) {
            double mu = mean.at<double>(0, c);
            double s = std::max(1e-12, stdv.at<double>(0, c)); // Evitar div por cero
            rowD.at<double>(0, c) = (rowD.at<double>(0, c) - mu) / s;
        }
    }

    // 4. Predecir
    cv::Mat rowF;
    rowD.convertTo(rowF, CV_32F); // SVM de OpenCV suele querer float (32F)

    float response = svm->predict(rowF);
    return static_cast<int>(response);
}