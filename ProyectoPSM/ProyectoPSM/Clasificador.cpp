//----------------------------------------------------------------
// Script para la implementación de la clase Clasificador
//----------------------------------------------------------------

#include "Clasificador.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
using std::string;
using std::vector;


Clasificador::Clasificador() {
    svmLoaded = false;
    hasScaler = false;
}

//----------------------------------------------------------------
// Carga el modelo SVM y opcionalmente el scaler (mean/stdv)
//----------------------------------------------------------------
bool Clasificador::Load(const std::string& modelPath, const std::string& scalerPath) {
    svmLoaded = false;
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

    svmLoaded = true;
    return true;
}

//----------------------------------------------------------------
// Predice la clase de una imagen recortada
//----------------------------------------------------------------
int Clasificador::Predict(const cv::Mat& img) {
    if (!svmLoaded || img.empty()) return -1;

    // 1. Extraer Características (ahora son 11)
    std::vector<double> feats;
    std::vector<std::string> dummyNames;
    FeatureExtractor::ExtractColorShapeFeatures(img, feats, dummyNames);

    if (feats.empty()) return -1; // Imagen no válida

    // 2. Preparar matriz de fila para OpenCV
    cv::Mat rowD(1, static_cast<int>(feats.size()), CV_64F);
    for (size_t i = 0; i < feats.size(); ++i) {
        rowD.at<double>(0, (int)i) = feats[i];
    }

    // 3. Aplicar Normalización (Scaler) si existe
    if (hasScaler && mean.cols == rowD.cols) {
        for (int c = 0; c < rowD.cols; ++c) {
            double mu = mean.at<double>(0, c);
            double s = std::max(1e-12, stdv.at<double>(0, c)); // Evitar div por cero
            rowD.at<double>(0, c) = (rowD.at<double>(0, c) - mu) / s;
        }
    }

    // 4. Predecir
    cv::Mat rowF;
    rowD.convertTo(rowF, CV_32F);

    float response = svm->predict(rowF);
    return static_cast<int>(response);
}
