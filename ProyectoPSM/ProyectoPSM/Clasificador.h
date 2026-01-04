#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>


class Clasificador {
public:
    Clasificador();
    
    // Carga el modelo .yml y opcionalmente el scaler (mean/std)
    bool Load(const std::string& modelPath, const std::string& scalerPath = "");

    // Predice la clase de una imagen recortada
    // Devuelve el ID de clase (ej: 1, 2, 5...) o -1 si error
    int Predict(const cv::Mat& img);

    // Helper para verificar si está cargado
    bool IsLoaded() const { return svmLoaded; }

private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::Mat mean, stdv; // Para la normalización (Z-score)
    bool svmLoaded = false;
    bool hasScaler = false;
};
