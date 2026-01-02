#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

class ClasificadorSVM {
public:
    ClasificadorSVM();

    // Carga el modelo .yml y opcionalmente el scaler (mean/std)
    bool Load(const std::string& modelPath, const std::string& scalerPath = "");

    // Predice la clase de una imagen recortada
    // Devuelve el ID de clase (ej: 1, 2, 5...) o -1 si error
    int Predict(const cv::Mat& imgRecortada);

    bool IsLoaded() const { return loaded; }

private:
    cv::Ptr<cv::ml::SVM> svm;
    cv::Mat mean, stdv; // Para la normalización (Z-score)
    bool loaded = false;
    bool hasScaler = false;
};