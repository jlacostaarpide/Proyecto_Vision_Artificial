#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

// ---------------------------------------------------------
// 1. CLASE PARA INFERENCIA
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// 2. FUNCIONES GLOBALES DE ENTRENAMIENTO
// ---------------------------------------------------------
namespace TrainSVM {
    struct Options {
        std::string inputFolder;
        std::string outModelPath;
        std::string csvOut;
        double C = 1.0;
        double gamma = 0.0;
        bool doScale = false;
        bool doGridSearch = true; // nuevo: habilita Grid Search durante train
    };
}

int RunTrain(const TrainSVM::Options& opts);
int RunTrainRefiner(const TrainSVM::Options& opts, bool doLOO = true);
int RunEval(int argc, char** argv);
int RunEvalRefinerOnly(const std::string& segFolder,
    const std::string& outTxt,
    const std::string& model912,
    const std::string& model912scaler = "");
int RunExtractTest(int argc, char** argv);
