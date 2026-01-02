#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional> // Necesario para pasar funciones como variables

// Estructura genérica: No importa si son 6, 12 o 100 features.
struct FeatureCacheData {
    std::string extractorVersion;
    cv::Mat X; // Matriz N filas x M columnas (M se decide al vuelo)
    cv::Mat y; // Etiquetas

    std::vector<std::string> filenames;
    std::vector<std::string> featNames; // Nombres de las columnas
};

namespace FeatureCache {

    // Definimos el "tipo" de función que aceptamos:
    // Recibe (Imagen), devuelve (vector valores, vector nombres) -> retorna bool si ok
    using ExtractorFunc = std::function<bool(const cv::Mat&, std::vector<double>&, std::vector<std::string>&)>;

    // La función ahora pide un "extractor" en vez de un "Mode"
    bool BuildFromFolder(const std::string& segFolder,
        ExtractorFunc extractor,
        FeatureCacheData& out,
        int* outSkippedNoGT = nullptr,
        int* outSkippedBad = nullptr);

    bool SaveYml(const std::string& ymlPath, const FeatureCacheData& d);
    bool LoadYml(const std::string& ymlPath, FeatureCacheData& d);
}