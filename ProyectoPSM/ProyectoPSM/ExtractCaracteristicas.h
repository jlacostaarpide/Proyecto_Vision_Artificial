
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace FeatureExtractor {

    // Extrae las 12 características (color + forma) y los nombres en el mismo orden
    // Entrada: I - imagen BGR (CV_8U o CV_32F) con fondo negro alrededor de la pieza
    // Salida: feat (12 valores), featNames (12 strings)
    void ExtractColorShapeFeatures(const cv::Mat& I_in, std::vector<double>& feat, std::vector<std::string>& featNames);

} // namespace FeatureExtractor