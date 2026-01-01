#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FeatureExtractor24 {

    struct Shape24Opts {
        // MATLAB defaults
        double tBlackMin = 0.03;
        int    minObjArea = 300;
        int    closeRadius = 3;
        int    openRadius = 2;
        int    holeSmallMaxArea = 200;
        bool   cropTight = true;

        // normalización escala
        int    normTargetSize = 220;   // lado largo del crop

        // proyecciones
        int    projSmooth = 7;     // movmean window

        // grid
        int    gridN = 3;

        // studs (círculos)
        bool   studsEnable = true;
        double studsSensitivity = 0.92;  // (OpenCV: param2 ajustado abajo)
        int    studsRmin = 6;
        int    studsRmax = 20;
    };

    void ExtractShapeFeatures24(const cv::Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape24Opts& opts = Shape24Opts());

} // namespace FeatureExtractor24
