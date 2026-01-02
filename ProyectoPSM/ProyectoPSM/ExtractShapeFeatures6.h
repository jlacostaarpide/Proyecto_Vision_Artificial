#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FeatureExtractor6 {

    struct Shape6Opts {
        // defaults (igual que tu MATLAB)
        double tBlackMin = 0.03;
        int    minObjArea = 300;
        int    closeRadius = 3;
        int    openRadius = 2;
        int    holeSmallMaxArea = 200;  // rellena SOLO agujeros pequeños
        bool   cropTight = true;

        int    normTargetSize = 220;  // lado largo objetivo

        int    projSmooth = 7;    // movmean
        int    gridN = 3;    // 3x3
    };

    struct Shape6Dbg {
        cv::Mat maskN;     // CV_8U 0/255
        cv::Mat IgN;       // CV_32F 0..1
        std::vector<double> projH_s; // proyección horizontal suavizada
        cv::Mat grid;      // CV_64F NxN
    };

    // N=6 features:
    // 1 AreaNorm
    // 2 PerimNorm
    // 3 ProjH_entropy
    // 4 GridOccFrac_3x3
    // 5 GridOccGini_3x3
    // 6 GridOccDiagDiff_3x3
    void ExtractShapeFeatures6(const cv::Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape6Opts& opts = Shape6Opts(),
        Shape6Dbg* dbg = nullptr);

    inline std::vector<std::string> DefaultNames6() {
        return {
            "AreaNorm","PerimNorm","ProjH_entropy",
            "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3"
        };
    }

} // namespace FeatureExtractor6
#pragma once
