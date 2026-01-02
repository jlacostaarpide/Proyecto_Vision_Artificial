#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FeatureExtractor8 {

    struct Shape8Opts {
        // --- mask/segment params (igual filosofía que el de 6) ---
        double tBlackMin = 0.05;   // umbral sobre gris 0..1
        int    minObjArea = 300;
        int    closeRadius = 3;
        int    openRadius = 2;
        int    holeSmallMaxArea = 800;

        // --- normalización geométrica ---
        bool cropTight = true;
        int  normTargetSize = 220;

        // --- features de proyección/grid (como en el de 6) ---
        int projSmooth = 11;
        int gridN = 3;

        // --- color features ---
        int kmeansAttempts = 3;     // K=2 sobre L
        int kmeansMaxIter = 50;
        double kmeansEps = 1e-3;
    };

    struct Shape8Dbg {
        cv::Mat maskN;     // máscara normalizada
        cv::Mat InColorN;  // imagen color normalizada (BGR) para depurar
        cv::Mat InLabN;    // Lab normalizado
    };

    std::vector<std::string> DefaultNames8();

    // 8 features:
    // 0 AreaNorm
    // 1 PerimNorm
    // 2 ProjH_entropy
    // 3 GridOccFrac_3x3
    // 4 GridOccGini_3x3
    // 5 GridOccDiagDiff_3x3
    // 6 LightFrac (K=2 sobre L dentro de máscara)
    // 7 LightContrast (diff medias L entre clusters)
    void ExtractShapeFeatures8(const cv::Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape8Opts& opts = Shape8Opts(),
        Shape8Dbg* dbg = nullptr);

} // namespace FeatureExtractor8
