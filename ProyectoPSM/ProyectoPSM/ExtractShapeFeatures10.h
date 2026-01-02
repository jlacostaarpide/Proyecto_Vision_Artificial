#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FeatureExtractor10 {

    struct Shape10Opts {
        // --- máscara / limpieza ---
        double tBlackMin = 0.05;     // umbral fijo sobre gris 0..1 (fondo negro)
        int minObjArea = 800;        // bwareaopen
        int closeRadius = 3;
        int openRadius = 2;
        int holeSmallMaxArea = 400;  // rellena SOLO agujeros pequeños
        bool cropTight = true;

        // --- normalización ---
        int normTargetSize = 220;    // lado largo -> esto

        // --- proyecciones ---
        int projSmooth = 9;          // movmean

        // --- grid ---
        int gridN = 3;               // 3x3

        // --- studs robust (tophat + CC) ---
        bool studsEnable = true;
        int studsTophatRadius = 7;     // struct elem radius (depende de tamaño en normTargetSize)
        double studsThresh = 0.18;     // umbral (0..1) sobre respuesta tophat normalizada
        int studsMinArea = 20;         // área mínima componente
        int studsMaxArea = 600;        // área máxima componente
        double studsMinCircularity = 0.35; // 4?A/P^2
    };

    struct Shape10Dbg {
        cv::Mat maskN;       // máscara normalizada
        cv::Mat IgN;         // gris normalizado
        cv::Mat studsBin;    // binario studs
        std::vector<double> projH_s;
        std::vector<double> projV_s;
        cv::Mat grid;
    };

    inline std::vector<std::string> DefaultNames10() {
        return {
            "AreaNorm","PerimNorm","Extent","Solidity","AspectRatio",
            "ProjH_entropy","ProjV_entropy","ProjH_peaks",
            "GridOccGini_3x3","StudsCount"
        };
    }

    void ExtractShapeFeatures10(
        const cv::Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape10Opts& opts = Shape10Opts(),
        Shape10Dbg* dbg = nullptr
    );

} // namespace FeatureExtractor10