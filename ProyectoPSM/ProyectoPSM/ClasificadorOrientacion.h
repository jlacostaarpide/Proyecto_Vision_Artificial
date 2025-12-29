#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct OrientationResult {
    int yaw = -1;
    int pitch = -1;
    float bestScore = std::numeric_limits<float>::quiet_NaN();
    float gap = std::numeric_limits<float>::quiet_NaN();
    bool ok = false;
    std::string matchedCode;   // por si quieres saber qué code ganó
    std::string matchedFile;   // fichero template ganador
};

class ClasificadorOrientacion {
public:
    explicit ClasificadorOrientacion(std::string templatesFolder, int outSize = 128);

    // Carga todas las plantillas *.yml/*.yaml del folder (una vez)
    bool loadAllTemplates();

    // Predice usando todas las plantillas cargadas.
    // Si filterCode != "" (por ej "03"), solo compite contra ese code.
    OrientationResult predict(const cv::Mat& Ipiece, const std::string& filterCode = "") const;

private:
    struct TemplateItem {
        std::string code;
        int yaw = 0;
        int pitch = 0;
        int size = 128;
        cv::Mat T;              // CV_32F outSize x outSize (media 0, L2=1)
        std::string file;
    };

    std::string templatesFolder_;
    int outSize_;
    std::vector<TemplateItem> templates_; // todas las templates

private:
    static void zeroMeanL2Norm(cv::Mat& M);
    static bool largestComponent(cv::Mat& binMask);

    bool extractMaskLego(const cv::Mat& I, cv::Mat& maskOut) const;
    bool normalizeMaskedPatchShift(const cv::Mat& I, int shiftX, int shiftY, cv::Mat& J) const;

    static bool readOneTemplateYml(const std::string& path, TemplateItem& outItem, int outSize);
};
