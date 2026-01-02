#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct FeatureCacheData {
    cv::Mat X12; // CV_32F Nx12
    cv::Mat X24; // CV_32F Nx24 (puede estar vacía)
    cv::Mat y;   // CV_32S Nx1
    std::vector<std::string> filenames;
    std::vector<std::string> featNames12;
    std::vector<std::string> featNames24;
    std::string extractorVersion;
};

namespace FeatureCache {

    enum Mode {
        GLOBAL_12,
        REFINER_24,
        BOTH_12_24
    };

    bool BuildFromFolder(
        const std::string& segFolder,
        Mode mode,
        FeatureCacheData& out,
        int* outSkippedNoGT = nullptr,
        int* outSkippedBad = nullptr
    );

    bool SaveYml(const std::string& ymlPath, const FeatureCacheData& data);
    bool LoadYml(const std::string& ymlPath, FeatureCacheData& data);

}
