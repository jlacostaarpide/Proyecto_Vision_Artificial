#include "FeatureCache.h"
#include <filesystem>
#include <regex>
#include <algorithm>

#include "ExtractCaracteristicas.h"
#include "ExtractCaracteristicas24Refinador.h"

namespace fs = std::filesystem;
using namespace cv;
using std::string;
using std::vector;

static const vector<string> exts = { ".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp" };

static bool hasSupportedExt(const fs::path& p) {
    string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return std::find(exts.begin(), exts.end(), e) != exts.end();
}

static bool parseGTfromFilenameInt(const string& fname, int& outLabel) {
    std::regex rx(R"(^(\d{1,2}))");
    std::smatch m;
    if (std::regex_search(fname, m, rx) && m.size() >= 2) {
        try { outLabel = std::stoi(m[1].str()); return true; }
        catch (...) { return false; }
    }
    return false;
}

namespace FeatureCache {

    static std::string kExtractorVersion() {
        // Sube este string cada vez que cambies extracción:
        return "v1_2026-01-01";
    }

    bool BuildFromFolder(const std::string& segFolder, Mode mode, FeatureCacheData& out,
        int* outSkippedNoGT, int* outSkippedBad)
    {
        out = FeatureCacheData{};
        out.extractorVersion = kExtractorVersion();

        if (outSkippedNoGT) *outSkippedNoGT = 0;
        if (outSkippedBad)  *outSkippedBad = 0;

        if (!fs::exists(segFolder) || !fs::is_directory(segFolder)) return false;

        vector<fs::path> files;
        for (auto& e : fs::directory_iterator(segFolder)) {
            if (!e.is_regular_file()) continue;
            if (hasSupportedExt(e.path())) files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) return false;

        Mat X12, X24, y;
        vector<string> names12, names24;
        bool names12Set = false, names24Set = false;

        for (size_t i = 0; i < files.size(); ++i) {
            const auto& p = files[i];
            string fname = p.filename().string();

            int label = -1;
            if (!parseGTfromFilenameInt(fname, label)) {
                if (outSkippedNoGT) (*outSkippedNoGT)++;
                continue;
            }

            Mat I = imread(p.string(), IMREAD_COLOR);
            if (I.empty()) {
                if (outSkippedBad) (*outSkippedBad)++;
                continue;
            }

            vector<double> f12, f24;
            vector<string> n12, n24;

            if (mode == GLOBAL_12 || mode == BOTH_12_24) {
                FeatureExtractor::ExtractColorShapeFeatures(I, f12, n12);
                if (f12.size() != 12) { if (outSkippedBad) (*outSkippedBad)++; continue; }
                if (!names12Set) { names12 = n12; names12Set = true; }
            }

            if (mode == REFINER_24 || mode == BOTH_12_24) {
                FeatureExtractor24::ExtractShapeFeatures24(I, f24, n24);
                if (f24.size() != 24) { if (outSkippedBad) (*outSkippedBad)++; continue; }
                if (!names24Set) { names24 = n24; names24Set = true; }
            }

            if (!f12.empty()) {
                Mat row(1, 12, CV_32F);
                for (int k = 0; k < 12; ++k) row.at<float>(0, k) = (float)f12[k];
                X12.push_back(row);
            }
            if (!f24.empty()) {
                Mat row(1, 24, CV_32F);
                for (int k = 0; k < 24; ++k) row.at<float>(0, k) = (float)f24[k];
                X24.push_back(row);
            }

            y.push_back(Mat(1, 1, CV_32S, Scalar(label)));
            out.filenames.push_back(fname);
        }

        if (y.rows == 0) return false;
        y = y.reshape(1, y.rows);

        out.X12 = X12;
        out.X24 = X24;
        out.y = y;
        out.featNames12 = names12;
        out.featNames24 = names24;
        return true;
    }

    bool SaveYml(const std::string& ymlPath, const FeatureCacheData& d)
    {
        fs::path outp(ymlPath);
        if (!outp.parent_path().empty()) fs::create_directories(outp.parent_path());

        cv::FileStorage fsw(ymlPath, cv::FileStorage::WRITE);
        if (!fsw.isOpened()) return false;

        fsw << "extractor_version" << d.extractorVersion;
        fsw << "X12" << d.X12;
        fsw << "X24" << d.X24;
        fsw << "y" << d.y;

        fsw << "filenames" << "[";
        for (auto& s : d.filenames) fsw << s;
        fsw << "]";

        fsw << "featNames12" << "[";
        for (auto& s : d.featNames12) fsw << s;
        fsw << "]";

        fsw << "featNames24" << "[";
        for (auto& s : d.featNames24) fsw << s;
        fsw << "]";

        fsw.release();
        return true;
    }

    bool LoadYml(const std::string& ymlPath, FeatureCacheData& d)
    {
        cv::FileStorage fsr(ymlPath, cv::FileStorage::READ);
        if (!fsr.isOpened()) return false;

        d = FeatureCacheData{};
        fsr["extractor_version"] >> d.extractorVersion;
        fsr["X12"] >> d.X12;
        fsr["X24"] >> d.X24;
        fsr["y"] >> d.y;

        cv::FileNode fn = fsr["filenames"];
        for (auto it = fn.begin(); it != fn.end(); ++it) d.filenames.push_back((string)*it);

        cv::FileNode f12 = fsr["featNames12"];
        for (auto it = f12.begin(); it != f12.end(); ++it) d.featNames12.push_back((string)*it);

        cv::FileNode f24 = fsr["featNames24"];
        for (auto it = f24.begin(); it != f24.end(); ++it) d.featNames24.push_back((string)*it);

        fsr.release();
        return !d.y.empty();
    }

} // namespace
