
#include "entrenarSVM.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <regex>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "ExtractCaracteristicas.h"

//recorre una carpeta de imágenes segmentadas, extrae las 12 features por imagen (usando ExtractColorShapeFeatures), 
// genera opcionalmente un CSV con label,feat1,..., aplica z‑score si pides --scale, entrena un SVM polinómico grado 2 (cuadratico) con OpenCV 
// y guarda: model.yml y opcionalmente model.yml_scaler.yml.

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
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
        try {
            int v = std::stoi(m[1].str());
            outLabel = v;
            return true;
        }
        catch (...) { return false; }
    }
    return false;
}

static void printUsage() {
    std::cout << "Usage: train_svm_quadratic <input_folder> <out_model.yml> [--csv features.csv] [--C 1.0] [--gamma 0.0] [--scale]\n";
    std::cout << "  --csv : optional CSV output with header: label,feat1,feat2,...\n";
    std::cout << "  --C : SVM C parameter (default 1.0)\n";
    std::cout << "  --gamma : gamma (default 0 = auto 1/n_features)\n";
    std::cout << "  --scale : apply z-score normalization (mean/std) and save scaler to <out_model>_scaler.yml\n";
}

namespace TrainSVM {

    int TrainQuadraticSVM(const Options& opts)
    {
        // validar carpeta
        if (!fs::exists(opts.inputFolder) || !fs::is_directory(opts.inputFolder)) {
            std::cerr << "Input folder not found or not a directory: " << opts.inputFolder << "\n";
            return 1;
        }

        // listar imágenes
        vector<fs::path> files;
        for (auto& entry : fs::directory_iterator(opts.inputFolder)) {
            if (!entry.is_regular_file()) continue;
            if (hasSupportedExt(entry.path())) files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "No supported images found in: " << opts.inputFolder << "\n";
            return 1;
        }

        std::cout << "Found " << files.size() << " images. Extracting features...\n";

        Mat samples; // NxD CV_32F
        Mat responses; // Nx1 CV_32S
        vector<string> featNames;
        bool featNamesSet = false;
        int skippedNoGT = 0, skippedBad = 0, processed = 0;

        std::ofstream csvStream;
        if (!opts.csvOut.empty()) {
            csvStream.open(opts.csvOut);
            if (!csvStream.is_open()) {
                std::cerr << "Could not open CSV output: " << opts.csvOut << "\n";
                return 1;
            }
        }

        for (const auto& p : files) {
            string fname = p.filename().string();
            int label = -1;
            if (!parseGTfromFilenameInt(fname, label)) {
                skippedNoGT++;
                continue;
            }

            Mat I = imread(p.string(), IMREAD_COLOR);
            if (I.empty()) { skippedBad++; continue; }

            vector<double> feat;
            vector<string> names;
            FeatureExtractor::ExtractColorShapeFeatures(I, feat, names);
            if (feat.size() != 12) { skippedBad++; continue; }

            if (!featNamesSet) {
                featNames = names;
                featNamesSet = true;
                if (csvStream.is_open()) {
                    csvStream << "label";
                    for (const auto& n : featNames) csvStream << "," << n;
                    csvStream << "\n";
                }
            }

            Mat row(1, static_cast<int>(feat.size()), CV_32F);
            for (size_t k = 0; k < feat.size(); ++k) row.at<float>(0, static_cast<int>(k)) = static_cast<float>(feat[k]);

            samples.push_back(row);
            responses.push_back(Mat(1, 1, CV_32S, Scalar(label)));

            if (csvStream.is_open()) {
                csvStream << label;
                for (double v : feat) csvStream << "," << v;
                csvStream << "\n";
            }

            processed++;
            if (processed % 200 == 0) std::cout << "Processed " << processed << " images...\n";
        }

        if (csvStream.is_open()) csvStream.close();

        std::cout << "Feature extraction done. Processed: " << processed << ", skipped no-GT: " << skippedNoGT << ", bad: " << skippedBad << "\n";
        if (processed == 0) { std::cerr << "No training samples. Aborting.\n"; return 1; }

        samples.convertTo(samples, CV_32F);
        responses = responses.reshape(1, responses.rows);
        responses.convertTo(responses, CV_32S);

        // optional z-score normalization
        Mat meanVec, stdVec;
        if (opts.doScale) {
            meanVec = Mat(1, samples.cols, CV_64F);
            stdVec = Mat(1, samples.cols, CV_64F);
            for (int c = 0; c < samples.cols; ++c) {
                Scalar mu, stddev;
                Mat col = samples.col(c);
                meanStdDev(col, mu, stddev);
                double s = stddev[0];
                if (s <= 1e-12) s = 1.0; // avoid div by zero
                meanVec.at<double>(0, c) = mu[0];
                stdVec.at<double>(0, c) = s;
                for (int r = 0; r < samples.rows; ++r) {
                    samples.at<float>(r, c) = static_cast<float>((samples.at<float>(r, c) - mu[0]) / s);
                }
            }
            // save scaler
            string scalerPath = opts.outModelPath + "_scaler.yml";
            FileStorage fsSc(scalerPath, FileStorage::WRITE);
            if (fsSc.isOpened()) {
                fsSc << "mean" << meanVec;
                fsSc << "std" << stdVec;
                fsSc.release();
                std::cout << "Saved scaler to: " << scalerPath << "\n";
            }
            else {
                std::cerr << "Warning: could not save scaler to: " << scalerPath << "\n";
            }
        }

        // config SVM polinomio grado 2
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::POLY);
        svm->setDegree(2);
        svm->setC(opts.C);
        if (opts.gamma > 0.0) svm->setGamma(opts.gamma);
        else svm->setGamma(1.0 / static_cast<double>(samples.cols));
        svm->setCoef0(0.0);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));

        std::cout << "Training quadratic SVM (degree=2) on " << samples.rows << " samples, " << samples.cols << " features...\n";
        bool ok = svm->train(samples, ROW_SAMPLE, responses);
        if (!ok) { std::cerr << "SVM training failed.\n"; return 1; }

        try {
            svm->save(opts.outModelPath);
            std::cout << "Saved SVM model to: " << opts.outModelPath << "\n";
        }
        catch (std::exception& e) {
            std::cerr << "Failed saving model: " << e.what() << "\n";
            return 1;
        }

        std::cout << "Done.\n";
        return 0;
    }

} // namespace TrainSVM

int main(int argc, char** argv)
{
    if (argc < 3) {
        printUsage();
        return 1;
    }

    TrainSVM::Options opts;
    opts.inputFolder = argv[1];
    opts.outModelPath = argv[2];
    opts.C = 1.0;
    opts.gamma = 0.0;
    opts.csvOut.clear();
    opts.doScale = false;

    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        if (a == "--csv" && i + 1 < argc) { opts.csvOut = argv[++i]; }
        else if (a == "--C" && i + 1 < argc) { opts.C = atof(argv[++i]); }
        else if (a == "--gamma" && i + 1 < argc) { opts.gamma = atof(argv[++i]); }
        else if (a == "--scale") { opts.doScale = true; }
        else { std::cerr << "Unknown arg: " << a << "\n"; printUsage(); return 1; }
    }

    return TrainSVM::TrainQuadraticSVM(opts);
}