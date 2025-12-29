// carga el modelo base M (y su scaler si existe), opcionalmente carga un refinador (model_912) y evalúa una carpeta completa:
// por cada imagen extrae las features, predice con M, si M predice clase 9 aplica el refinador (con ExtractShapeFeatures),
// genera logging por imagen y un TXT resumen con accuracy, cambios, fallos, etc.

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <algorithm>
#include <iomanip>

#include "ExtractCaracteristicas.h"

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

static std::pair<bool, string> parseGTfromFilename(const string& fname) {
    std::regex rx(R"(^(\d{1,2}))");
    std::smatch m;
    if (!std::regex_search(fname, m, rx)) return { false, "" };
    try {
        int v = std::stoi(m[1].str());
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(2) << v;
        return { true, ss.str() };
    }
    catch (...) {
        return { false, "" };
    }
}

static bool loadScalerIfExists(const string& scalerPath, Mat& mean, Mat& stdv) {
    mean.release(); stdv.release();
    if (!fs::exists(scalerPath)) return false;
    FileStorage fsr(scalerPath, FileStorage::READ);
    if (!fsr.isOpened()) return false;
    fsr["mean"] >> mean;
    fsr["std"] >> stdv;
    fsr.release();
    if (mean.empty() || stdv.empty()) return false;
    mean.convertTo(mean, CV_64F);
    stdv.convertTo(stdv, CV_64F);
    return true;
}

static int predictWithSVM(const Ptr<SVM>& svm, const Mat& mean, const Mat& stdv, bool hasScaler, const vector<double>& feat) {
    if (!svm) return -1;
    Mat rowD(1, static_cast<int>(feat.size()), CV_64F);
    for (size_t i = 0; i < feat.size(); ++i) rowD.at<double>(0, (int)i) = feat[i];

    if (hasScaler && !mean.empty() && !stdv.empty() && mean.cols == rowD.cols) {
        for (int c = 0; c < rowD.cols; ++c) {
            double mu = mean.at<double>(0, c);
            double s = std::max(1e-12, stdv.at<double>(0, c));
            rowD.at<double>(0, c) = (rowD.at<double>(0, c) - mu) / s;
        }
    }

    Mat rowF;
    rowD.convertTo(rowF, CV_32F);
    float r = svm->predict(rowF);
    return static_cast<int>(r);
}

static void printUsage() {
    std::cout << "Usage: evaluate_classifiers <segFolder> <outTxt> <modelM.yml> [modelM_scaler.yml] [--refine <model912.yml> [model912_scaler.yml]]\n";
}

int main(int argc, char** argv) {
    if (argc < 4) { printUsage(); return 1; }

    string segFolder = argv[1];
    string outTxt = argv[2];
    string modelM = argv[3];
    string modelMscaler;
    string model912, model912scaler;
    bool useRefiner = false;

    int idx = 4;
    if (idx < argc && argv[idx][0] != '-') {
        modelMscaler = argv[idx++];
    }
    // parse optional --refine
    for (; idx < argc; ++idx) {
        string a = argv[idx];
        if (a == "--refine" && idx + 1 < argc) {
            useRefiner = true;
            model912 = argv[++idx];
            if (idx + 1 < argc && argv[idx + 1][0] != '-') {
                model912scaler = argv[++idx];
            }
            continue;
        }
    }

    if (!fs::exists(segFolder) || !fs::is_directory(segFolder)) {
        std::cerr << "segFolder not found: " << segFolder << "\n";
        return 1;
    }

    // list files
    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(segFolder)) {
        if (!e.is_regular_file()) continue;
        if (hasSupportedExt(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    int N = static_cast<int>(files.size());
    if (N == 0) { std::cerr << "No images found in: " << segFolder << "\n"; return 1; }

    // load SVM M
    Ptr<SVM> svmM;
    try { svmM = Algorithm::load<SVM>(modelM); }
    catch (const cv::Exception& e) { std::cerr << "Failed loading modelM: " << e.what() << "\n"; return 1; }
    Mat meanM, stdM; bool hasScalerM = false;
    if (!modelMscaler.empty()) hasScalerM = loadScalerIfExists(modelMscaler, meanM, stdM);
    else { string tryPath = modelM + "_scaler.yml"; hasScalerM = loadScalerIfExists(tryPath, meanM, stdM); }

    // load refiner if requested
    Ptr<SVM> svm912;
    Mat mean912, std912; bool hasScaler912 = false;
    if (useRefiner) {
        try { svm912 = Algorithm::load<SVM>(model912); }
        catch (const cv::Exception& e) { std::cerr << "Failed loading refiner: " << e.what() << "\n"; return 1; }
        if (!model912scaler.empty()) hasScaler912 = loadScalerIfExists(model912scaler, mean912, std912);
        else { string tryPath = model912 + "_scaler.yml"; hasScaler912 = loadScalerIfExists(tryPath, mean912, std912); }
    }

    // abrir TXT
    std::ofstream fout(outTxt);
    if (!fout.is_open()) { std::cerr << "Cannot create output: " << outTxt << "\n"; return 1; }

    fout << "EVALUACIÓN SOBRE CARPETA (modelo M + cascada 9-12)\n";
    fout << "==================================================\n\n";
    fout << "Carpeta imágenes: " << segFolder << "\n\n";

    int nOK = 0, nFail = 0, nMissing = 0, nNoGT = 0;
    int nRef912 = 0, nFlip912 = 0;
    std::vector<string> failList;
    std::vector<string> changeList;

    // predictor names and shape feat names (for potential CSV or debug)
    vector<string> predictorNames = { "Extent","Solidity","V_mean","Eccentricity","SkelLenNorm","Circularity",
        "H_mean_circ","S_mean","V_IQR","S_median","FD5","EulerNumber" };
    vector<string> featNames_shape = {
        "Circularity","AspectRatio","Extent","Solidity","ConvexFrac","Eccentricity","EulerNumber",
        "SkelLenNorm","SkelEndpoints","SkelBranchpoints","FD2","FD3","FD4","FD5"
    };

    for (int i = 0; i < N; ++i) {
        string imgName = files[i].filename().string();
        string imgPath = files[i].string();

        auto gtPair = parseGTfromFilename(imgName);
        bool hasGT = gtPair.first;
        string trueLabelStr = hasGT ? gtPair.second : "---";
        if (!hasGT) nNoGT++;

        if (!fs::exists(imgPath)) {
            fout << "[" << (i + 1) << "/" << N << "] " << imgName << " | REAL=" << trueLabelStr << " | PRED=--- | ERROR: NO FILE\n";
            nMissing++;
            continue;
        }

        Mat Ipiece = imread(imgPath, IMREAD_COLOR);
        if (Ipiece.empty()) {
            fout << "[" << (i + 1) << "/" << N << "] " << imgName << " | REAL=" << trueLabelStr << " | PRED=--- | ERROR: CANNOT READ\n";
            nMissing++;
            continue;
        }

        // 1) BASE
        vector<double> feat12; vector<string> tmpNames;
        FeatureExtractor::ExtractColorShapeFeatures(Ipiece, feat12, tmpNames);
        int predBase = predictWithSVM(svmM, meanM, stdM, hasScalerM, feat12);
        string predBaseStr;
        if (predBase < 0) predBaseStr = "---";
        else {
            std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << predBase; predBaseStr = ss.str();
        }
        string predFinalStr = predBaseStr;
        string refinador = "none";

        // 2) CASCADA AMARILLO 9-12 (refinador)
        if (useRefiner && predBase >= 0 && predBase == 9) {
            nRef912++;
            vector<double> featShape; vector<string> shapeNames;
            FeatureExtractor::ExtractShapeFeatures(Ipiece, featShape, shapeNames); // devuelve 14 features
            int predRef = predictWithSVM(svm912, mean912, std912, hasScaler912, featShape);
            string predRefStr = (predRef < 0) ? "---" : ((std::ostringstream() << std::setfill('0') << std::setw(2) << predRef, std::ostringstream().str()));
            // above line built unsafely; build properly:
            if (predRef < 0) predRefStr = "---";
            else { std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << predRef; predRefStr = ss.str(); }

            predFinalStr = predRefStr;
            refinador = "9-12";

            if (predFinalStr != predBaseStr) {
                nFlip912++;
                std::ostringstream sschg;
                sschg << imgName << " | REAL=" << trueLabelStr << " | BASE=" << predBaseStr << " -> FINAL=" << predFinalStr << " | REF=" << refinador;
                changeList.push_back(sschg.str());
            }
        }

        // 3) comparar
        bool isCorrect = true;
        if (hasGT) isCorrect = (predFinalStr == trueLabelStr);

        string resultStr;
        if (hasGT) {
            if (isCorrect) { nOK++; resultStr = "OK"; }
            else {
                nFail++; resultStr = "FAIL";
                std::ostringstream ss; ss << imgName << " | REAL=" << trueLabelStr << " | BASE=" << predBaseStr << " | FINAL=" << predFinalStr << " | REF=" << refinador;
                failList.push_back(ss.str());
            }
        }
        else resultStr = "NO_GT";

        // 4) log por imagen
        fout << "[" << (i + 1) << "/" << N << "] " << imgName
            << " | REAL=" << trueLabelStr
            << " | PRED_BASE=" << predBaseStr
            << " | PRED_FINAL=" << predFinalStr
            << " | REF=" << refinador
            << " | " << resultStr << "\n";

        if ((i + 1) % 50 == 0 || i == N - 1) {
            std::cout << "Procesadas " << (i + 1) << "/" << N << "\n";
        }
    }

    int totalEvaluated = nOK + nFail;
    double acc = 0.0;
    if (totalEvaluated > 0) acc = 100.0 * (double(nOK) / double(totalEvaluated));

    fout << "\n\nRESUMEN\n";
    fout << "------\n";
    fout << "Total imágenes carpeta     : " << N << "\n";
    fout << "Imágenes no encontradas    : " << nMissing << "\n";
    fout << "Imágenes sin GT en nombre  : " << nNoGT << "\n";
    fout << "Evaluadas (con GT)         : " << totalEvaluated << "\n";
    fout << "Aciertos                   : " << nOK << "\n";
    fout << "Fallos                     : " << nFail << "\n";
    fout << "Accuracy (solo con GT)     : " << std::fixed << std::setprecision(2) << acc << " %\n\n";

    fout << "USO REFINADOR 9-12\n";
    fout << "-----------------\n";
    fout << "Ref 9-12 usado            : " << nRef912 << "\n";
    fout << "Cambios BASE->FINAL (flip): " << nFlip912 << "\n\n";

    fout << "LISTA DE FALLOS (si los hay)\n";
    fout << "----------------------------\n";
    if (nFail == 0) fout << "Ninguno.\n";
    else {
        for (auto& s : failList) fout << s << "\n";
    }

    fout << "\nLISTA DE CAMBIOS (BASE -> FINAL)\n";
    fout << "--------------------------------\n";
    if (changeList.empty()) fout << "Ninguno (el refinador nunca cambió la clase).\n";
    else {
        for (auto& s : changeList) fout << s << "\n";
    }

    fout.close();

    std::cout << "\nHecho. TXT guardado en:\n" << outTxt << "\n";
    std::cout << "Aciertos: " << nOK << " | Fallos: " << nFail << " | Acc: " << acc << "% | Ref9-12 usado: " << nRef912 << " | flips: " << nFlip912 << "\n";

    return 0;
}