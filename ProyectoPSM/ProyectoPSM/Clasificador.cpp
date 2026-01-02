// ClasificacionUnificada.cpp
// ï¿½nico ejecutable con subcomandos: train, eval, extract

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <QDebug>
#include <QString>
#include <random>

#include "ExtractCaracteristicas.h"
#include "Clasificador.h"
#include "ExtractCaracteristicas24Refinador.h"
#include "ExtractShapeFeatures6.h"
#include "ExtractShapeFeatures10.h"

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

// load scaler YAML (mean/std) if exists
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

// predict with SVM (optional scaler)
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

// ------------------------- TRAIN  -------------------------
//Classificador global
int RunTrain(const TrainSVM::Options& opts) {

    fs::path inPath = fs::path(QString::fromStdString(opts.inputFolder).toStdWString());

    // validar carpeta
    if (!fs::exists(inPath) || !fs::is_directory(inPath)) {
        qCritical() << "Input folder not found or not a directory:" << QString::fromStdWString(inPath.wstring());
        return 1;
    }

    // listar imï¿½genes
    vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(inPath)) {
        if (!entry.is_regular_file()) continue;
        if (hasSupportedExt(entry.path())) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        qCritical() << "No supported images found in:" << QString::fromStdWString(inPath.wstring());
        return 1;
    }

    qDebug() << "Found" << static_cast<int>(files.size()) << "images. Extracting features...";

    Mat samples; // NxD CV_32F
    Mat responses; // Nx1 CV_32S
    vector<string> featNames;
    bool featNamesSet = false;
    int skippedNoGT = 0, skippedBad = 0, processed = 0;

    std::ofstream csvStream;
    if (!opts.csvOut.empty()) {
        csvStream.open(opts.csvOut);
        if (!csvStream.is_open()) {
            qCritical() << "Could not open CSV output:" << QString::fromStdString(opts.csvOut);
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
        if (processed % 200 == 0) qDebug() << "Processed" << processed << "images...";
    }

    if (csvStream.is_open()) csvStream.close();

    qDebug() << "Feature extraction done. Processed:" << processed << ", skipped no-GT:" << skippedNoGT << ", bad:" << skippedBad;
    if (processed == 0) { qCritical() << "No training samples. Aborting."; return 1; }

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
        string scalerPath = (fs::path(opts.outModelPath).parent_path() /
            (fs::path(opts.outModelPath).stem().string() + "_scaler.yml")).string();
        FileStorage fsSc(scalerPath, FileStorage::WRITE);
        if (fsSc.isOpened()) {
            fsSc << "mean" << meanVec;
            fsSc << "std" << stdVec;
            fsSc.release();
            qDebug() << "Saved scaler to:" << QString::fromStdString(scalerPath);
        }
        else {
            qWarning() << "Warning: could not save scaler to:" << QString::fromStdString(scalerPath);
        }
    }

    // ---------------------------
    // Sustituye LOO por Grid-Search K-fold (RBF)
    // Ejecuta cuando opts.doGridSearch == true (comportamiento cambiado: ahora hace grid-search)
    // ---------------------------
    if (opts.doGridSearch && samples.rows > 1) {
        // grid values (ajusta según necesites)
        std::vector<double> Cvals = { 0.1, 1, 10, 100 };
        std::vector<double> gammaVals = { 0.001, 0.01, 0.1, 1 };
        int K = 5;
        int N = samples.rows;
        if (K > N) K = N;

        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        // shuffle fijo para reproducibilidad
        std::mt19937 rng(1234);
        std::shuffle(indices.begin(), indices.end(), rng);

        auto crossValidateRBF = [&](double C, double gamma) -> double {
            int correct = 0;
            int total = 0;

            for (int k = 0; k < K; ++k) {
                cv::Mat trainS, trainR, testS, testR;

                for (int i = 0; i < N; ++i) {
                    int idx = indices[i];
                    if ((i % K) == k) {
                        testS.push_back(samples.row(idx));
                        testR.push_back(responses.row(idx));
                    }
                    else {
                        trainS.push_back(samples.row(idx));
                        trainR.push_back(responses.row(idx));
                    }
                }

                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
                svm->setType(cv::ml::SVM::C_SVC);
                svm->setKernel(cv::ml::SVM::RBF);
                svm->setC(C);
                svm->setGamma(gamma);
                svm->setTermCriteria(
                    cv::TermCriteria(
                        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                        2000, 1e-6));

                bool okTrain = svm->train(trainS, cv::ml::ROW_SAMPLE, trainR);
                if (!okTrain) continue;

                for (int ti = 0; ti < testS.rows; ++ti) {
                    int pred = static_cast<int>(svm->predict(testS.row(ti)));
                    int gt = testR.at<int>(ti, 0);
                    if (pred == gt) correct++;
                    total++;
                }
            }

            return (total > 0) ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
            };

        double bestAcc = 0.0;
        double bestC = Cvals.front();
        double bestGamma = gammaVals.front();

        for (double C : Cvals) {
            for (double gamma : gammaVals) {

                double acc = crossValidateRBF(C, gamma);

                qDebug() << "GridSearch: C =" << C
                    << "gamma =" << gamma
                    << "CV acc =" << acc << "%";

                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestC = C;
                    bestGamma = gamma;
                }
            }
        }

        qDebug() << "==============================";
        qDebug() << "BEST PARAMETERS (grid-search):";
        qDebug() << "C =" << bestC;
        qDebug() << "gamma =" << bestGamma;
        qDebug() << "CV accuracy =" << bestAcc << "%";

        // Entrena modelo final RBF con mejores hiperparámetros
        Ptr<SVM> svmRBF = SVM::create();
        svmRBF->setType(SVM::C_SVC);
        svmRBF->setKernel(SVM::RBF);
        svmRBF->setC(bestC);
        svmRBF->setGamma(bestGamma);
        svmRBF->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));

        qDebug() << "Training final RBF SVM on" << samples.rows << "samples," << samples.cols << "features...";
        bool trainOk = svmRBF->train(samples, ROW_SAMPLE, responses);
        if (!trainOk) { qCritical() << "SVM training failed."; return 1; }

        try {
            svmRBF->save(opts.outModelPath);
            qDebug() << "Saved SVM model to:" << QString::fromStdString(opts.outModelPath);
        }
        catch (std::exception& e) {
            qCritical() << "Failed saving model:" << e.what();
            return 1;
        }

        qDebug() << "Done (grid-search + train).";
        return 0;
    }

    // Si no se pidió grid-search (opts.doLOO == false), se entrena el SVM polinómico como antes.
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY);
    svm->setDegree(2);
    svm->setC(opts.C);
    if (opts.gamma > 0.0) svm->setGamma(opts.gamma);
    else svm->setGamma(1.0 / static_cast<double>(samples.cols));
    svm->setCoef0(0.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));

    qDebug() << "Training quadratic SVM (degree=2) on" << samples.rows << "samples," << samples.cols << "features...";
    bool ok = svm->train(samples, ROW_SAMPLE, responses);
    if (!ok) { qCritical() << "SVM training failed."; return 1; }

    try {
        svm->save(opts.outModelPath);
        qDebug() << "Saved SVM model to:" << QString::fromStdString(opts.outModelPath);
    }
    catch (std::exception& e) {
        qCritical() << "Failed saving model:" << e.what();
        return 1;
    }

    qDebug() << "Done.";
    return 0;
}

//Clasificador refiner (9 vs 12) piezas amarillas
int RunTrainRefiner(const TrainSVM::Options& opts, bool doLOO) {

    // If doLOO==true computes leave-one-out accuracy (printed) before training final model.
    fs::path inPath = fs::path(QString::fromStdString(opts.inputFolder).toStdWString());

    if (!fs::exists(inPath) || !fs::is_directory(inPath)) {
        qCritical() << "Input folder not found or not a directory:" << QString::fromStdWString(inPath.wstring());
        return 1;
    }

    vector<fs::path> files;
    for (auto& entry : fs::directory_iterator(inPath)) {
        if (!entry.is_regular_file()) continue;
        if (!hasSupportedExt(entry.path())) continue;
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        qCritical() << "No supported images found in:" << QString::fromStdString(opts.inputFolder);
        return 1;
    }

    Mat samples;        // NxD CV_32F
    Mat responses;      // Nx1 CV_32S
    vector<string> featNames;
    bool featNamesSet = false;
    int skipped = 0, processed = 0;

    for (const auto& p : files) {
        string fname = p.filename().string();
        int label = -1;
        if (!parseGTfromFilenameInt(fname, label)) { skipped++; continue; }
        if (!(label == 9 || label == 12)) { skipped++; continue; } // only 9 or 12

        Mat I = imread(p.string(), IMREAD_COLOR);
        if (I.empty()) { skipped++; continue; }

        vector<double> feat;
        vector<string> names;
        //FeatureExtractor24::ExtractShapeFeatures24(I, feat, names);
        //FeatureExtractor6::ExtractShapeFeatures6(I, feat, names);
        FeatureExtractor10::ExtractShapeFeatures10(I, feat, names);

        if (feat.size() != 10) { skipped++; continue; } // 24 si se usa el otro

        if (!featNamesSet) { featNames = names; featNamesSet = true; }

        Mat row(1, static_cast<int>(feat.size()), CV_32F);
        for (size_t k = 0; k < feat.size(); ++k) row.at<float>(0, static_cast<int>(k)) = static_cast<float>(feat[k]);

        samples.push_back(row);
        responses.push_back(Mat(1, 1, CV_32S, Scalar(label)));
        processed++;
        if (processed % 50 == 0) qDebug() << "Processed" << processed << "images...";
    }

    qDebug() << "Refiner dataset: processed =" << processed << ", skipped =" << skipped;
    if (processed == 0) { qCritical() << "No samples for refiner (need images starting with 9 or 12)."; return 1; }

    samples.convertTo(samples, CV_32F);
    responses = responses.reshape(1, responses.rows);
    responses.convertTo(responses, CV_32S);

    // Optional scaling
    Mat meanVec, stdVec;
    bool hasScaler = false;
    if (opts.doScale) {
        meanVec = Mat(1, samples.cols, CV_64F);
        stdVec = Mat(1, samples.cols, CV_64F);
        for (int c = 0; c < samples.cols; ++c) {
            Scalar mu, stddev;
            Mat col = samples.col(c);
            meanStdDev(col, mu, stddev);
            double s = stddev[0];
            if (s <= 1e-12) s = 1.0;
            meanVec.at<double>(0, c) = mu[0];
            stdVec.at<double>(0, c) = s;
            for (int r = 0; r < samples.rows; ++r) {
                samples.at<float>(r, c) = static_cast<float>((samples.at<float>(r, c) - mu[0]) / s);
            }
        }
        // save scaler using stem + _scaler.yml
        string scalerPath = (fs::path(opts.outModelPath).parent_path() /
            (fs::path(opts.outModelPath).stem().string() + "_scaler.yml")).string();
        FileStorage fsSc(scalerPath, FileStorage::WRITE);
        if (fsSc.isOpened()) {
            fsSc << "mean" << meanVec;
            fsSc << "std" << stdVec;
            fsSc.release();
            qDebug() << "Saved refiner scaler to:" << QString::fromStdString(scalerPath);
            hasScaler = true;
        }
        else {
            qWarning() << "Warning: could not save refiner scaler to:" << QString::fromStdString(scalerPath);
        }
    }

    // Leave-one-out cross-validation (optional)
    if (doLOO && samples.rows > 1) {
        int correct = 0;
        int attempted = 0;
        int failTrain = 0;
        qDebug() << "Starting leave-one-out (N =" << samples.rows << ") - this may be slow...";
        for (int loo = 0; loo < samples.rows; ++loo) {
            // build train set excluding loo
            Mat trainS, trainR;
            for (int r = 0; r < samples.rows; ++r) {
                if (r == loo) continue;
                trainS.push_back(samples.row(r));
                trainR.push_back(responses.row(r));
            }

            // configure SVM (same hyperparams as main)
            Ptr<SVM> svm = SVM::create();
            svm->setType(SVM::C_SVC);
            svm->setKernel(SVM::POLY);
            svm->setDegree(2);
            svm->setC(opts.C);
            if (opts.gamma > 0.0) svm->setGamma(opts.gamma);
            else svm->setGamma(1.0 / static_cast<double>(trainS.cols));
            svm->setCoef0(0.0);
            svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));

            bool ok = svm->train(trainS, ROW_SAMPLE, trainR);
            if (!ok) { ++failTrain; continue; }
            ++attempted;

            Mat sampleRow;
            samples.row(loo).convertTo(sampleRow, CV_32F);
            float pred = svm->predict(sampleRow);
            int ipred = static_cast<int>(pred);
            int igt = responses.at<int>(loo, 0);
            if (ipred == igt) correct++;

            if ((loo % 50) == 0) qDebug() << "LOO progress:" << loo << "/" << samples.rows;
        }
        double acc = attempted > 0 ? (100.0 * double(correct) / double(attempted)) : 0.0;
        qDebug() << "LOO accuracy:" << correct << "/" << attempted << "(" << QString::number(acc, 'f', 2) << "% )"
            << "| failed train calls:" << failTrain;
    }

    // Train final refiner on full data and save
    Ptr<SVM> svmFinal = SVM::create();
    svmFinal->setType(SVM::C_SVC);
    svmFinal->setKernel(SVM::POLY);
    svmFinal->setDegree(2);
    svmFinal->setC(opts.C);
    if (opts.gamma > 0.0) svmFinal->setGamma(opts.gamma);
    else svmFinal->setGamma(1.0 / static_cast<double>(samples.cols));
    svmFinal->setCoef0(0.0);
    svmFinal->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 2000, 1e-6));

    qDebug() << "Training refiner SVM on" << samples.rows << "samples," << samples.cols << "features...";
    bool ok = svmFinal->train(samples, ROW_SAMPLE, responses);
    if (!ok) { qCritical() << "Refiner SVM training failed."; return 1; }

    try {
        svmFinal->save(opts.outModelPath);
        qDebug() << "Saved refiner model to:" << QString::fromStdString(opts.outModelPath);
    }
    catch (std::exception& e) {
        qCritical() << "Failed saving refiner model:" << e.what();
        return 1;
    }

    qDebug() << "Refiner training done.";
    return 0;
}

// ------------------------- EVALUATE (adaptaciï¿½n compacta de EvaluacionClasificador.cpp) -------------------------
static void printEvalUsage() {
    qDebug() << "Usage: eval <segFolder> <outTxt> <modelM.yml> [modelM_scaler.yml] [--refine <model912.yml> [model912_scaler.yml]] [--templates <templatesFolder>]";
}

int RunEval(int argc, char** argv) {
    if (argc < 4) { printEvalUsage(); return 1; }

    qDebug("EMPIEZA EVAL");

    string segFolder = argv[1];
    string outTxt = argv[2];
    string modelM = argv[3];
    string modelMscaler;
    string model912, model912scaler;
    string templatesFolder;
    bool useRefiner = false;

    int idx = 4;
    if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
        modelMscaler = argv[idx++];
    }
    for (; idx < argc; ++idx) {
        string a = argv[idx];
        if (a == "--refine" && idx + 1 < argc) {
            useRefiner = true;
            model912 = argv[++idx];
            if (idx + 1 < argc && std::string(argv[idx + 1]).rfind("--", 0) != 0) {
                model912scaler = argv[++idx];
            }
        }
        else if (a == "--templates" && idx + 1 < argc) {
            templatesFolder = argv[++idx];
        }
        else {
            qCritical() << "Unknown arg:" << QString::fromStdString(a);
            printEvalUsage();
            return 1;
        }
    }

    if (!fs::exists(segFolder) || !fs::is_directory(segFolder)) {
        qCritical() << "segFolder not found:" << QString::fromStdString(segFolder);
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
    if (N == 0) { qCritical() << "No images found in:" << QString::fromStdString(segFolder); return 1; }

    // load main SVM model
    Ptr<SVM> svmM;
    try { svmM = Algorithm::load<SVM>(modelM); }
    catch (const cv::Exception& e) { qCritical() << "Failed loading modelM:" << e.what(); return 1; }
    Mat meanM, stdM; bool hasScalerM = false;
    if (!modelMscaler.empty()) hasScalerM = loadScalerIfExists(modelMscaler, meanM, stdM);
    else {
        fs::path pm(modelM);
        string tryPath = (pm.parent_path() / (pm.stem().string() + "_scaler.yml")).string();
        hasScalerM = loadScalerIfExists(tryPath, meanM, stdM);
    }

    // auto-detect refiner model near modelM if user didn't pass --refine
    if (!useRefiner) {
        try {
            fs::path pm(modelM);
            fs::path dir = pm.parent_path();
            if (fs::exists(dir) && fs::is_directory(dir)) {
                for (auto& entry : fs::directory_iterator(dir)) {
                    if (!entry.is_regular_file()) continue;
                    string name = entry.path().filename().string();
                    string stem = entry.path().stem().string();
                    string name_l = name; std::transform(name_l.begin(), name_l.end(), name_l.begin(), ::tolower);
                    string stem_l = stem; std::transform(stem_l.begin(), stem_l.end(), stem_l.begin(), ::tolower);
                    if (name_l.find("912") != string::npos || name_l.find("refiner") != string::npos ||
                        stem_l.find("912") != string::npos || stem_l.find("refiner") != string::npos) {
                        model912 = entry.path().string();
                        fs::path trySc = entry.path().parent_path() / (entry.path().stem().string() + "_scaler.yml");
                        if (fs::exists(trySc)) model912scaler = trySc.string();
                        useRefiner = true;
                        qDebug() << "Auto-detected refiner model:" << QString::fromStdString(model912)
                            << (model912scaler.empty() ? "" : QString(" (scaler: %1)").arg(QString::fromStdString(model912scaler)));
                        break;
                    }
                }
            }
        }
        catch (const std::exception& e) {
            qWarning() << "Refiner auto-detect failed:" << e.what();
        }
    }

    // load refiner if requested
    Ptr<SVM> svm912;
    Mat mean912, std912; bool hasScaler912 = false;
    if (useRefiner && !model912.empty()) {
        try {
            svm912 = Algorithm::load<SVM>(model912);
            if (!model912scaler.empty()) hasScaler912 = loadScalerIfExists(model912scaler, mean912, std912);
            else {
                fs::path p912(model912);
                string tryPath = (p912.parent_path() / (p912.stem().string() + "_scaler.yml")).string();
                hasScaler912 = loadScalerIfExists(tryPath, mean912, std912);
            }
            if (!svm912) {
                qWarning() << "Refiner model specified but failed to load:" << QString::fromStdString(model912);
                useRefiner = false;
            }
            else {
                qDebug() << "Refiner loaded.";
            }
        }
        catch (const cv::Exception& e) {
            qWarning() << "Failed loading refiner model:" << e.what();
            useRefiner = false;
        }
    }

    // ensure output directory exists (best-effort)
    {
        fs::path outp(outTxt);
        fs::path parent = outp.parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            try { fs::create_directories(parent); }
            catch (...) { qWarning() << "Could not create output directory:" << QString::fromStdString(parent.string()); }
        }
    }

    // open output file
    std::ofstream out(outTxt, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        qCritical() << "Cannot open output file for writing:" << QString::fromStdString(outTxt);
        return 1;
    }

    // write header
    out << "filename,gt,pred\n";

    int total = 0;
    int correct = 0;
    int skipped = 0;

    for (size_t i = 0; i < files.size(); ++i) {
        const fs::path& p = files[i];
        string fname = p.filename().string();

        int gt = -1;
        if (!parseGTfromFilenameInt(fname, gt)) {
            qDebug() << "Skipping (no GT in filename):" << QString::fromStdString(fname);
            skipped++;
            continue;
        }

        Mat I = imread(p.string(), IMREAD_COLOR);
        if (I.empty()) {
            qDebug() << "Skipping (cannot read):" << QString::fromStdString(p.string());
            skipped++;
            continue;
        }

        // Extract combined features for main model
        vector<double> feat;
        vector<string> names;
        FeatureExtractor::ExtractColorShapeFeatures(I, feat, names);
        if (feat.empty()) {
            qDebug() << "Skipping (no features extracted):" << QString::fromStdString(fname);
            skipped++;
            continue;
        }

        int pred = predictWithSVM(svmM, meanM, stdM, hasScalerM, feat);

        if (useRefiner && svm912 && (pred == 9 || pred == 12)) {
            vector<double> featS; vector<string> namesS;
            FeatureExtractor24::ExtractShapeFeatures24(I, featS, namesS);
            if (featS.size() == 24) {
                int predR = predictWithSVM(svm912, mean912, std912, hasScaler912, featS);
                if (predR == 9 || predR == 12) pred = predR;
            }
        }

        out << fname << "," << gt << "," << pred << "\n";

        if (pred == gt) correct++;
        total++;

        if ((i % 50) == 0) qDebug() << "Eval progress:" << i << "/" << N;
    }

    double acc = total > 0 ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;

    // summary
    out << "\n#summary\n";
    out << "total," << total << "\n";
    out << "correct," << correct << "\n";
    out << "skipped," << skipped << "\n";
    out << "accuracy_pct," << std::fixed << std::setprecision(2) << acc << "\n";
    out.close();

    qDebug() << "EVAL finished. total=" << total << " correct=" << correct << " skipped=" << skipped
        << " acc(%)=" << QString::number(acc, 'f', 2);

    return 0;
}

int RunEvalRefinerOnly(const std::string& segFolder,
    const std::string& outTxt,
    const std::string& model912,
    const std::string& model912scaler)
{
    if (!fs::exists(segFolder) || !fs::is_directory(segFolder)) {
        qCritical() << "segFolder not found:" << QString::fromStdString(segFolder);
        return 1;
    }

    qDebug() << "RunEvalRefinerOnly segFolder =" << QString::fromStdString(segFolder);
    qDebug() << "RunEvalRefinerOnly outTxt    =" << QString::fromStdString(outTxt);

    // list files
    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(segFolder)) {
        if (!e.is_regular_file()) continue;
        if (hasSupportedExt(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { qCritical() << "No images found in:" << QString::fromStdString(segFolder); return 1; }

    // load model912
    Ptr<SVM> svm912;
    try { svm912 = Algorithm::load<SVM>(model912); }
    catch (const cv::Exception& e) { qCritical() << "Failed loading model912:" << e.what(); return 1; }

    Mat mean912, std912; bool hasScaler912 = false;
    if (!model912scaler.empty()) hasScaler912 = loadScalerIfExists(model912scaler, mean912, std912);
    else {
        fs::path p(model912);
        hasScaler912 = loadScalerIfExists((p.parent_path() / (p.stem().string() + "_scaler.yml")).string(), mean912, std912);
    }

    // ensure output directory exists
    {
        fs::path outp(outTxt);
        fs::path parent = outp.parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            try { fs::create_directories(parent); }
            catch (...) { qWarning() << "Could not create output dir:" << QString::fromStdString(parent.string()); }
        }
    }

    std::ofstream out(outTxt, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        qCritical() << "Cannot open output:" << QString::fromStdString(outTxt);
        return 1;
    }
    out << "filename,gt,pred\n";

    int total = 0, correct = 0, skipped = 0;

    for (auto& p : files) {
        string fname = p.filename().string();

        int gt = -1;
        if (!parseGTfromFilenameInt(fname, gt) || !(gt == 9 || gt == 12)) {
            skipped++;
            continue;
        }

        Mat I = imread(p.string(), IMREAD_COLOR);
        if (I.empty()) { skipped++; continue; }

        vector<double> feat; vector<string> names;
        //FeatureExtractor24::ExtractShapeFeatures24(I, feat, names);
        //FeatureExtractor6::ExtractShapeFeatures6(I, feat, names);
        FeatureExtractor10::ExtractShapeFeatures10(I, feat, names);

        if (feat.size() != 10) { skipped++; continue; } //24 si se usa el otro
		// predict

        int pred = predictWithSVM(svm912, mean912, std912, hasScaler912, feat);

        out << fname << "," << gt << "," << pred << "\n";
        if (pred == gt) correct++;
        total++;
    }

   

    double acc = total ? 100.0 * double(correct) / double(total) : 0.0;
    out << "\n#summary\n";
    out << "total," << total << "\n";
    out << "correct," << correct << "\n";
    out << "skipped," << skipped << "\n";
    out << "accuracy_pct," << std::fixed << std::setprecision(2) << acc << "\n";
    out.close();

    qDebug() << "EVAL REFiner-only finished. total=" << total
        << " correct=" << correct << " skipped=" << skipped
        << " acc(%)=" << QString::number(acc, 'f', 2);

    return 0;
}
