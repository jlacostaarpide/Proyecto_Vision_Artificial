// ClasificacionUnificada.cpp
// Único ejecutable con subcomandos: train, eval, extract
// Compilable con C++14. Requiere ExtractCaracteristicas.cpp/h en el mismo proyecto.

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

#include "ExtractCaracteristicas.h"
#include "Clasificador.h"

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

// ------------------------- TRAIN (copiado/adaptado de entrenarSVM.cpp) -------------------------
int RunTrain(const TrainSVM::Options& opts) {

    fs::path inPath = fs::path(QString::fromStdString(opts.inputFolder).toStdWString());

    // validar carpeta
    if (!fs::exists(inPath) || !fs::is_directory(inPath)) {
        qCritical() << "Input folder not found or not a directory:" << QString::fromStdWString(inPath.wstring());
        return 1;
    }

    // listar imágenes
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

// ------------------------- EVALUATE (adaptación compacta de EvaluacionClasificador.cpp) -------------------------
static void printEvalUsage() {
    qDebug() << "Usage: eval <segFolder> <outTxt> <modelM.yml> [modelM_scaler.yml] [--refine <model912.yml> [model912_scaler.yml]] [--templates <templatesFolder>]";
}

int RunEval(int argc, char** argv) {
    if (argc < 4) { printEvalUsage(); return 1; }

    qDebug("EMPIEZA EVAL");

    QString segFolderQ = QString::fromUtf8(argv[1]);
    fs::path segPath = fs::path(segFolderQ.toStdWString());

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

    if (!fs::exists(segPath) || !fs::is_directory(segPath)) {
        qCritical() << "segFolder not found:" << segFolderQ;
        return 1;
    }

    // list files
    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(segPath)) {
        if (!e.is_regular_file()) continue;
        if (hasSupportedExt(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    int N = static_cast<int>(files.size());
    if (N == 0) { qCritical() << "No images found in:" << segFolderQ; return 1; }

    // load SVM M
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

    // load refiner if requested
    Ptr<SVM> svm912;
    Mat mean912, std912; bool hasScaler912 = false;
    if (useRefiner) {
        try { svm912 = Algorithm::load<SVM>(model912); }
        catch (const cv::Exception& e) { qCritical() << "Failed loading refiner:" << e.what(); return 1; }
        if (!model912scaler.empty()) hasScaler912 = loadScalerIfExists(model912scaler, mean912, std912);
        else { // And similarly for model912 (if you use the refiner)
            fs::path p912(model912);
            string tryPath912 = (p912.parent_path() / (p912.stem().string() + "_scaler.yml")).string();
            hasScaler912 = loadScalerIfExists(tryPath912, mean912, std912);
        }
    }

    std::ofstream fout(outTxt);
    if (!fout.is_open()) { qCritical() << "Cannot create output:" << QString::fromStdString(outTxt); return 1; }

    int nOK = 0, nFail = 0, nMissing = 0, nNoGT = 0;
    int nRef912 = 0, nFlip912 = 0;
    std::vector<string> failList;
    std::vector<string> changeList;

    for (int i = 0; i < N; ++i) {
        string imgName = files[i].filename().string();
        fs::path imgPath = files[i];

        std::regex rx(R"(^(\d{1,2}))");
        std::smatch m;
        bool hasGT = false;
        string trueLabelStr = "---";
        if (std::regex_search(imgName, m, rx) && m.size() >= 2) {
            try {
                int v = std::stoi(m[1].str());
                std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << v;
                trueLabelStr = ss.str();
                hasGT = true;
            }
            catch (...) { hasGT = false; }
        }
        if (!hasGT) nNoGT++;

        if (!fs::exists(imgPath)) { fout << "[" << (i + 1) << "/" << N << "] " << imgName << " | REAL=" << trueLabelStr << " | PRED=--- | ERROR: NO FILE\n"; nMissing++; continue; }

        Mat Ipiece = imread(imgPath.string(), IMREAD_COLOR);
        if (Ipiece.empty()) { fout << "[" << (i + 1) << "/" << N << "] " << imgName << " | REAL=" << trueLabelStr << " | PRED=--- | ERROR: CANNOT READ\n"; nMissing++; continue; }

        // BASE
        vector<double> feat12; vector<string> tmpNames;
        FeatureExtractor::ExtractColorShapeFeatures(Ipiece, feat12, tmpNames);
        int predBase = predictWithSVM(svmM, meanM, stdM, hasScalerM, feat12);
        string predBaseStr = (predBase < 0) ? string("---") : string();
        if (predBase >= 0) { std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << predBase; predBaseStr = ss.str(); }

        string predFinalStr = predBaseStr;
        string refinador = "none";

        // REFINER 9-12
        if (useRefiner && predBase >= 0 && (predBase == 9 || predBase == 12)) {
            nRef912++;
            vector<double> featShape; vector<string> shapeNames;
            FeatureExtractor::ExtractShapeFeatures(Ipiece, featShape, shapeNames); // 14 features
            int predRef = predictWithSVM(svm912, mean912, std912, hasScaler912, featShape);
            string predRefStr = (predRef < 0) ? string("---") : string();
            if (predRef >= 0) { std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << predRef; predRefStr = ss.str(); }
            predFinalStr = predRefStr;
            refinador = "9-12";
            if (predFinalStr != predBaseStr) {
                nFlip912++;
                std::ostringstream sschg;
                sschg << imgName << " | REAL=" << trueLabelStr << " | BASE=" << predBaseStr << " -> FINAL=" << predFinalStr << " | REF=" << refinador;
                changeList.push_back(sschg.str());
            }
        }

        bool isCorrect = true;
        if (hasGT) isCorrect = (predFinalStr == trueLabelStr);

        if (hasGT) {
            if (isCorrect) { nOK++; }
            else { nFail++; std::ostringstream ss; ss << imgName << " | REAL=" << trueLabelStr << " | BASE=" << predBaseStr << " | FINAL=" << predFinalStr << " | REF=" << refinador; failList.push_back(ss.str()); }
        }

        fout << "[" << (i + 1) << "/" << N << "] " << imgName << " | REAL=" << trueLabelStr << " | PRED_BASE=" << predBaseStr << " | PRED_FINAL=" << predFinalStr << " | REF=" << refinador;
        fout << " | " << (hasGT ? (isCorrect ? "OK" : "FAIL") : "NO_GT") << "\n";

        if ((i + 1) % 50 == 0 || i == N - 1) qDebug() << "Procesadas" << (i + 1) << "/" << N;
    }

    int totalEvaluated = nOK + nFail;
    double acc = 0.0;
    if (totalEvaluated > 0) acc = 100.0 * (double(nOK) / double(totalEvaluated));

    fout << "\n\nRESUMEN\n";
    fout << "Total imágenes carpeta     : " << N << "\n";
    fout << "Imágenes no encontradas    : " << nMissing << "\n";
    fout << "Imágenes sin GT en nombre  : " << nNoGT << "\n";
    fout << "Evaluadas (con GT)         : " << totalEvaluated << "\n";
    fout << "Aciertos                   : " << nOK << "\n";
    fout << "Fallos                     : " << nFail << "\n";
    fout << "Accuracy (solo con GT)     : " << std::fixed << std::setprecision(2) << acc << " %\n\n";

    fout << "USO REFINADOR 9-12\n";
    fout << "Ref 9-12 usado            : " << nRef912 << "\n";
    fout << "Cambios BASE->FINAL (flip): " << nFlip912 << "\n\n";

    fout << "LISTA DE FALLOS\n";
    if (nFail == 0) fout << "Ninguno.\n";
    else for (auto& s : failList) fout << s << "\n";

    fout.close();

    qDebug() << "Hecho. TXT guardado en:" << QString::fromStdString(outTxt);
    qDebug() << "Aciertos:" << nOK << "| Fallos:" << nFail << "| Acc:" << QString::number(acc, 'f', 2) + "%" << "| Ref9-12 usado:" << nRef912 << "| flips:" << nFlip912;
    return 0;
}

// ------------------------- EXTRACT (pequeño test) -------------------------
int RunExtractTest(int argc, char** argv) {
    if (argc < 2) { qDebug() << "Usage: extract <image>"; return 1; }
    Mat I = imread(argv[1], IMREAD_COLOR);
    if (I.empty()) { qCritical() << "Cannot read image"; return 1; }
    vector<double> feat; vector<string> names;
    FeatureExtractor::ExtractColorShapeFeatures(I, feat, names);
    qDebug() << "Extracted" << static_cast<int>(feat.size()) << "features:";
    for (size_t i = 0; i < feat.size(); ++i) {
        qDebug() << QString::fromStdString(names[i]) << "=" << feat[i];
    }
    return 0;
}
 // Train a refiner SVM using only images whose GT is 9 or 12 and using shape features.
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
        FeatureExtractor::ExtractShapeFeatures(I, feat, names); // shape-only
        if (feat.empty()) { skipped++; continue; }

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
            if (!ok) continue;

            Mat sampleRow;
            samples.row(loo).convertTo(sampleRow, CV_32F);
            float pred = svm->predict(sampleRow);
            int ipred = static_cast<int>(pred);
            int igt = responses.at<int>(loo, 0);
            if (ipred == igt) correct++;
        }
        double acc = 100.0 * double(correct) / double(samples.rows);
        qDebug() << "LOO accuracy:" << correct << "/" << samples.rows << "(" << QString::number(acc, 'f', 2) << "% )";
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