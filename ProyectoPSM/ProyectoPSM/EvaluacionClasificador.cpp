// Traducción a C++ del script MATLAB 'Testeo_dobleClass_12_24_Carac.m'
// - Evalúa una carpeta con: modelo base M (12 feats) + opcional refinador 9-12 (shape features)
// - Para cada imagen: extrae features, predice con M, si aplica usa refinador, predice orientación con plantillas,
//   genera un TXT con logs por imagen y resumen final.
//
// Uso:
//   evaluate_classifiers <segFolder> <outTxt> <modelM.yml> [modelM_scaler.yml] [--templates <templatesFolder>] [--refine <model912.yml> [model912_scaler.yml]]
//
// Requiere: OpenCV (core, imgproc, imgcodecs, ml, highgui optional) y FeatureExtractor (ExtractCaracteristicas.h/.cpp)
// Compilable con C++14.

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
#include <map>

#include "ExtractCaracteristicas.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
using std::string;
using std::vector;

// soportadas
static const vector<string> exts = { ".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp" };
static bool hasSupportedExt(const fs::path& p) {
    string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return std::find(exts.begin(), exts.end(), e) != exts.end();
}

// helper: extrae GT desde los primeros dígitos del nombre (igual que el MATLAB)
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

// carga scaler YAML (mean/std) si existe
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

// predicción con SVM (aplica scaler opcional)
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

// ------------------- ORIENTACIÓN: templates -------------------
// estructura simple para plantilla
struct TemplateItem {
    int yaw = 0;
    int pitch = 0;
    int size = 0;
    cv::Mat T; // CV_64F matrix size x size
};

// cache de plantillas por código
static std::map<std::string, std::vector<TemplateItem>> g_templatesCache;

// extrae máscara similar a extractMaskLego.m (básico pero funcional)
static cv::Mat extractMaskLego_cpp(const cv::Mat& I) {
    if (I.empty()) return cv::Mat();
    Mat If;
    if (I.type() == CV_8U || I.type() == CV_8UC3) {
        I.convertTo(If, CV_32F, 1.0 / 255.0);
    }
    else {
        I.convertTo(If, CV_32F);
    }

    Mat Ig;
    if (If.channels() == 3) cvtColor(If, Ig, COLOR_BGR2GRAY);
    else Ig = If;

    const double t = 0.03; // umbral
    Mat mask = (Ig > t);

    // limpieza morfológica: bwareaopen (eliminar pequeños) -> implementamos manteniendo contornos grandes
    std::vector<std::vector<Point>> contours;
    Mat maskU8;
    mask.convertTo(maskU8, CV_8U, 255);
    findContours(maskU8.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat cleaned = Mat::zeros(maskU8.size(), CV_8U);
    for (const auto& c : contours) {
        double a = contourArea(c);
        if (a >= 150.0) { // similar a bwareaopen(150)
            drawContours(cleaned, std::vector<std::vector<Point>>{c}, 0, Scalar(255), FILLED);
        }
    }
    mask = (cleaned > 0);

    // close + open + fill holes
    Mat seClose = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat seOpen = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, seClose);
    morphologyEx(mask, mask, MORPH_OPEN, seOpen);

    // rellenar agujeros por floodFill
    Mat im_flood;
    mask.convertTo(im_flood, CV_8U, 255);
    Mat ff = im_flood.clone();
    // ensure border is background; create mask for floodFill (2px border)
    Mat maskFF = Mat::zeros(ff.rows + 2, ff.cols + 2, CV_8U);
    floodFill(ff, maskFF, Point(0, 0), Scalar(255));
    Mat ff_inv;
    bitwise_not(ff, ff_inv);
    Mat filled = im_flood | ff_inv;

    // quedarnos con el componente mayor
    findContours(filled.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return Mat::zeros(mask.size(), CV_8U);
    double maxA = 0; int imax = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > maxA) { maxA = a; imax = static_cast<int>(i); }
    }
    Mat keep = Mat::zeros(mask.size(), CV_8U);
    drawContours(keep, contours, imax, Scalar(255), FILLED);
    Mat out = (keep > 0);
    return out;
}

// crea patch normalizado centroid+shift, devuelve Jr (CV_64F), ok flag
static bool normalizeMaskedPatch_shift_cpp(const cv::Mat& I, int outSize, int shiftX, int shiftY, cv::Mat& outJ) {
    outJ = Mat::zeros(outSize, outSize, CV_64F);
    Mat mask = extractMaskLego_cpp(I);
    if (mask.empty() || !countNonZero(mask)) return false;

    // bounding box del componente mayor
    std::vector<std::vector<Point>> contours;
    Mat maskU8; mask.convertTo(maskU8, CV_8U, 255);
    findContours(maskU8.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;
    Rect bb = boundingRect(contours[0]);
    for (size_t k = 1; k < contours.size(); ++k) {
        double a = contourArea(contours[k]);
        if (a > contourArea(contours[0])) {
            bb = boundingRect(contours[k]);
        }
    }

    Mat G;
    if (I.channels() == 3) {
        Mat tmp; I.convertTo(tmp, CV_32F, 1.0 / 255.0); cvtColor(tmp, G, COLOR_BGR2GRAY);
    }
    else {
        I.convertTo(G, CV_32F, 1.0 / 255.0);
    }

    int x = static_cast<int>(std::floor(bb.x)) + shiftX;
    int y = static_cast<int>(std::floor(bb.y)) + shiftY;
    int w = static_cast<int>(bb.width);
    int h = static_cast<int>(bb.height);

    x = std::max(x, 0); y = std::max(y, 0);
    int x2 = std::min(x + w - 1, G.cols - 1);
    int y2 = std::min(y + h - 1, G.rows - 1);
    if (x2 <= x || y2 <= y) return false;

    Mat Gc = G(Range(y, y2 + 1), Range(x, x2 + 1)).clone();
    Mat Mc = mask(Range(y, y2 + 1), Range(x, x2 + 1)).clone();
    // aplicar máscara: poner a 0 donde máscara es 0
    Mat GcF; Gc.convertTo(GcF, CV_64F);
    for (int r = 0; r < GcF.rows; ++r) {
        for (int c = 0; c < GcF.cols; ++c) {
            if (Mc.at<uchar>(r, c) == 0) GcF.at<double>(r, c) = 0.0;
        }
    }

    int H = GcF.rows, W = GcF.cols;
    int S = std::max(H, W);
    int padY = (S - H) / 2;
    int padX = (S - W) / 2;
    Mat Gp = Mat::zeros(S, S, CV_64F);
    GcF.copyTo(Gp(Rect(padX, padY, W, H)));

    Mat Jr;
    resize(Gp, Jr, Size(outSize, outSize), 0, 0, INTER_LINEAR);
    // normalizar: restar media y dividir por norma
    Scalar mu = mean(Jr);
    Jr = Jr - mu[0];
    double nrm = norm(Jr);
    if (nrm < 1e-9) return false;
    Jr = Jr / nrm;

    Jr.copyTo(outJ);
    return true;
}

// predictYawPitch_byTemplate (matching por correlación dot entre patch normalizado y T)
static bool predictYawPitch_byTemplate_cpp(const cv::Mat& I, const std::vector<TemplateItem>& templates,
    int& outYaw, int& outPitch, std::vector<double>& outScores, double& outGap) {
    if (templates.empty()) { outYaw = outPitch = 0; outScores.clear(); outGap = std::numeric_limits<double>::quiet_NaN(); return false; }
    int outSize = templates[0].size;
    std::vector<int> shifts = { -4, 0, 4 };

    double bestGlobal = -1e300;
    int bestIdx = -1;
    std::vector<double> bestScores(templates.size(), -1e300);
    bool okAny = false;

    for (int dy : shifts) {
        for (int dx : shifts) {
            Mat J;
            if (!normalizeMaskedPatch_shift_cpp(I, outSize, dx, dy, J)) continue;
            okAny = true;
            // J is CV_64F outSize x outSize
            Mat Jvec = J.reshape(1, 1); // 1 x (outSize*outSize)
            for (size_t k = 0; k < templates.size(); ++k) {
                const Mat& T = templates[k].T; // expected CV_64F same size
                if (T.empty()) { bestScores[k] = std::max(bestScores[k], -1e300); continue; }
                Mat Tvec = T.reshape(1, 1);
                // dot product
                double s = Jvec.dot(Tvec);
                if (s > bestScores[k]) bestScores[k] = s;
            }
            // check global best
            for (size_t k = 0; k < bestScores.size(); ++k) {
                if (bestScores[k] > bestGlobal) {
                    bestGlobal = bestScores[k];
                    bestIdx = static_cast<int>(k);
                }
            }
        }
    }

    if (!okAny) {
        outYaw = outPitch = 0;
        outScores = std::vector<double>(templates.size(), std::numeric_limits<double>::quiet_NaN());
        outGap = std::numeric_limits<double>::quiet_NaN();
        return false;
    }

    // recompute scores (they already hold best found across shifts)
    outScores = bestScores;
    if (bestIdx < 0) {
        outYaw = outPitch = 0;
        outGap = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
    outYaw = templates[bestIdx].yaw;
    outPitch = templates[bestIdx].pitch;

    // gap between best and second best
    double bestScore = outScores[bestIdx];
    double second = -1e300;
    for (size_t k = 0; k < outScores.size(); ++k) if ((int)k != bestIdx) second = std::max(second, outScores[k]);
    outGap = bestScore - second;
    return true;
}

// carga plantillas desde folder para un código (cachea)
static const std::vector<TemplateItem>& loadTemplatesForCode(const string& templatesFolder, const string& codeKey) {
    auto it = g_templatesCache.find(codeKey);
    if (it != g_templatesCache.end()) return it->second;

    std::vector<TemplateItem> items;
    if (!fs::exists(templatesFolder) || !fs::is_directory(templatesFolder)) {
        g_templatesCache[codeKey] = items;
        return g_templatesCache[codeKey];
    }

    // buscar ficheros tpl_<code>_*.yml/.yaml
    std::string prefix = "tpl_" + codeKey + "_";
    for (auto& p : fs::directory_iterator(templatesFolder)) {
        if (!p.is_regular_file()) continue;
        string name = p.path().filename().string();
        string ln = name;
        std::transform(ln.begin(), ln.end(), ln.begin(), ::tolower);
        if (ln.find(prefix) != 0) continue;
        string ext = p.path().extension().string();
        if (ext != ".yml" && ext != ".yaml") continue;

        FileStorage fsr(p.path().string(), FileStorage::READ);
        if (!fsr.isOpened()) continue;
        TemplateItem ti;
        // read yaw/pitch/size/T (T -> opencv matrix)
        int yawI = 0, pitchI = 0, sizeI = 0;
        fsr["yaw"] >> yawI;
        fsr["pitch"] >> pitchI;
        fsr["size"] >> sizeI;
        Mat T;
        fsr["T"] >> T;
        fsr.release();
        if (T.empty() || sizeI <= 0) continue;
        // ensure T is CV_64F
        Mat Td;
        T.convertTo(Td, CV_64F);
        ti.yaw = yawI; ti.pitch = pitchI; ti.size = sizeI; ti.T = Td;
        items.push_back(ti);
    }

    // si no encontró .yml, también intentar .mat convertido -> pero asumimos YAML ya creado por convertidor
    g_templatesCache[codeKey] = items;
    return g_templatesCache[codeKey];
}

// wrapper: predictOrientationFromTemplates
static void predictOrientationFromTemplates(const cv::Mat& Ipiece, const string& codeStr, const string& templatesFolder,
    int& yaw, int& pitch, double& bestScore, double& gap) {
    yaw = 0; pitch = 0; bestScore = std::numeric_limits<double>::quiet_NaN(); gap = std::numeric_limits<double>::quiet_NaN();
    if (templatesFolder.empty()) return;
    string codeKey = codeStr;
    // trim leading/trailing spaces
    codeKey.erase(codeKey.find_last_not_of(" \n\r\t") + 1);
    codeKey.erase(0, codeKey.find_first_not_of(" \n\r\t"));

    const auto& templates = loadTemplatesForCode(templatesFolder, codeKey);
    if (templates.empty()) return;

    std::vector<double> scores;
    double localGap;
    int yawOut, pitchOut;
    bool ok = predictYawPitch_byTemplate_cpp(Ipiece, templates, yawOut, pitchOut, scores, localGap);
    if (!ok) { yaw = 0; pitch = 0; bestScore = std::numeric_limits<double>::quiet_NaN(); gap = std::numeric_limits<double>::quiet_NaN(); return; }
    yaw = yawOut; pitch = pitchOut;
    bestScore = *std::max_element(scores.begin(), scores.end());
    gap = localGap;
}

// usage
static void printUsage() {
    std::cout << "Usage: evaluate_classifiers <segFolder> <outTxt> <modelM.yml> [modelM_scaler.yml] [--templates <templatesFolder>] [--refine <model912.yml> [model912_scaler.yml]]\n";
}

int test_evaluacionClasificador(int argc, char** argv) {
    if (argc < 4) { printUsage(); return 1; }

    string segFolder = argv[1];
    string outTxt = argv[2];
    string modelM = argv[3];
    string modelMscaler;
    string model912, model912scaler;
    string templatesFolder;
    bool useRefiner = false;

    int idx = 4;
    // optional modelMscaler if next arg not starting with --
    if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
        modelMscaler = argv[idx++];
    }
    // parse remaining args
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
            std::cerr << "Unknown arg: " << a << "\n"; printUsage(); return 1;
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

    // predictor and shape names (kept for reference)
    vector<string> predictorNames = { "Extent","Solidity","V_mean","Eccentricity","SkelLenNorm","Circularity",
        "H_mean_circ","S_mean","V_IQR","S_median","FD5","EulerNumber" };
    vector<string> featNames_shape = {
        "AreaNorm","PerimNorm","Circularity","Extent","Solidity","Eccentricity","AspectRatio","EulerNumber",
        "HolesCount","HolesAreaFrac","SkelLenNorm","SkelEndpoints","SkelBranchpoints",
        "ProjV_peaks","ProjH_peaks","ProjV_entropy","ProjH_entropy",
        "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3",
        "StudsCount","StudsCountNormArea","StudsMeanRadius","StudsRadiusStd"
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
        else { std::ostringstream ss; ss << std::setfill('0') << std::setw(2) << predBase; predBaseStr = ss.str(); }
        string predFinalStr = predBaseStr;
        string refinador = "none";

        // 2) CASCADA AMARILLO 9-12 (refinador)
        if (useRefiner && predBase >= 0 && (predBase == 9 || predBase == 12)) {
            nRef912++;
            vector<double> featShape; vector<string> shapeNames;
            FeatureExtractor::ExtractShapeFeatures(Ipiece, featShape, shapeNames); // 14 features
            int predRef = predictWithSVM(svm912, mean912, std912, hasScaler912, featShape);
            string predRefStr;
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

        // 3) ORIENTACIÓN (según código final) - si templatesFolder proporcionado
        int yaw = 0, pitch = 0; double oScore = std::numeric_limits<double>::quiet_NaN(), oGap = std::numeric_limits<double>::quiet_NaN();
        if (!templatesFolder.empty()) {
            predictOrientationFromTemplates(Ipiece, predFinalStr, templatesFolder, yaw, pitch, oScore, oGap);
        }

        // 4) comparar
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

        // 5) log por imagen (CON ORIENTACIÓN si existe)
        fout << "[" << (i + 1) << "/" << N << "] " << imgName
            << " | REAL=" << trueLabelStr
            << " | PRED_BASE=" << predBaseStr
            << " | PRED_FINAL=" << predFinalStr
            << " | REF=" << refinador;
        if (!std::isnan(oScore)) {
            fout << " | ORI=" << std::setfill('0') << std::setw(3) << yaw << "/" << std::setfill('0') << std::setw(2) << pitch
                << " (s=" << std::fixed << std::setprecision(3) << oScore << " g=" << oGap << ")";
        }
        else {
            fout << " | ORI=---";
        }
        fout << " | " << resultStr << "\n";

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