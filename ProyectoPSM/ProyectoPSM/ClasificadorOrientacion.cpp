//------------------------------------------------------------
// Script para clasificación de orientación LEGO
//------------------------------------------------------------

#include "ClasificadorOrientacion.h"
#include <filesystem>
#include <algorithm>
#include <cmath>

using namespace cv;

ClasificadorOrientacion::ClasificadorOrientacion(std::string templatesFolder, int outSize)
    : templatesFolder_(std::move(templatesFolder)), outSize_(outSize) {
}

// ------------------------------------------------------------
// Carga de plantillas .yml/.yaml
// ------------------------------------------------------------
bool ClasificadorOrientacion::loadAllTemplates() {
    namespace fs = std::filesystem;
    templates_.clear();

    // Convertir desde UTF-8 a path nativo Windows
    fs::path folder = fs::u8path(templatesFolder_);

    std::error_code ec;
    if (!fs::exists(folder, ec) || ec) {
        return false;
    }

    for (const auto& entry : fs::directory_iterator(folder, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;

        std::string extLower = entry.path().extension().string();
        std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);
        if (extLower != ".yml" && extLower != ".yaml") continue;

        TemplateItem item;
        const std::string path = entry.path().string(); 
        if (readOneTemplateYml(path, item, outSize_)) {
            templates_.push_back(std::move(item));
        }
    }

    std::sort(templates_.begin(), templates_.end(),
        [](const TemplateItem& a, const TemplateItem& b) {
            if (a.code != b.code) return a.code < b.code;
            if (a.pitch != b.pitch) return a.pitch < b.pitch;
            return a.yaw < b.yaw;
        });

    return !templates_.empty();
}

// ------------------------------------------------------------
// Lectura de una plantilla .yml/.yaml
// ------------------------------------------------------------
bool ClasificadorOrientacion::readOneTemplateYml(const std::string& path, TemplateItem& outItem, int outSize) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    // Campos
    std::string code;
    fs["code"] >> code;
    outItem.code = code;

    outItem.yaw = (int)fs["yaw"];
    outItem.pitch = (int)fs["pitch"];
    outItem.size = (int)fs["size"];

    cv::Mat T64;
    fs["T"] >> T64; // normalmente CV_64F si dt: d
    fs.release();

    if (T64.empty()) return false;

    // Asegurar 1 canal
    if (T64.channels() > 1) cv::cvtColor(T64, T64, cv::COLOR_BGR2GRAY);

    // Redimensionar si hace falta
    if (T64.rows != outSize || T64.cols != outSize) {
        cv::resize(T64, T64, cv::Size(outSize, outSize), 0, 0, cv::INTER_LINEAR);
    }

    // Convertir a float
    cv::Mat T32;
    T64.convertTo(T32, CV_32F);

    // Normalización 
    zeroMeanL2Norm(T32);

    outItem.T = std::move(T32);
    outItem.file = path;
    return true;
}

// ------------------------------------------------------------
// Extracción de máscara LEGO
// ------------------------------------------------------------
bool ClasificadorOrientacion::extractMaskLego(const cv::Mat& I, cv::Mat& maskOut) const {
    if (I.empty()) return false;

    cv::Mat gray;
    if (I.channels() == 3) cv::cvtColor(I, gray, cv::COLOR_BGR2GRAY);
    else gray = I;

    cv::Mat gf;
    if (gray.type() == CV_8U) gray.convertTo(gf, CV_32F, 1.0 / 255.0);
    else gray.convertTo(gf, CV_32F);

    // threshold ~ 0.03
    cv::Mat mask = gf > 0.03f;
    mask.convertTo(mask, CV_8U, 255);

    // close (disk 2 aprox) + open (disk 1 aprox)
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // fill holes (floodfill desde borde)
    cv::Mat holes = mask.clone();
    cv::floodFill(holes, cv::Point(0, 0), cv::Scalar(255));
    cv::bitwise_not(holes, holes);
    mask = mask | holes;

    // componente mayor
    if (!largestComponent(mask)) {
        maskOut = cv::Mat::zeros(mask.size(), CV_8U);
        return false;
    }

    maskOut = mask;
    return cv::countNonZero(maskOut) > 0;
}


// ------------------------------------------------------------
// Componente principal
// ------------------------------------------------------------
bool ClasificadorOrientacion::largestComponent(cv::Mat& binMask) {
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(binMask, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return false;

    int best = -1, bestArea = 0;
    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) { bestArea = area; best = i; }
    }
    if (best < 0) return false;

    binMask = (labels == best);
    binMask.convertTo(binMask, CV_8U, 255);
    return true;
}

// ------------------------------------------------------------
// Normalización de parche con máscara y shift
// ------------------------------------------------------------
bool ClasificadorOrientacion::normalizeMaskedPatchShift(const cv::Mat& I, int shiftX, int shiftY, cv::Mat& J) const {
    J = cv::Mat::zeros(outSize_, outSize_, CV_32F);

    cv::Mat mask;
    if (!extractMaskLego(I, mask)) return false;

    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);
    if (pts.empty()) return false;

    cv::Rect bb = cv::boundingRect(pts);

    bb.x += shiftX;
    bb.y += shiftY;

    bb.x = std::max(0, bb.x);
    bb.y = std::max(0, bb.y);
    bb.width = std::min(bb.width, I.cols - bb.x);
    bb.height = std::min(bb.height, I.rows - bb.y);
    if (bb.width <= 1 || bb.height <= 1) return false;

    cv::Mat gray;
    if (I.channels() == 3) cv::cvtColor(I, gray, cv::COLOR_BGR2GRAY);
    else gray = I;

    cv::Mat gf;
    if (gray.type() == CV_8U) gray.convertTo(gf, CV_32F, 1.0 / 255.0);
    else gray.convertTo(gf, CV_32F);

    cv::Mat Gc = gf(bb).clone();
    cv::Mat Mc = mask(bb).clone();
    Mc = (Mc > 0);

    Gc.setTo(0.0f, ~Mc);

    int H = Gc.rows, W = Gc.cols;
    int S = std::max(H, W);

    int top = (S - H) / 2;
    int bottom = S - H - top;
    int left = (S - W) / 2;
    int right = S - W - left;

    cv::Mat Gp;
    cv::copyMakeBorder(Gc, Gp, top, bottom, left, right, cv::BORDER_CONSTANT, 0.0f);

    cv::Mat Jr;
    cv::resize(Gp, Jr, cv::Size(outSize_, outSize_), 0, 0, cv::INTER_LINEAR);

    zeroMeanL2Norm(Jr);
    J = Jr;
    return true;
}

// ------------------------------------------------------------
// Normalización media
// ------------------------------------------------------------
void ClasificadorOrientacion::zeroMeanL2Norm(cv::Mat& M) {
    CV_Assert(M.type() == CV_32F);
    cv::Scalar mu = cv::mean(M);
    M -= (float)mu[0];

    double nrm = cv::norm(M, cv::NORM_L2);
    if (nrm < 1e-9) return;
    M /= (float)nrm;
}

// ------------------------------------------------------------
// Predicción de orientación
// ------------------------------------------------------------
OrientationResult ClasificadorOrientacion::predict(const cv::Mat& Ipiece, const std::string& filterCode) const {
    OrientationResult res;
    if (templates_.empty() || Ipiece.empty()) return res;

    // Filtrar candidates por code si aplica
    std::vector<const TemplateItem*> cand;
    cand.reserve(templates_.size());
    for (const auto& t : templates_) {
        if (!filterCode.empty() && t.code != filterCode) continue;
        cand.push_back(&t);
    }
    if (cand.empty()) return res;

    const int shifts[] = { -4, 0, 4 };

    bool okAny = false;
    float bestGlobalScore = -1e9f;
    int bestIdx = -1;
    std::vector<float> bestScores; // para gap

    const int K = (int)cand.size();

    for (int dy : shifts) {
        for (int dx : shifts) {
            cv::Mat J;
            if (!normalizeMaskedPatchShift(Ipiece, dx, dy, J)) continue;
            okAny = true;

            std::vector<float> scores(K, -1e9f);
            for (int k = 0; k < K; ++k) {
                // ambos normalizados => dot = similitud 
                scores[k] = (float)J.dot(cand[k]->T);
            }

            auto itMax = std::max_element(scores.begin(), scores.end());
            float smax = *itMax;
            int idx = (int)std::distance(scores.begin(), itMax);

            if (smax > bestGlobalScore) {
                bestGlobalScore = smax;
                bestIdx = idx;
                bestScores = std::move(scores);
            }
        }
    }

    if (!okAny || bestIdx < 0) return res;

    float second = -1e9f;
    for (int k = 0; k < K; ++k) {
        if (k == bestIdx) continue;
        second = std::max(second, bestScores[k]);
    }

    const TemplateItem* bestT = cand[bestIdx];

    res.yaw = bestT->yaw;
    res.pitch = bestT->pitch;
    res.bestScore = bestGlobalScore;
    res.gap = bestGlobalScore - second;
    res.ok = true;
    res.matchedCode = bestT->code;
    res.matchedFile = bestT->file;
    return res;
}
