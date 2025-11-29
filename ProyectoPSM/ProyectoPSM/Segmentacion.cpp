#include "segmentacion.h"
#include <QDebug>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>

Segmentacion::Segmentacion(QObject* parent)
    : QObject(parent)
{
    // Registrar para poder usar cv::Mat en conexiones queued entre hilos
    qRegisterMetaType<cv::Mat>("cv::Mat");
}

Segmentacion::~Segmentacion() = default;

void Segmentacion::processImage(const cv::Mat &input)
{
    if (input.empty()) return;

    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    cv::Mat mask = createMask(gray);

    cv::Mat result;
    if (input.channels() == 3) {
        // copiar color original donde la máscara sea no cero
        input.copyTo(result, mask);
    } else {
        result = mask;
    }

    emit segmentedImage(result);
}

cv::Mat Segmentacion::createMask(const cv::Mat &gray)
{
    cv::Mat blurred, thresh;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);
    cv::adaptiveThreshold(blurred, thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);

    return thresh;
}

cv::Mat Segmentacion::Segment(const cv::Mat& src)
{
    if (src.empty()) return cv::Mat();

    // Copia de trabajo
    cv::Mat img = src.clone();

    // Convertir a HSV y normalizar canales a [0,1] como en MATLAB
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    cv::Mat H_u8 = ch[0], S_u8 = ch[1], V_u8 = ch[2];

    cv::Mat H, S, V;
    H_u8.convertTo(H, CV_32F, 1.0f / 179.0f);
    S_u8.convertTo(S, CV_32F, 1.0f / 255.0f);
    V_u8.convertTo(V, CV_32F, 1.0f / 255.0f);

    // Stretchlim equivalente: percentiles 1% y 95% sobre V
    std::vector<float> Vvals;
    Vvals.reserve(V.total());
    for (int r = 0; r < V.rows; ++r) {
        const float* p = V.ptr<float>(r);
        for (int c = 0; c < V.cols; ++c) Vvals.push_back(p[c]);
    }
    if (Vvals.empty()) return src;
    std::sort(Vvals.begin(), Vvals.end());
    auto pct = [&](double p) {
        size_t idx = std::min<size_t>(Vvals.size() - 1, static_cast<size_t>(std::round(p * (Vvals.size() - 1))));
        return Vvals[idx];
    };
    float low = pct(0.01), high = pct(0.95);
    if (high - low < 1e-6f) high = low + 1e-6f;
    cv::Mat V_eq = (V - low) / (high - low);
    cv::threshold(V_eq, V_eq, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(V_eq, V_eq, 1.0, 1.0, cv::THRESH_TRUNC);
    V = V_eq;

    // Multi-level Otsu sobre S (implementación aproximada por histogram + búsqueda)
    cv::Mat S_proc;
    cv::pow(S, 1.0, S_proc);

    const int NBINS = 256;
    int histSize = NBINS;
    float rangeA[] = { 0.0f, 1.0f };
    const float* ranges[] = { rangeA };
    cv::Mat hist;
    cv::calcHist(&S_proc, 1, std::vector<int>{0}.data(), cv::Mat(), hist, 1, &histSize, ranges, true, false);
    hist /= (float)S_proc.total();

    std::vector<float> P(NBINS), Pcum(NBINS), meanCum(NBINS);
    for (int i = 0; i < NBINS; ++i) P[i] = hist.at<float>(i);
    Pcum[0] = P[0];
    meanCum[0] = P[0] * 0.0f;
    for (int i = 1; i < NBINS; ++i) {
        Pcum[i] = Pcum[i - 1] + P[i];
        meanCum[i] = meanCum[i - 1] + P[i] * (i / float(NBINS - 1));
    }

    double bestScore = -1.0;
    int best_t1 = 0, best_t2 = NBINS - 1;
    for (int t1 = 0; t1 < NBINS - 1; ++t1) {
        for (int t2 = t1 + 1; t2 < NBINS; ++t2) {
            float w0 = Pcum[t1];
            float w1 = Pcum[t2] - Pcum[t1];
            float w2 = 1.0f - Pcum[t2];
            if (w0 <= 1e-6 || w1 <= 1e-6 || w2 <= 1e-6) continue;
            float m0 = meanCum[t1] / w0;
            float m1 = (meanCum[t2] - meanCum[t1]) / w1;
            float m2 = (meanCum[NBINS - 1] - meanCum[t2]) / w2;
            double score = w0 * (m0 - meanCum[NBINS - 1]) * (m0 - meanCum[NBINS - 1])
                + w1 * (m1 - meanCum[NBINS - 1]) * (m1 - meanCum[NBINS - 1])
                + w2 * (m2 - meanCum[NBINS - 1]) * (m2 - meanCum[NBINS - 1]);
            if (score > bestScore) { bestScore = score; best_t1 = t1; best_t2 = t2; }
        }
    }

    float th1 = best_t1 / float(NBINS - 1);
    float th2 = best_t2 / float(NBINS - 1);

    // Clasificar en 3 clases y calcular ratio de clase 2
    cv::Mat L_quant = cv::Mat::zeros(S_proc.size(), CV_8U);
    for (int r = 0; r < S_proc.rows; ++r) {
        const float* pS = S_proc.ptr<float>(r);
        uint8_t* pL = L_quant.ptr<uint8_t>(r);
        for (int c = 0; c < S_proc.cols; ++c) {
            float v = pS[c];
            if (v <= th1) pL[c] = 1;
            else if (v <= th2) pL[c] = 2;
            else pL[c] = 3;
        }
    }
    int count_mid = cv::countNonZero(L_quant == 2);
    double ratio_mid = double(count_mid) / double(S_proc.total());
    double umbral_area_max_mid = 0.15;
    float level_otsu_S = (ratio_mid > umbral_area_max_mid) ? th2 : th1;

    cv::Mat mask_S;
    cv::threshold(S_proc, mask_S, level_otsu_S, 1.0, cv::THRESH_BINARY);
    mask_S.convertTo(mask_S, CV_8U, 255.0);

    // H purple rescue: rango [0.68,0.88] y S > 0.4*level_otsu_S
    float min_sat_H = 0.4f * level_otsu_S;
    cv::Mat mask_H = cv::Mat::zeros(H.size(), CV_8U);
    for (int r = 0; r < H.rows; ++r) {
        const float* pH = H.ptr<float>(r);
        const float* pS = S.ptr<float>(r);
        uint8_t* pm = mask_H.ptr<uint8_t>(r);
        for (int c = 0; c < H.cols; ++c) {
            float hv = pH[c], sv = pS[c];
            if (hv >= 0.68f && hv <= 0.88f && sv > min_sat_H) pm[c] = 255;
        }
    }

    // mask V dark: el .m la desactiva -> mantenemos cero
    cv::Mat mask_V = cv::Mat::zeros(V.size(), CV_8U);

    // Unión y limpieza
    cv::Mat mask_comb;
    cv::bitwise_or(mask_S, mask_H, mask_comb);
    cv::bitwise_or(mask_comb, mask_V, mask_comb);

    // imfill (relleno de huecos) usando floodFill sobre el inverso
    cv::Mat mask_filled;
    {
        cv::Mat inv;
        cv::bitwise_not(mask_comb, inv);
        cv::Mat ff = inv.clone();
        cv::floodFill(ff, cv::Point(0, 0), cv::Scalar(255));
        cv::bitwise_not(ff, ff);
        mask_filled = mask_comb | ff;
    }

    // apertura con disco r=3
    int r_noise = 3;
    cv::Mat se_noise = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r_noise + 1, 2 * r_noise + 1));
    cv::Mat mask_clean;
    cv::morphologyEx(mask_filled, mask_clean, cv::MORPH_OPEN, se_noise);

    // imclearborder: eliminar componentes que tocan borde
    cv::Mat labels;
    cv::Mat mask_noborder = mask_clean.clone();
    int nlabels = cv::connectedComponents(mask_noborder, labels, 8, CV_32S);
    if (nlabels > 1) {
        std::vector<char> remove(nlabels, 0);
        // top/bottom
        for (int c = 0; c < labels.cols; ++c) {
            int t = labels.at<int>(0, c);
            int b = labels.at<int>(labels.rows - 1, c);
            if (t > 0) remove[t] = 1;
            if (b > 0) remove[b] = 1;
        }
        // left/right
        for (int r = 0; r < labels.rows; ++r) {
            int l = labels.at<int>(r, 0);
            int rr = labels.at<int>(r, labels.cols - 1);
            if (l > 0) remove[l] = 1;
            if (rr > 0) remove[rr] = 1;
        }
        cv::Mat tmp = cv::Mat::zeros(labels.size(), CV_8U);
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                int L = labels.at<int>(y, x);
                if (L > 0 && !remove[L]) tmp.at<uint8_t>(y, x) = 255;
            }
        }
        mask_noborder = tmp;
    }

    cv::Mat mask_final = mask_noborder.clone();

    // cierre con radio 12 y nuevo relleno, luego apertura r=5
    int radio_pegamento = 12;
    cv::Mat se_merge = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radio_pegamento + 1, 2 * radio_pegamento + 1));
    cv::Mat mask_merged;
    cv::morphologyEx(mask_final, mask_merged, cv::MORPH_CLOSE, se_merge);

    {
        cv::Mat inv;
        cv::bitwise_not(mask_merged, inv);
        cv::Mat ff = inv.clone();
        cv::floodFill(ff, cv::Point(0, 0), cv::Scalar(255));
        cv::bitwise_not(ff, ff);
        mask_merged = mask_merged | ff;
    }

    int r_smooth = 5;
    cv::Mat se_smooth = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * r_smooth + 1, 2 * r_smooth + 1));
    cv::Mat mask_final_consolidated;
    cv::morphologyEx(mask_merged, mask_final_consolidated, cv::MORPH_OPEN, se_smooth);

    mask_final = mask_final_consolidated;

    // connected components y extracción de stats
    cv::Mat labels2;
    int nl2 = cv::connectedComponents(mask_final, labels2, 8, CV_32S);
    struct R { int label; int area; cv::Rect bbox; cv::Point2d c; double per; double circ; cv::Mat mask; };
    std::vector<R> regs;
    if (nl2 > 1) {
        std::vector<int> areas(nl2, 0);
        std::vector<cv::Rect> bbs(nl2);
        std::vector<cv::Point2d> cents(nl2, cv::Point2d(0, 0));
        for (int y = 0; y < labels2.rows; ++y) {
            for (int x = 0; x < labels2.cols; ++x) {
                int L = labels2.at<int>(y, x);
                if (L <= 0) continue;
                areas[L]++;
                cents[L].x += x; cents[L].y += y;
                if (areas[L] == 1) bbs[L] = cv::Rect(x, y, 1, 1);
                else bbs[L] |= cv::Rect(x, y, 1, 1);
            }
        }
        for (int L = 1; L < nl2; ++L) {
            if (areas[L] == 0) continue;
            cents[L].x /= areas[L]; cents[L].y /= areas[L];
            R r; r.label = L; r.area = areas[L]; r.bbox = bbs[L]; r.c = cents[L];
            cv::Mat local = (labels2 == L);
            local.convertTo(local, CV_8U, 255);
            r.mask = local;
            std::vector<std::vector<cv::Point>> conts;
            cv::findContours(local, conts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            double per = 0;
            for (auto &ct : conts) per += cv::arcLength(ct, true);
            r.per = per;
            r.circ = (per > 1e-6) ? (4.0 * CV_PI * r.area / (per * per)) : 0.0;
            regs.push_back(r);
        }
    }

    // filtrar por area relativa 5% del max
    std::vector<R> regs_final;
    if (!regs.empty()) {
        int max_area = 0;
        for (auto &rg : regs) if (rg.area > max_area) max_area = rg.area;
        int umbral_area = static_cast<int>(0.05 * std::max(1, max_area));
        for (auto &rg : regs) if (rg.area > umbral_area) regs_final.push_back(rg);
    }

    // Dibujar sobre copia de original
    cv::Mat out = src.clone();
    int idx = 1;
    for (auto &rg : regs_final) {
        cv::rectangle(out, rg.bbox, cv::Scalar(0, 255, 0), 2);
        cv::circle(out, rg.c, 3, cv::Scalar(0, 0, 255), -1);

        std::ostringstream ss;
        ss << "#" << idx << " C:" << std::fixed << std::setprecision(2) << rg.circ << " A:" << rg.area;
        std::string label = ss.str();

        int baseLine = 0;
        cv::Size tsize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseLine);
        cv::Point torig(std::max(0, rg.bbox.x), std::max(0, rg.bbox.y - 6));
        cv::rectangle(out, torig + cv::Point(0, baseLine), torig + cv::Point(tsize.width, -tsize.height), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(out, label, torig, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 1);

        ++idx;
    }

    return out;
}