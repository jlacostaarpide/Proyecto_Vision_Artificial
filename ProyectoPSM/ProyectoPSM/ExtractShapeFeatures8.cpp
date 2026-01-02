#include "ExtractShapeFeatures8.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;

namespace FeatureExtractor8 {

    // ---------------- helpers (copiados/adaptados de tu ExtractShapeFeatures6) ----------------

    static Mat toGrayFloat01(const Mat& in) {
        if (in.empty()) return Mat();
        Mat f;
        if (in.depth() == CV_8U) in.convertTo(f, CV_32F, 1.0 / 255.0);
        else {
            in.convertTo(f, CV_32F);
            double mn, mx; minMaxLoc(f, &mn, &mx);
            if (mx > 1.0) f *= (1.0f / 255.0f);
        }
        Mat g;
        if (f.channels() == 3) cvtColor(f, g, COLOR_BGR2GRAY);
        else if (f.channels() == 4) cvtColor(f, g, COLOR_BGRA2GRAY);
        else g = f;
        return g; // CV_32F, 0..1
    }

    static Mat bwareaopen_u8(const Mat& binU8, int minArea) {
        std::vector<std::vector<Point>> cnts;
        findContours(binU8.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat out = Mat::zeros(binU8.size(), CV_8U);
        for (auto& c : cnts) {
            if (contourArea(c) >= minArea) {
                drawContours(out, std::vector<std::vector<Point>>{c}, 0, 255, FILLED);
            }
        }
        return out;
    }

    static void fillSmallHoles_keepBig(Mat& maskU8, int holeSmallMaxArea) {
        Mat flood = maskU8.clone();
        floodFill(flood, Point(0, 0), Scalar(255));
        Mat floodInv; bitwise_not(flood, floodInv);

        Mat filled = maskU8 | floodInv;
        Mat holes = filled & (~maskU8);

        std::vector<std::vector<Point>> hc;
        findContours(holes.clone(), hc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat holesSmall = Mat::zeros(maskU8.size(), CV_8U);
        for (auto& c : hc) {
            if (contourArea(c) < holeSmallMaxArea) {
                drawContours(holesSmall, std::vector<std::vector<Point>>{c}, 0, 255, FILLED);
            }
        }
        bitwise_or(maskU8, holesSmall, maskU8);
    }

    static bool keepLargestComponent(Mat& maskU8) {
        std::vector<std::vector<Point>> cnts;
        findContours(maskU8.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (cnts.empty()) return false;

        int imax = 0; double amax = 0.0;
        for (int i = 0; i < (int)cnts.size(); ++i) {
            double a = std::abs(contourArea(cnts[i]));
            if (a > amax) { amax = a; imax = i; }
        }
        Mat keep = Mat::zeros(maskU8.size(), CV_8U);
        drawContours(keep, cnts, imax, 255, FILLED);
        maskU8 = keep;
        return true;
    }

    static std::vector<double> movmean(const std::vector<double>& x, int W) {
        int n = (int)x.size();
        if (n == 0) return {};
        if (W < 1) W = 1;
        int hw = std::max(1, W / 2);
        std::vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            int a = std::max(0, i - hw);
            int b = std::min(n - 1, i + hw);
            double s = 0.0; int cnt = 0;
            for (int k = a; k <= b; ++k) { s += x[k]; cnt++; }
            y[i] = s / std::max(1, cnt);
        }
        return y;
    }

    static double entropyFromNonNeg(const std::vector<double>& x) {
        double sum = std::accumulate(x.begin(), x.end(), 0.0);
        sum = std::max(sum, 1e-12);
        double e = 0.0;
        for (double v : x) {
            double p = v / sum;
            e += -p * std::log(p + 1e-12);
        }
        return e;
    }

    static double giniCoeff(std::vector<double> x) {
        for (double& v : x) if (v < 0) v = 0;
        double s = std::accumulate(x.begin(), x.end(), 0.0);
        if (s < 1e-12) return 0.0;
        std::sort(x.begin(), x.end());
        int n = (int)x.size();
        double num = 0.0;
        for (int i = 0; i < n; ++i) num += (i + 1) * x[i];
        double g = (2.0 * num) / (n * s) - (n + 1.0) / n;
        return std::abs(g);
    }

    static Mat gridOccupancy(const Mat& maskU8, int N) {
        Mat out = Mat::zeros(N, N, CV_64F);
        int h = maskU8.rows, w = maskU8.cols;

        std::vector<int> ys(N + 1), xs(N + 1);
        for (int i = 0; i <= N; ++i) {
            ys[i] = (int)std::round(i * (h / (double)N));
            xs[i] = (int)std::round(i * (w / (double)N));
        }
        ys[N] = h; xs[N] = w;

        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int y1 = std::max(0, ys[r]);
                int y2 = std::min(h, ys[r + 1]);
                int x1 = std::max(0, xs[c]);
                int x2 = std::min(w, xs[c + 1]);
                if (y2 <= y1 || x2 <= x1) { out.at<double>(r, c) = 0; continue; }

                Rect rc(x1, y1, x2 - x1, y2 - y1);
                Mat cell = maskU8(rc);
                double frac = (double)countNonZero(cell) / std::max(1, rc.area());
                out.at<double>(r, c) = frac;
            }
        }
        return out;
    }

    // ---------------- API ----------------

    std::vector<std::string> DefaultNames8() {
        return {
            "AreaNorm","PerimNorm","ProjH_entropy",
            "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3",
            "LightFrac","LightContrast"
        };
    }

    static void computeLightBicolorFeatures_LabK2(const Mat& InColorN, const Mat& maskN,
        const Shape8Opts& opts,
        double& LightFrac, double& LightContrast,
        Mat* dbgLabOut = nullptr)
    {
        LightFrac = 0.0;
        LightContrast = 0.0;
        if (InColorN.empty() || maskN.empty() || countNonZero(maskN) == 0) return;

        Mat lab;
        cvtColor(InColorN, lab, COLOR_BGR2Lab); // L: 0..255
        if (dbgLabOut) *dbgLabOut = lab.clone();

        // Extraer L dentro de máscara
        std::vector<float> Lvals;
        Lvals.reserve((size_t)countNonZero(maskN));

        for (int y = 0; y < lab.rows; ++y) {
            const Vec3b* p = lab.ptr<Vec3b>(y);
            const uchar* m = maskN.ptr<uchar>(y);
            for (int x = 0; x < lab.cols; ++x) {
                if (!m[x]) continue;
                Lvals.push_back((float)p[x][0]); // L
            }
        }
        if (Lvals.size() < 20) return; // muy pocos píxeles: no fiarse

        Mat data((int)Lvals.size(), 1, CV_32F, Lvals.data());
        data = data.clone(); // kmeans necesita buffer propio

        Mat labels, centers;
        TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, opts.kmeansMaxIter, opts.kmeansEps);

        // K=2
        kmeans(data, 2, labels, tc, opts.kmeansAttempts, KMEANS_PP_CENTERS, centers);

        float c0 = centers.at<float>(0, 0);
        float c1 = centers.at<float>(1, 0);
        int brightLabel = (c0 >= c1) ? 0 : 1;
        float Lbright = std::max(c0, c1);
        float Ldark = std::min(c0, c1);

        int brightCount = 0;
        for (int i = 0; i < labels.rows; ++i) {
            if (labels.at<int>(i, 0) == brightLabel) brightCount++;
        }

        LightFrac = (double)brightCount / std::max(1.0, (double)labels.rows);
        LightContrast = (double)(Lbright - Ldark) / 255.0; // normalizado 0..1 aprox
    }

    void ExtractShapeFeatures8(const Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape8Opts& opts,
        Shape8Dbg* dbg)
    {
        featNames = DefaultNames8();
        feat.assign(8, 0.0);

        if (dbg) { dbg->maskN.release(); dbg->InColorN.release(); dbg->InLabN.release(); }

        try {
            if (I_in.empty()) return;

            // 0) preparar máscara base desde gris
            Mat Ig = toGrayFloat01(I_in);
            if (Ig.empty()) return;

            Mat mask = (Ig > (float)opts.tBlackMin);
            mask.convertTo(mask, CV_8U, 255);

            mask = bwareaopen_u8(mask, opts.minObjArea);

            morphologyEx(mask, mask, MORPH_CLOSE,
                getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.closeRadius + 1, 2 * opts.closeRadius + 1)));
            morphologyEx(mask, mask, MORPH_OPEN,
                getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.openRadius + 1, 2 * opts.openRadius + 1)));

            fillSmallHoles_keepBig(mask, opts.holeSmallMaxArea);

            if (!keepLargestComponent(mask)) return;

            // orientación por momentos
            double angleDeg = 0.0;
            {
                Moments mu = moments(mask, true);
                if (mu.m00 > 1e-9) {
                    double mu20 = mu.mu20 / mu.m00;
                    double mu02 = mu.mu02 / mu.m00;
                    double mu11 = mu.mu11 / mu.m00;
                    double theta = 0.5 * std::atan2(2.0 * mu11, (mu20 - mu02));
                    angleDeg = theta * 180.0 / CV_PI;
                }
            }

            // rotación "loose" (igual que en tu de 6), pero aplicada también a COLOR
            Point2f ctr(mask.cols / 2.f, mask.rows / 2.f);
            Mat R = getRotationMatrix2D(ctr, -angleDeg, 1.0);
            Rect bbox = RotatedRect(ctr, mask.size(), (float)-angleDeg).boundingRect();
            R.at<double>(0, 2) += bbox.width / 2.0 - ctr.x;
            R.at<double>(1, 2) += bbox.height / 2.0 - ctr.y;

            Mat maskR;
            warpAffine(mask, maskR, R, bbox.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

            Mat Icolor;
            if (I_in.channels() == 4) cvtColor(I_in, Icolor, COLOR_BGRA2BGR);
            else Icolor = I_in;

            Mat IcolorR;
            warpAffine(Icolor, IcolorR, R, bbox.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

            // crop tight
            if (opts.cropTight) {
                std::vector<std::vector<Point>> cnts;
                findContours(maskR.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                if (cnts.empty()) return;

                int imax = 0; double amax = 0.0;
                for (int i = 0; i < (int)cnts.size(); ++i) {
                    double a = std::abs(contourArea(cnts[i]));
                    if (a > amax) { amax = a; imax = i; }
                }
                Rect bb = boundingRect(cnts[imax]);

                maskR = maskR(bb).clone();
                IcolorR = IcolorR(bb).clone();
            }

            // normalizar escala
            int h = maskR.rows, w = maskR.cols;
            double scale = (double)opts.normTargetSize / std::max(1, std::max(h, w));
            if (!std::isfinite(scale) || scale <= 0) scale = 1.0;

            Mat maskN, InColorN;
            resize(maskR, maskN, Size(), scale, scale, INTER_NEAREST);
            resize(IcolorR, InColorN, Size(), scale, scale, INTER_LINEAR);

            double A = (double)countNonZero(maskN);
            if (A < 1.0) return;

            // 1) AreaNorm, PerimNorm
            double P = 1e-9;
            {
                std::vector<std::vector<Point>> cnts;
                findContours(maskN.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
                if (!cnts.empty()) {
                    int imax = 0; double amax = 0.0;
                    for (int i = 0; i < (int)cnts.size(); ++i) {
                        double a = std::abs(contourArea(cnts[i]));
                        if (a > amax) { amax = a; imax = i; }
                    }
                    P = std::max(arcLength(cnts[imax], true), 1e-9);
                }
            }

            double AreaNorm = A / (double)maskN.total();
            double PerimNorm = P / std::max(1e-9, 2.0 * (maskN.rows + maskN.cols));

            // 2) ProjH_entropy
            std::vector<double> projH(maskN.rows, 0.0);
            for (int y = 0; y < maskN.rows; ++y) {
                const uchar* p = maskN.ptr<uchar>(y);
                int rowSum = 0;
                for (int x = 0; x < maskN.cols; ++x) if (p[x]) rowSum++;
                projH[y] = (double)rowSum;
            }
            std::vector<double> projH_s = movmean(projH, opts.projSmooth);
            double ProjH_entropy = entropyFromNonNeg(projH_s);

            // 3) grid occupancy 3x3
            Mat G = gridOccupancy(maskN, opts.gridN);
            std::vector<double> g; g.reserve(G.rows * G.cols);
            for (int r = 0; r < G.rows; ++r)
                for (int c = 0; c < G.cols; ++c)
                    g.push_back(G.at<double>(r, c));

            double GridOccFrac_3x3 = 0.0;
            for (double v : g) if (v > 0.15) GridOccFrac_3x3 += 1.0;
            GridOccFrac_3x3 /= std::max(1.0, (double)g.size());

            double GridOccGini_3x3 = giniCoeff(g);

            double diag1 = 0.0, diag2 = 0.0;
            for (int i = 0; i < opts.gridN; ++i) {
                diag1 += G.at<double>(i, i);
                diag2 += G.at<double>(i, opts.gridN - 1 - i);
            }
            double GridOccDiagDiff_3x3 = std::abs(diag1 - diag2);

            // 4) COLOR: bicoloridad en L (Lab) dentro de máscara
            double LightFrac = 0.0, LightContrast = 0.0;
            Mat dbgLab;
            computeLightBicolorFeatures_LabK2(InColorN, maskN, opts, LightFrac, LightContrast, dbg ? &dbgLab : nullptr);

            // ensamblar 8
            feat[0] = AreaNorm;
            feat[1] = PerimNorm;
            feat[2] = ProjH_entropy;
            feat[3] = GridOccFrac_3x3;
            feat[4] = GridOccGini_3x3;
            feat[5] = GridOccDiagDiff_3x3;
            feat[6] = LightFrac;
            feat[7] = LightContrast;

            if (dbg) {
                dbg->maskN = maskN.clone();
                dbg->InColorN = InColorN.clone();
                dbg->InLabN = dbgLab.clone();
            }
        }
        catch (...) {
            feat.assign(8, 0.0);
        }
    }

} // namespace FeatureExtractor8
