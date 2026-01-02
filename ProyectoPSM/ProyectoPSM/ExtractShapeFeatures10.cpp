#include "ExtractShapeFeatures10.h"
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;

namespace FeatureExtractor10 {

    // ---------------- helpers ----------------

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

        Mat filled = maskU8 | floodInv;     // imfill(mask,'holes')
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

    static int countPeaks01(const std::vector<double>& xin, double thr01 = 0.35) {
        if (xin.size() < 5) return 0;
        std::vector<double> x = xin;
        double mn = *std::min_element(x.begin(), x.end());
        for (double& v : x) v -= mn;
        double mx = *std::max_element(x.begin(), x.end());
        if (mx < 1e-12) return 0;
        for (double& v : x) v /= mx;

        int n = 0;
        for (int i = 1; i + 1 < (int)x.size(); ++i) {
            bool isPeak = (x[i] > x[i - 1] && x[i] > x[i + 1]);
            if (isPeak && x[i] > thr01) ++n;
        }
        return n;
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

    static double studsCountRobust(const Mat& IgN01, const Mat& maskU8,
        const Shape10Opts& opts, Mat* dbgBinOut = nullptr)
    {
        if (!opts.studsEnable) return 0.0;
        if (IgN01.empty() || maskU8.empty() || countNonZero(maskU8) == 0) return 0.0;

        Mat I = IgN01.clone();
        // enmascarar fondo para evitar falsos
        Mat I_m = Mat::zeros(I.size(), CV_32F);
        I.copyTo(I_m, maskU8);

        // Top-hat (resalta bultos claros)
        int k = std::max(1, opts.studsTophatRadius);
        Mat se = getStructuringElement(MORPH_ELLIPSE, Size(2 * k + 1, 2 * k + 1));

        Mat I8; I_m.convertTo(I8, CV_8U, 255.0);
        Mat tophat8;
        morphologyEx(I8, tophat8, MORPH_TOPHAT, se);

        Mat tophat;
        tophat8.convertTo(tophat, CV_32F, 1.0 / 255.0);

        // binarizar y limpiar
        Mat bin = (tophat > (float)opts.studsThresh);
        bin.convertTo(bin, CV_8U, 255);
        bin = bin & maskU8;

        // abrir para quitar ruido fino
        morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

        // componentes conexas con filtros geométricos
        Mat labels, stats, centroids;
        int n = connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
        if (n <= 1) {
            if (dbgBinOut) *dbgBinOut = bin.clone();
            return 0.0;
        }

        int count = 0;
        for (int i = 1; i < n; ++i) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < opts.studsMinArea || area > opts.studsMaxArea) continue;

            int x = stats.at<int>(i, CC_STAT_LEFT);
            int y = stats.at<int>(i, CC_STAT_TOP);
            int w = stats.at<int>(i, CC_STAT_WIDTH);
            int h = stats.at<int>(i, CC_STAT_HEIGHT);

            Rect rc(x, y, w, h);
            rc &= Rect(0, 0, bin.cols, bin.rows);
            if (rc.area() <= 0) continue;

            Mat comp = (labels(rc) == i);
            comp.convertTo(comp, CV_8U, 255);

            std::vector<std::vector<Point>> cnts;
            findContours(comp, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (cnts.empty()) continue;

            double per = arcLength(cnts[0], true);
            if (per < 1e-9) continue;
            double circ = (4.0 * CV_PI * (double)area) / (per * per);
            if (circ < opts.studsMinCircularity) continue;

            count++;
        }

        if (dbgBinOut) *dbgBinOut = bin.clone();
        return (double)count;
    }

    // ---------------- main ----------------

    void ExtractShapeFeatures10(const Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape10Opts& opts,
        Shape10Dbg* dbg)
    {
        featNames = DefaultNames10();
        feat.assign(10, 0.0);

        if (dbg) {
            dbg->maskN.release();
            dbg->IgN.release();
            dbg->studsBin.release();
            dbg->projH_s.clear();
            dbg->projV_s.clear();
            dbg->grid.release();
        }

        try {
            if (I_in.empty()) return;

            // 0) preparar
            Mat Ig = toGrayFloat01(I_in); // CV_32F 0..1
            if (Ig.empty()) return;

            Mat mask = (Ig > (float)opts.tBlackMin);
            mask.convertTo(mask, CV_8U, 255);

            mask = bwareaopen_u8(mask, opts.minObjArea);

            morphologyEx(mask, mask, MORPH_CLOSE,
                getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.closeRadius + 1, 2 * opts.closeRadius + 1)));
            morphologyEx(mask, mask, MORPH_OPEN,
                getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.openRadius + 1, 2 * opts.openRadius + 1)));

            fillSmallHoles_keepBig(mask, opts.holeSmallMaxArea);

            if (!keepLargestComponent(mask)) {
                if (dbg) dbg->maskN = mask.clone();
                return;
            }

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

            Point2f ctr(mask.cols / 2.f, mask.rows / 2.f);
            Mat R = getRotationMatrix2D(ctr, -angleDeg, 1.0);
            Rect bbox = RotatedRect(ctr, mask.size(), (float)-angleDeg).boundingRect();
            R.at<double>(0, 2) += bbox.width / 2.0 - ctr.x;
            R.at<double>(1, 2) += bbox.height / 2.0 - ctr.y;

            Mat maskR, IgR;
            warpAffine(mask, maskR, R, bbox.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
            warpAffine(Ig, IgR, R, bbox.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

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
                IgR = IgR(bb).clone();
            }

            // normalizar escala: lado largo -> normTargetSize
            int h0 = maskR.rows, w0 = maskR.cols;
            double scale = (double)opts.normTargetSize / std::max(1, std::max(h0, w0));
            if (!std::isfinite(scale) || scale <= 0) scale = 1.0;

            Mat maskN, IgN;
            resize(maskR, maskN, Size(), scale, scale, INTER_NEAREST);
            resize(IgR, IgN, Size(), scale, scale, INTER_LINEAR);

            double A = (double)countNonZero(maskN);
            if (A < 1.0) {
                if (dbg) { dbg->maskN = maskN; dbg->IgN = IgN; }
                return;
            }

            // contorno principal
            std::vector<std::vector<Point>> cnts;
            findContours(maskN.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            if (cnts.empty()) return;

            int imax = 0; double amax = 0.0;
            for (int i = 0; i < (int)cnts.size(); ++i) {
                double a = std::abs(contourArea(cnts[i]));
                if (a > amax) { amax = a; imax = i; }
            }

            double P = std::max(arcLength(cnts[imax], true), 1e-9);

            // ---- features geométricas ----
            double AreaNorm = A / (double)maskN.total();
            double PerimNorm = P / std::max(1e-9, 2.0 * (maskN.rows + maskN.cols));

            Rect bb = boundingRect(cnts[imax]);
            double Extent = A / std::max(1.0, (double)bb.area());

            std::vector<Point> hull;
            convexHull(cnts[imax], hull);
            double hullA = std::max(std::abs(contourArea(hull)), 1e-9);
            double Solidity = A / hullA;

            double AspectRatio = (double)bb.width / std::max(1, bb.height);
            if (AspectRatio < 1.0) AspectRatio = 1.0 / std::max(1e-9, AspectRatio);

            // ---- proyecciones ----
            std::vector<double> projH(maskN.rows, 0.0);
            std::vector<double> projV(maskN.cols, 0.0);

            for (int y = 0; y < maskN.rows; ++y) {
                const uchar* p = maskN.ptr<uchar>(y);
                int rowSum = 0;
                for (int x = 0; x < maskN.cols; ++x) {
                    if (p[x]) {
                        rowSum++;
                        projV[x] += 1.0;
                    }
                }
                projH[y] = (double)rowSum;
            }

            std::vector<double> projH_s = movmean(projH, opts.projSmooth);
            std::vector<double> projV_s = movmean(projV, opts.projSmooth);

            double ProjH_entropy = entropyFromNonNeg(projH_s);
            double ProjV_entropy = entropyFromNonNeg(projV_s);
            int ProjH_peaks = countPeaks01(projH_s, 0.35);

            // ---- grid ----
            Mat G = gridOccupancy(maskN, opts.gridN);
            std::vector<double> g; g.reserve(G.rows * G.cols);
            for (int r = 0; r < G.rows; ++r)
                for (int c = 0; c < G.cols; ++c)
                    g.push_back(G.at<double>(r, c));

            double GridOccGini_3x3 = giniCoeff(g);

            // ---- studs robust ----
            Mat studsBin;
            double StudsCount = studsCountRobust(IgN, maskN, opts, (dbg ? &studsBin : nullptr));

            // ensamblar (10)
            feat[0] = AreaNorm;
            feat[1] = PerimNorm;
            feat[2] = Extent;
            feat[3] = Solidity;
            feat[4] = AspectRatio;
            feat[5] = ProjH_entropy;
            feat[6] = ProjV_entropy;
            feat[7] = (double)ProjH_peaks;
            feat[8] = GridOccGini_3x3;
            feat[9] = StudsCount;

            if (dbg) {
                dbg->maskN = maskN.clone();
                dbg->IgN = IgN.clone();
                dbg->studsBin = studsBin.clone();
                dbg->projH_s = projH_s;
                dbg->projV_s = projV_s;
                dbg->grid = G.clone();
            }
        }
        catch (...) {
            feat.assign(10, 0.0);
        }
    }

} // namespace FeatureExtractor10
