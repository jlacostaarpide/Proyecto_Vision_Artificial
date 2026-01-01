#include "ExtractCaracteristicas24Refinador.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>


using namespace cv;

namespace FeatureExtractor24 {

    // ---------- helpers ----------
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
        else g = f;
        return g;
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
        // maskU8: 0/255. Rellena SOLO agujeros pequeños (como MATLAB).
        Mat flood = maskU8.clone();
        floodFill(flood, Point(0, 0), Scalar(255));
        Mat floodInv; bitwise_not(flood, floodInv);

        Mat filled = maskU8 | floodInv;            // imfill(mask,'holes')
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

    static double giniCoeff(const std::vector<double>& xin) {
        std::vector<double> x = xin;
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

    static int countPeaks(const std::vector<double>& xin) {
        if (xin.size() < 5) return 0;
        std::vector<double> x = xin;
        double mn = *std::min_element(x.begin(), x.end());
        for (double& v : x) v -= mn;
        double mx = *std::max_element(x.begin(), x.end());
        if (mx < 1e-9) return 0;
        for (double& v : x) v /= mx;

        int n = 0;
        for (int i = 1; i + 1 < (int)x.size(); ++i) {
            bool isPeak = (x[i] > x[i - 1] && x[i] > x[i + 1]);
            if (isPeak && x[i] > 0.35) ++n;
        }
        return n;
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

    static Mat morphologicalSkeleton(const Mat& binU8) {
        Mat skel(binU8.size(), CV_8U, Scalar(0));
        Mat m = binU8.clone();
        Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

        while (true) {
            Mat eroded, opened, temp;
            erode(m, eroded, element);
            morphologyEx(eroded, opened, MORPH_OPEN, element);
            subtract(eroded, opened, temp);
            bitwise_or(skel, temp, skel);
            m = eroded;
            if (countNonZero(m) == 0) break;
        }
        return skel;
    }

    static void skelEndpointsBranchpoints(const Mat& skelU8, int& endpoints, int& branchpoints) {
        // Cuenta vecinos 8-conectados: endpoints=1 vecino, branchpoints>=3 vecinos
        endpoints = 0; branchpoints = 0;
        if (skelU8.empty()) return;

        Mat sk;
        skelU8.convertTo(sk, CV_8U);
        for (int y = 1; y < sk.rows - 1; ++y) {
            const uchar* p = sk.ptr<uchar>(y);
            for (int x = 1; x < sk.cols - 1; ++x) {
                if (p[x] == 0) continue;
                int nb = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const uchar* q = sk.ptr<uchar>(y + dy);
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (q[x + dx]) nb++;
                    }
                }
                if (nb == 1) endpoints++;
                else if (nb >= 3) branchpoints++;
            }
        }
    }

    static void regionMomentsAxesEcc(const Mat& maskU8,
        double& area, double& perimeter,
        double& extent, double& solidity,
        double& eccentricity, double& aspectRatio,
        double& eulerNumber) {
        area = (double)countNonZero(maskU8);
        perimeter = 1e-9;

        // contorno principal
        std::vector<std::vector<Point>> cnts;
        findContours(maskU8.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (cnts.empty() || area < 1.0) {
            extent = solidity = eccentricity = aspectRatio = 0.0;
            eulerNumber = 1.0;
            return;
        }
        int imax = 0; double amax = 0;
        for (int i = 0; i < (int)cnts.size(); ++i) {
            double a = std::abs(contourArea(cnts[i]));
            if (a > amax) { amax = a; imax = i; }
        }
        perimeter = std::max(arcLength(cnts[imax], true), 1e-9);

        Rect bb = boundingRect(cnts[imax]);
        extent = area / std::max(1.0, (double)bb.area());

        std::vector<Point> hull;
        convexHull(cnts[imax], hull);
        double convexA = std::max(std::abs(contourArea(hull)), 1e-9);
        solidity = area / convexA;

        // momentos -> covarianza
        Moments mu = moments(maskU8, true);
        if (mu.m00 <= 1e-9) {
            eccentricity = 0.0; aspectRatio = 0.0;
        }
        else {
            double mu20 = mu.mu20 / mu.m00;
            double mu02 = mu.mu02 / mu.m00;
            double mu11 = mu.mu11 / mu.m00;

            double tr = mu20 + mu02;
            double det = mu20 * mu02 - mu11 * mu11;
            double disc = std::max(0.0, tr * tr - 4.0 * det);
            double s = std::sqrt(disc);

            double l1 = 0.5 * (tr + s);
            double l2 = 0.5 * (tr - s);
            if (l1 < l2) std::swap(l1, l2);
            l1 = std::max(l1, 1e-12);
            l2 = std::max(l2, 1e-12);

            // ecc = sqrt(1 - b^2/a^2)
            eccentricity = std::sqrt(std::max(0.0, 1.0 - (l2 / l1)));

            // AspectRatio ~ sqrt(l1)/sqrt(l2)
            aspectRatio = std::sqrt(l1) / std::sqrt(l2);
        }

        // EulerNumber = 1 - holesCount (con agujeros "grandes")
        Mat flood = maskU8.clone();
        floodFill(flood, Point(0, 0), Scalar(255));
        Mat floodInv; bitwise_not(flood, floodInv);
        Mat filled = maskU8 | floodInv;
        Mat holes = filled & (~maskU8);

        Mat labels;
        int nlabels = connectedComponents(holes, labels, 8, CV_32S);
        int holesCount = std::max(0, nlabels - 1);
        eulerNumber = 1.0 - (double)holesCount;
    }

    static void detectStudsHough(const Mat& IgFloat01, const Mat& maskU8,
        int rmin, int rmax, double sensitivity,
        double& studsCount, double& studsCountNormArea,
        double& studsMeanRadius, double& studsRadiusStd,
        double areaA) {
        studsCount = 0; studsCountNormArea = 0; studsMeanRadius = 0; studsRadiusStd = 0;
        if (IgFloat01.empty() || countNonZero(maskU8) == 0) return;

        Mat Ieq;
        // "adapthisteq" aproximado: CLAHE sobre 8U
        Mat I8; IgFloat01.convertTo(I8, CV_8U, 255.0);
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        clahe->apply(I8, Ieq);

        // gradiente
        Mat gx, gy; Sobel(Ieq, gx, CV_32F, 1, 0, 3);
        Sobel(Ieq, gy, CV_32F, 0, 1, 3);
        Mat gmag; magnitude(gx, gy, gmag);

        // enmascarar fondo
        Mat gmask = Mat::zeros(gmag.size(), CV_32F);
        gmag.copyTo(gmask, maskU8);

        // HoughCircles necesita 8U/float 1 canal
        Mat g8; normalize(gmask, g8, 0, 255, NORM_MINMAX);
        g8.convertTo(g8, CV_8U);

        std::vector<cv::Vec3f> circles;

        // param2 controla "sensitivity" (más bajo => detecta más)
        double param2 = 60.0 - 50.0 * std::clamp(sensitivity, 0.0, 1.0);

        cv::HoughCircles(g8, circles, cv::HOUGH_GRADIENT,
            1.0,                 // dp
            rmin * 2.0,          // minDist (base; luego hacemos NMS mejor)
            100.0,               // param1
            param2,              // param2
            rmin, rmax);

        // --- 1) Filtrar candidatos válidos (centro dentro de pieza) ---
        struct Cand { float x, y, r, score; };
        std::vector<Cand> cand;
        cand.reserve(circles.size());

        for (const auto& c : circles) {
            float cx = c[0], cy = c[1], rr = c[2];
            int ix = (int)std::round(cx);
            int iy = (int)std::round(cy);
            if (ix < 0 || iy < 0 || ix >= maskU8.cols || iy >= maskU8.rows) continue;
            if (maskU8.at<uchar>(iy, ix) == 0) continue;

            // score simple: preferimos radios más grandes y centros más "limpios" (opcional)
            // Si no quieres score, pon score = rr;
            float sc = rr;
            cand.push_back({ cx, cy, rr, sc });
        }

        if (cand.empty()) return;

        // --- 2) NMS por distancia entre centros ---
        // Idea: ordenar por score y quedarnos con el mejor de cada zona.
        // Umbral de distancia: ~ 0.8*(r_i + r_j) suele funcionar bien para studs.
        std::sort(cand.begin(), cand.end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });

        std::vector<Cand> kept;
        kept.reserve(cand.size());

        auto dist2 = [](float ax, float ay, float bx, float by) {
            float dx = ax - bx, dy = ay - by;
            return dx * dx + dy * dy;
            };

        for (const auto& c : cand) {
            bool ok = true;
            for (const auto& k : kept) {
                float dthr = 0.8f * (c.r + k.r);      // ajustable (0.7..1.0)
                float dthr2 = dthr * dthr;
                if (dist2(c.x, c.y, k.x, k.y) < dthr2) { ok = false; break; }
            }
            if (ok) kept.push_back(c);
        }

        if (kept.empty()) return;

        // --- 3) Estadísticos finales ---
        studsCount = (double)kept.size();
        studsCountNormArea = studsCount / std::max(areaA / 1e4, 1e-9);

        double sumR = 0.0;
        for (const auto& k : kept) sumR += (double)k.r;
        studsMeanRadius = sumR / std::max(1.0, (double)kept.size());

        double var = 0.0;
        for (const auto& k : kept) {
            double dr = (double)k.r - studsMeanRadius;
            var += dr * dr;
        }
        var /= std::max(1.0, (double)kept.size());
        studsRadiusStd = std::sqrt(var);

    }

    // ---------- main 24 features ----------
    void ExtractShapeFeatures24(const Mat& I_in,
        std::vector<double>& feat,
        std::vector<std::string>& featNames,
        const Shape24Opts& opts) {
        feat.clear();
        featNames = {
            "AreaNorm","PerimNorm","Circularity","Extent","Solidity","Eccentricity","AspectRatio","EulerNumber",
            "HolesCount","HolesAreaFrac","SkelLenNorm","SkelEndpoints","SkelBranchpoints",
            "ProjV_peaks","ProjH_peaks","ProjV_entropy","ProjH_entropy",
            "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3",
            "StudsCount","StudsCountNormArea","StudsMeanRadius","StudsRadiusStd"
        };

        feat.assign(24, 0.0);
        if (I_in.empty()) return;

        // 0) preparar (mask base)
        Mat Ig = toGrayFloat01(I_in);
        if (Ig.empty()) return;

        Mat mask = (Ig > (float)opts.tBlackMin);
        mask.convertTo(mask, CV_8U, 255);

        mask = bwareaopen_u8(mask, opts.minObjArea);
        morphologyEx(mask, mask, MORPH_CLOSE,
            getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.closeRadius + 1, 2 * opts.closeRadius + 1)));
        morphologyEx(mask, mask, MORPH_OPEN,
            getStructuringElement(MORPH_ELLIPSE, Size(2 * opts.openRadius + 1, 2 * opts.openRadius + 1)));

        // conservar agujeros grandes, rellenar pequeños
        fillSmallHoles_keepBig(mask, opts.holeSmallMaxArea);

        // componente mayor
        {
            std::vector<std::vector<Point>> cnts;
            findContours(mask.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (cnts.empty()) return;
            int imax = 0; double amax = 0;
            for (int i = 0; i < (int)cnts.size(); ++i) {
                double a = std::abs(contourArea(cnts[i]));
                if (a > amax) { amax = a; imax = i; }
            }
            Mat keep = Mat::zeros(mask.size(), CV_8U);
            drawContours(keep, cnts, imax, 255, FILLED);
            mask = keep;
        }

        // orientación (moments) y rotación "loose"
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
            int imax = 0; double amax = 0;
            for (int i = 0; i < (int)cnts.size(); ++i) {
                double a = std::abs(contourArea(cnts[i]));
                if (a > amax) { amax = a; imax = i; }
            }
            Rect bb = boundingRect(cnts[imax]);
            maskR = maskR(bb).clone();
            IgR = IgR(bb).clone();
        }

        // normalizar escala (lado largo = normTargetSize)
        int h = maskR.rows, w = maskR.cols;
        double scale = (double)opts.normTargetSize / std::max(1, std::max(h, w));
        if (!std::isfinite(scale) || scale <= 0) scale = 1.0;

        Mat maskN, IgN;
        resize(maskR, maskN, Size(), scale, scale, INTER_NEAREST);
        resize(IgR, IgN, Size(), scale, scale, INTER_LINEAR);

        double A = (double)countNonZero(maskN);
        if (A < 1.0) return;

        // 1) props globales
        double area, perim, extent, solidity, ecc, aspect, euler;
        regionMomentsAxesEcc(maskN, area, perim, extent, solidity, ecc, aspect, euler);

        double AreaNorm = area / (double)maskN.total();
        double PerimNorm = perim / std::max(1e-9, 2.0 * (maskN.rows + maskN.cols));
        double Circularity = (4.0 * CV_PI * area) / std::max(1e-12, perim * perim);

        // agujeros "grandes" (estructura)
        Mat flood = maskN.clone();
        floodFill(flood, Point(0, 0), Scalar(255));
        Mat floodInv; bitwise_not(flood, floodInv);
        Mat filled = maskN | floodInv;
        Mat holes = filled & (~maskN);

        Mat hlab;
        int nH = connectedComponents(holes, hlab, 8, CV_32S);
        int HolesCount = std::max(0, nH - 1);
        double HolesAreaFrac = (double)countNonZero(holes) / std::max(area, 1e-9);

        // 2) skeleton
        Mat skel = morphologicalSkeleton(maskN);
        double SkelLenNorm = (double)countNonZero(skel) / std::max(1e-9, std::sqrt(area));
        int SkelEndpoints = 0, SkelBranchpoints = 0;
        skelEndpointsBranchpoints(skel, SkelEndpoints, SkelBranchpoints);

        // 3) proyecciones
        std::vector<double> projV(maskN.cols, 0.0), projH(maskN.rows, 0.0);
        for (int y = 0; y < maskN.rows; ++y) {
            const uchar* p = maskN.ptr<uchar>(y);
            int rowSum = 0;
            for (int x = 0; x < maskN.cols; ++x) {
                if (p[x]) { rowSum++; projV[x] += 1.0; }
            }
            projH[y] = (double)rowSum;
        }

        // suavizado movmean
        auto movmean = [&](const std::vector<double>& x, int W) {
            int n = (int)x.size();
            std::vector<double> y(n, 0.0);
            int hw = std::max(1, W / 2);
            for (int i = 0; i < n; ++i) {
                int a = std::max(0, i - hw);
                int b = std::min(n - 1, i + hw);
                double s = 0; int cnt = 0;
                for (int k = a; k <= b; ++k) { s += x[k]; cnt++; }
                y[i] = s / std::max(1, cnt);
            }
            return y;
            };

        std::vector<double> projV_s = movmean(projV, opts.projSmooth);
        std::vector<double> projH_s = movmean(projH, opts.projSmooth);

        // entropía
        auto entropy = [&](const std::vector<double>& x) {
            double sum = std::accumulate(x.begin(), x.end(), 0.0);
            sum = std::max(sum, 1e-12);
            double e = 0.0;
            for (double v : x) {
                double p = v / sum;
                e += -p * std::log(p + 1e-12);
            }
            return e;
            };
        double ProjV_entropy = entropy(projV_s);
        double ProjH_entropy = entropy(projH_s);

        int ProjV_peaks = countPeaks(projV_s);
        int ProjH_peaks = countPeaks(projH_s);

        // 4) grid occupancy
        Mat G = gridOccupancy(maskN, opts.gridN);
        std::vector<double> g;
        g.reserve(opts.gridN * opts.gridN);
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

        // 5) studs
        double StudsCount = 0, StudsCountNormArea = 0, StudsMeanRadius = 0, StudsRadiusStd = 0;
        if (opts.studsEnable) {
            detectStudsHough(IgN, maskN,
                opts.studsRmin, opts.studsRmax, opts.studsSensitivity,
                StudsCount, StudsCountNormArea, StudsMeanRadius, StudsRadiusStd,
                area);
        }

        // ensamblar (24)
        feat = {
            AreaNorm, PerimNorm, Circularity, extent, solidity, ecc, aspect, euler,
            (double)HolesCount, HolesAreaFrac, SkelLenNorm, (double)SkelEndpoints, (double)SkelBranchpoints,
            (double)ProjV_peaks, (double)ProjH_peaks, ProjV_entropy, ProjH_entropy,
            GridOccFrac_3x3, GridOccGini_3x3, GridOccDiagDiff_3x3,
            StudsCount, StudsCountNormArea, StudsMeanRadius, StudsRadiusStd
        };
    }

} // namespace FeatureExtractor24
