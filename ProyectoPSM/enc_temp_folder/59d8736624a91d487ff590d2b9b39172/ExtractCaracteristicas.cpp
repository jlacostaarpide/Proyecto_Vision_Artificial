#include "ExtractCaracteristicas.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <complex>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;


//Script para obtener vectores de características

namespace FeatureExtractor {

    // --- Helpers ----------------------------------------------------------------
    static void percentile_and_iqr(std::vector<float>& v, double& p50, double& iqr) {
        if (v.empty()) { p50 = 0; iqr = 0; return; }
        std::sort(v.begin(), v.end());
        auto at = [&](double q)->double {
            double idx = q * (v.size() - 1);
            size_t i0 = static_cast<size_t>(floor(idx));
            size_t i1 = static_cast<size_t>(ceil(idx));
            if (i0 == i1) return v[i0];
            double frac = idx - i0;
            return v[i0] * (1.0 - frac) + v[i1] * frac;
            };
        p50 = at(0.5);
        double p25 = at(0.25);
        double p75 = at(0.75);
        iqr = p75 - p25;
    }

    static Mat toFloat01(const Mat& in) {
        Mat out;
        if (in.empty()) return out;
        if (in.type() == CV_32F || in.type() == CV_64F) {
            in.convertTo(out, CV_32F);
            double minVal, maxVal;
            minMaxLoc(out, &minVal, &maxVal);
            if (maxVal > 1.0) out = out / 255.0f;
        }
        else {
            in.convertTo(out, CV_32F, 1.0 / 255.0);
        }
        return out;
    }

    // helper: cuenta picos en vector (umbral relativo como en MATLAB)
    static int countPeaks(const std::vector<double>& xin) {
        if (xin.size() < 5) return 0;
        std::vector<double> x = xin;
        double mn = *std::min_element(x.begin(), x.end());
        for (double& v : x) v -= mn;
        double mx = *std::max_element(x.begin(), x.end());
        if (mx < 1e-9) return 0;
        for (double& v : x) v /= mx;
        std::vector<char> isPeak(x.size(), 0);
        for (size_t i = 1; i + 1 < x.size(); ++i) {
            if (x[i] > x[i - 1] && x[i] > x[i + 1]) isPeak[i] = 1;
        }
        int n = 0;
        for (size_t i = 0; i < x.size(); ++i) if (isPeak[i] && x[i] > 0.35) ++n;
        return n;
    }

    // helper: rejilla NxN con fracción de pixeles "on" por celda
    static Mat gridOccupancy(const Mat& mask, int N) {
        Mat out = Mat::zeros(N, N, CV_64F);
        int h = mask.rows, w = mask.cols;
        std::vector<int> ys(N + 1), xs(N + 1);
        for (int i = 0; i <= N; ++i) {
            ys[i] = static_cast<int>(std::round(i * (h / static_cast<double>(N))));
            xs[i] = static_cast<int>(std::round(i * (w / static_cast<double>(N))));
        }
        // ensure last index covers end
        ys[N] = h; xs[N] = w;
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int y1 = std::max(0, ys[r]);
                int y2 = std::min(h, ys[r + 1]);
                int x1 = std::max(0, xs[c]);
                int x2 = std::min(w, xs[c + 1]);
                if (y2 <= y1 || x2 <= x1) { out.at<double>(r, c) = 0.0; continue; }
                Rect rc(x1, y1, x2 - x1, y2 - y1);
                Mat cell = mask(rc);
                double nn = countNonZero(cell);
                double frac = nn / static_cast<double>(rc.area());
                out.at<double>(r, c) = frac;
            }
        }
        return out;
    }

    // helper: gini coefficient
    static double giniCoeff(const std::vector<double>& xin) {
        std::vector<double> x = xin;
        for (double& v : x) if (v < 0) v = 0;
        double s = std::accumulate(x.begin(), x.end(), 0.0);
        if (s < 1e-12) return 0.0;
        std::sort(x.begin(), x.end());
        int n = static_cast<int>(x.size());
        double num = 0.0;
        for (int i = 0; i < n; ++i) num += (i + 1) * x[i];
        double g = (2.0 * num) / (n * s) - (n + 1.0) / n;
        return std::abs(g);
    }

    // ----------------- Color features (8) ---------------------------------------
    static std::vector<double> local_extractColorFeatures(const Mat& I_float01) {
        // I_float01: BGR float 0..1
        std::vector<double> feat(8, 0.0);
        if (I_float01.empty()) return feat;

        Mat I = I_float01.clone();

        // Convert BGR (0..1) -> Lab (L in 0..100 when src is float)
        Mat Ilab;
        cvtColor(I, Ilab, COLOR_BGR2Lab); // float -> L in [0..100]
        std::vector<Mat> labChannels;
        split(Ilab, labChannels);
        Mat L = labChannels[0]; // 0..100
        Mat A = labChannels[1];
        Mat B = labChannels[2];

        // CLAHE on L: convert to 8U [0..255], apply CLAHE, back to 0..100
        Mat L8;
        L.convertTo(L8, CV_8U, 255.0 / 100.0);
        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(3.0);
        Mat Lcl;
        clahe->apply(L8, Lcl);
        Mat Lcorr;
        Lcl.convertTo(Lcorr, CV_32F, 100.0 / 255.0);

        // reconstruct Lab and convert back to BGR (float 0..1)
        std::vector<Mat> lab2 = { Lcorr, A, B };
        Mat Ilab2;
        merge(lab2, Ilab2);
        Mat I_corr;
        cvtColor(Ilab2, I_corr, COLOR_Lab2BGR);
        // clamp to 0..1
        cv::min(I_corr, 1.0f, I_corr);
        cv::max(I_corr, 0.0f, I_corr);

        // HSV (expects float 0..1)
        Mat hsv;
        cvtColor(I_corr, hsv, COLOR_BGR2HSV);
        std::vector<Mat> hsvC;
        split(hsv, hsvC);
        Mat H = hsvC[0]; // H in [0..1]
        Mat S = hsvC[1];
        Mat V = hsvC[2];

        // mask = any(I > 0 in original) & (V > 0.05)
        Mat anyNonZero;
        std::vector<Mat> bgrChannels;
        split(I, bgrChannels);
        anyNonZero = (bgrChannels[0] > 0) | (bgrChannels[1] > 0) | (bgrChannels[2] > 0);
        Mat mask = anyNonZero & (V > 0.05f);

        // extract samples
        std::vector<float> Hv, Sv, Vv;
        Hv.reserve(1024); Sv.reserve(1024); Vv.reserve(1024);
        for (int r = 0; r < mask.rows; ++r) {
            const uchar* pm = mask.ptr<uchar>(r);
            const float* pH = H.ptr<float>(r);
            const float* pS = S.ptr<float>(r);
            const float* pV = V.ptr<float>(r);
            for (int c = 0; c < mask.cols; ++c) {
                if (pm[c]) {
                    Hv.push_back(pH[c]);
                    Sv.push_back(pS[c]);
                    Vv.push_back(pV[c]);
                }
            }
        }

        if (Hv.empty()) {
            return feat; // zeros
        }

        // Circular mean and circular variance for H
        double sumRe = 0.0, sumIm = 0.0;
        for (float h : Hv) {
            double ang = 2.0 * CV_PI * static_cast<double>(h);
            sumRe += cos(ang);
            sumIm += sin(ang);
        }
        double meanRe = sumRe / Hv.size();
        double meanIm = sumIm / Hv.size();
        std::complex<double> R(meanRe, meanIm);
        double H_mean_circ = std::atan2(meanIm, meanRe) / (2.0 * CV_PI);
        if (H_mean_circ < 0) H_mean_circ += 1.0;
        double H_var_circ = 1.0 - std::abs(R);

        // S_median, S_IQR, V_median, V_IQR
        double S_median = 0.0, S_iqr = 0.0, V_median = 0.0, V_iqr = 0.0;
        percentile_and_iqr(Sv, S_median, S_iqr);
        percentile_and_iqr(Vv, V_median, V_iqr);

        // S_mean, V_mean
        double S_mean = 0.0, V_mean = 0.0;
        for (float s : Sv) S_mean += s;
        for (float v : Vv) V_mean += v;
        S_mean /= Sv.size();
        V_mean /= Vv.size();

        feat[0] = H_mean_circ;
        feat[1] = H_var_circ;
        feat[2] = S_median;
        feat[3] = S_iqr;
        feat[4] = V_median;
        feat[5] = V_iqr;
        feat[6] = S_mean;
        feat[7] = V_mean;
        return feat;
    }

    // ----------------- Shape features (14) - versión compatible con MATLAB -------------
    // Implementa la lógica del local_extractShapeFeatures(matlab) usada por ExtractColorShapeFeatures.
    static std::vector<double> local_extractShapeFeatures_14(const cv::Mat& I_float01) {
        const double tBlackMin = 0.03;
        const int minObjArea = 300;
        const int holeSmallMaxArea = 200;
        const int closeRadius = 3;
        const int openRadius = 2;
        const int Nboundary = 128;
        const int Kfourier = 5;

        std::vector<double> feat(14, 0.0);
        if (I_float01.empty()) return feat;

        // grayscale
        cv::Mat Ig;
        if (I_float01.channels() == 3)
            cv::cvtColor(I_float01, Ig, cv::COLOR_BGR2GRAY);
        else
            Ig = I_float01.clone();

        // binary mask
        cv::Mat mask = (Ig > tBlackMin);

        // remove small objects (bwareaopen)
        {
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8U);
            for (auto& c : cnts)
                if (cv::contourArea(c) >= minObjArea)
                    cv::drawContours(clean, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
            mask = (clean > 0);
        }

        // close + open
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * closeRadius + 1, 2 * closeRadius + 1)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * openRadius + 1, 2 * openRadius + 1)));

        // fill holes and keep large holes
        cv::Mat maskU8; mask.convertTo(maskU8, CV_8U, 255);
        cv::Mat flood = maskU8.clone();
        cv::floodFill(flood, Point(0, 0), Scalar(255));
        cv::Mat floodInv; cv::bitwise_not(flood, floodInv);
        cv::Mat maskFilled = maskU8 | floodInv;
        cv::Mat holes = maskFilled & (~maskU8);

        cv::Mat holesToFill = cv::Mat::zeros(holes.size(), CV_8U);
        {
            std::vector<std::vector<cv::Point>> hc;
            cv::findContours(holes.clone(), hc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (auto& c : hc)
                if (cv::contourArea(c) < holeSmallMaxArea)
                    cv::drawContours(holesToFill, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
        }
        cv::bitwise_or(maskU8, holesToFill, maskU8);
        mask = (maskU8 > 0);

        // keep largest CC
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (cnts.empty()) return feat;
        int imax = 0;
        double amax = 0;
        for (int i = 0; i < (int)cnts.size(); ++i) {
            double a = cv::contourArea(cnts[i]);
            if (a > amax) { amax = a; imax = i; }
        }

        // orientation via PCA on largest contour
        cv::Mat data((int)cnts[imax].size(), 2, CV_64F);
        for (int i = 0; i < data.rows; ++i) {
            data.at<double>(i, 0) = cnts[imax][i].x;
            data.at<double>(i, 1) = cnts[imax][i].y;
        }
        cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
        double angle = atan2(pca.eigenvectors.at<double>(0, 1),
            pca.eigenvectors.at<double>(0, 0)) * 180.0 / CV_PI;

        // rotate + crop (loose)
        cv::Point2f ctr(mask.cols / 2.f, mask.rows / 2.f);
        cv::Mat R = cv::getRotationMatrix2D(ctr, -angle, 1.0);
        cv::Rect bbox = cv::RotatedRect(ctr, mask.size(), -angle).boundingRect();
        R.at<double>(0, 2) += bbox.width / 2.0 - ctr.x;
        R.at<double>(1, 2) += bbox.height / 2.0 - ctr.y;

        cv::Mat maskR;
        cv::warpAffine(maskU8, maskR, R, bbox.size(), cv::INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

        // crop to bounding box of largest contour in rotated image
        std::vector<std::vector<cv::Point>> cntR;
        cv::findContours(maskR.clone(), cntR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (cntR.empty()) return feat;
        int imaxR = 0; double amaxR = 0;
        for (int i = 0; i < (int)cntR.size(); ++i) {
            double a = cv::contourArea(cntR[i]);
            if (a > amaxR) { amaxR = a; imaxR = i; }
        }
        cv::Rect bb = cv::boundingRect(cntR[imaxR]);
        cv::Mat maskRc = maskR(bb);
        cv::Mat maskN = (maskRc > 0); // NOTE: no scale normalization here, faithful a la versión MATLAB

        // ---------------- region props ----------------
        std::vector<std::vector<cv::Point>> cntsN;
        cv::findContours(maskN.clone(), cntsN, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (cntsN.empty()) return feat;
        int idxN = 0; double aNmax = 0;
        for (int i = 0; i < (int)cntsN.size(); ++i) {
            double a = cv::contourArea(cntsN[i]);
            if (a > aNmax) { aNmax = a; idxN = i; }
        }
        double A = cv::countNonZero(maskN);
        double P = std::max(cv::arcLength(cntsN[idxN], true), 1e-9);
        double circ = 4.0 * CV_PI * A / (P * P + 1e-9);

        cv::Rect bbN = cv::boundingRect(cntsN[idxN]);
        double ext = A / std::max(1.0, (double)bbN.area());

        std::vector<cv::Point> hull;
        cv::convexHull(cntsN[idxN], hull);
        double convexA = std::max(1e-9, cv::contourArea(hull));
        double conv = A / convexA;

        double maj = std::max(bbN.width, bbN.height);
        double mino = std::min(bbN.width, bbN.height);
        double aspect = maj / std::max(1.0, mino);
        double ecc = sqrt(std::max(0.0, 1.0 - (mino * mino) / (maj * maj)));

        // Euler + holes
        cv::Mat filled = maskN.clone();
        {
            Mat temp = maskN.clone();
            temp.convertTo(temp, CV_8U, 255);
            Mat fl = temp.clone();
            floodFill(fl, Point(0, 0), Scalar(255));
            Mat flInv; bitwise_not(fl, flInv);
            Mat filledU8 = temp | flInv;
            filled = (filledU8 > 0);
        }
        cv::Mat holesN = (filled & (~maskN));
        Mat labels;
        int nlabels = cv::connectedComponents(holesN, labels);
        int HolesCount = std::max(0, nlabels - 1);
        double EulerNumber = 1 - HolesCount;

        // ---------------- skeleton ----------------
        cv::Mat skel = cv::Mat::zeros(maskN.size(), CV_8U);
        cv::Mat m = maskN.clone();
        cv::Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
        while (true) {
            cv::Mat eroded; cv::erode(m, eroded, element);
            cv::Mat tempOpen; cv::morphologyEx(eroded, tempOpen, MORPH_OPEN, element);
            cv::Mat diff = eroded - tempOpen;
            cv::bitwise_or(skel, diff, skel);
            m = eroded.clone();
            if (countNonZero(m) == 0) break;
        }
        double skelLen = cv::countNonZero(skel);
        double skelLenNorm = skelLen / std::max(1.0, std::sqrt(A));

        int nEnd = 0, nBranch = 0;
        for (int r = 1; r < skel.rows - 1; ++r) {
            for (int c = 1; c < skel.cols - 1; ++c) {
                if (!skel.at<uchar>(r, c)) continue;
                int n = 0;
                for (int rr = -1; rr <= 1; ++rr)
                    for (int cc = -1; cc <= 1; ++cc)
                        if (rr != 0 || cc != 0)
                            n += skel.at<uchar>(r + rr, c + cc) ? 1 : 0;
                if (n == 1) nEnd++;
                else if (n >= 3) nBranch++;
            }
        }

        // ---------------- fourier descriptors FD2..FD5 ----------------
        std::vector<double> fd(4, 0.0);
        {
            // use the contour points (CHAIN_APPROX_NONE gives full boundary)
            std::vector<cv::Point> b = cntsN[idxN];
            if (!b.empty()) {
                int M = static_cast<int>(b.size());
                // create complex vector z = x + i*y (MATLAB used x + i*y where x=b(:,2), y=b(:,1))
                std::vector<std::complex<double>> z0(M);
                for (int i = 0; i < M; ++i) {
                    double x = static_cast<double>(b[i].x);
                    double y = static_cast<double>(b[i].y);
                    z0[i] = std::complex<double>(x, y);
                }
                // resample/interpolate to Nboundary points (linear)
                std::vector<std::complex<double>> z(Nboundary);
                if (M == 1) {
                    for (int k = 0; k < Nboundary; ++k) z[k] = z0[0];
                } else {
                    for (int k = 0; k < Nboundary; ++k) {
                        double idx = k * (M - 1.0) / (Nboundary - 1.0);
                        int i0 = static_cast<int>(floor(idx));
                        int i1 = static_cast<int>(ceil(idx));
                        if (i1 >= M) i1 = M - 1;
                        double frac = idx - i0;
                        z[k] = z0[i0] * (1.0 - frac) + z0[i1] * frac;
                    }
                }
                // subtract mean
                std::complex<double> mean(0, 0);
                for (auto& v : z) mean += v;
                mean /= (double)z.size();
                for (auto& v : z) v -= mean;

                // prepare cv::Mat complex for DFT (CV_64FC2)
                cv::Mat dftIn(Nboundary, 1, CV_64FC2);
                for (int i = 0; i < Nboundary; ++i) {
                    dftIn.at<cv::Vec2d>(i, 0)[0] = z[i].real();
                    dftIn.at<cv::Vec2d>(i, 0)[1] = z[i].imag();
                }
                cv::Mat dftOut;
                cv::dft(dftIn, dftOut, cv::DFT_ROWS);

                // magnitudes
                std::vector<double> mag(Nboundary, 0.0);
                for (int i = 0; i < Nboundary; ++i) {
                    double re = dftOut.at<cv::Vec2d>(i, 0)[0];
                    double im = dftOut.at<cv::Vec2d>(i, 0)[1];
                    mag[i] = std::hypot(re, im);
                }
                double den = std::max(mag.size() > 1 ? mag[1] : 0.0, 1e-12);
                // MATLAB selected indices Z(3..6) -> zero-based 2..5
                for (int k = 0; k < 4; ++k) {
                    int idx = 2 + k;
                    if (idx < (int)mag.size()) fd[k] = mag[idx] / den;
                }
            }
        }

        // ---------------- assemble ----------------
        feat = {
            circ, aspect, ext, conv, conv /* placeholder for Solidity? */,
            ecc, EulerNumber,
            skelLenNorm, (double)nEnd, (double)nBranch,
            fd[0], fd[1], fd[2], fd[3]
        };

        // Note: MATLAB order in your provided script is:
        // [circ, aspect, ext, sol, conv, ecc, euler, skelLenNorm, nEnd, nBranch, FD2, FD3, FD4, FD5]
        // Above we placed 'conv' twice mistakenly in position 4 (index 3). Fix mapping to match MATLAB:
        // set correct Solidity (sol) as ratio A/ConvexArea, and conv as ratio A/ConvexArea? MATLAB had conv = A / ConvexArea and sol = S.Solidity earlier
        // We computed 'conv' as A / convexA and 'conv' variable currently holds that. We must also compute Solidity from regionprops; we didn't compute solidity earlier.
        // Compute Solidity properly (A / convexA) is actually solidity, and conv (in MATLAB script) was defined as A / max(S.ConvexArea,1e-9) which is the same as solidity in practise.
        // To follow the exact ordering expected by the MATLAB snippet, we set:
        // position 4: sol (solidity)
        // position 5: conv (convexity-like) - but both are same here; keep same value.

        double Solidity = A / convexA;
        double ConvMetric = A / convexA;

        feat[3] = Solidity;   // sol
        feat[4] = ConvMetric; // conv

        return feat;
    }

    // ----------------- Shape features (24, NEW STRUCTURAL VERSION) -----------------
<<<<<<< HEAD
    //static std::vector<double> local_extractShapeFeatures(const cv::Mat& I_float01) {
=======
    static std::vector<double> local_extractShapeFeatures_24(const cv::Mat& I_float01) {
>>>>>>> main

    //    const double tBlackMin = 0.03;
    //    const int minObjArea = 300;
    //    const int holeSmallMaxArea = 200;
    //    const int closeRadius = 3;
    //    const int openRadius = 2;
    //    const int normTargetSize = 220;
    //    const int projSmooth = 7;
    //    const int gridN = 3;

    //    // studs
    //    const bool studs_enable = true;
    //    const int studs_rmin = 6;
    //    const int studs_rmax = 20;

    //    std::vector<double> feat(24, 0.0);
    //    if (I_float01.empty()) return feat;

    //    // ---------------- 0) prepare ----------------
    //    cv::Mat Ig;
    //    if (I_float01.channels() == 3)
    //        cv::cvtColor(I_float01, Ig, cv::COLOR_BGR2GRAY);
    //    else
    //        Ig = I_float01.clone();

    //    cv::Mat mask = (Ig > tBlackMin);

    //    // remove small objects
    //    {
    //        std::vector<std::vector<cv::Point>> cnts;
    //        cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //        cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8U);
    //        for (auto& c : cnts)
    //            if (cv::contourArea(c) >= minObjArea)
    //                cv::drawContours(clean, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
    //        mask = (clean > 0);
    //    }

    //    // morph close + open
    //    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
    //        cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * closeRadius + 1, 2 * closeRadius + 1)));
    //    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
    //        cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * openRadius + 1, 2 * openRadius + 1)));

    //    // fill holes (keep large holes)
    //    cv::Mat maskU8; mask.convertTo(maskU8, CV_8U, 255);
    //    cv::Mat flood = maskU8.clone();
    //    // ensure border is background: if border pixel is foreground, set point slightly inside
    //    cv::floodFill(flood, Point(0, 0), Scalar(255));
    //    cv::Mat floodInv; cv::bitwise_not(flood, floodInv);
    //    cv::Mat maskFilled = maskU8 | floodInv;
    //    cv::Mat holes = maskFilled & (~maskU8);

    //    cv::Mat holesToFill = cv::Mat::zeros(holes.size(), CV_8U);
    //    {
    //        std::vector<std::vector<cv::Point>> hc;
    //        cv::findContours(holes.clone(), hc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //        for (auto& c : hc)
    //            if (cv::contourArea(c) < holeSmallMaxArea)
    //                cv::drawContours(holesToFill, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
    //    }
    //    cv::bitwise_or(maskU8, holesToFill, maskU8);
    //    mask = (maskU8 > 0);

    //    // keep largest CC
    //    std::vector<std::vector<cv::Point>> cnts;
    //    cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //    if (cnts.empty()) return feat;
    //    int imax = 0;
    //    double amax = 0;
    //    for (int i = 0; i < (int)cnts.size(); ++i) {
    //        double a = cv::contourArea(cnts[i]);
    //        if (a > amax) { amax = a; imax = i; }
    //    }

    //    // orientation via PCA on largest contour
    //    cv::Mat data((int)cnts[imax].size(), 2, CV_64F);
    //    for (int i = 0; i < data.rows; ++i) {
    //        data.at<double>(i, 0) = cnts[imax][i].x;
    //        data.at<double>(i, 1) = cnts[imax][i].y;
    //    }
    //    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    //    double angle = atan2(pca.eigenvectors.at<double>(0, 1),
    //        pca.eigenvectors.at<double>(0, 0)) * 180.0 / CV_PI;

    //    // rotate + crop
    //    cv::Point2f ctr(mask.cols / 2.f, mask.rows / 2.f);
    //    cv::Mat R = cv::getRotationMatrix2D(ctr, -angle, 1.0);
    //    cv::Rect bbox = cv::RotatedRect(ctr, mask.size(), -angle).boundingRect();
    //    R.at<double>(0, 2) += bbox.width / 2.0 - ctr.x;
    //    R.at<double>(1, 2) += bbox.height / 2.0 - ctr.y;

    //    cv::Mat maskR;
    //    cv::warpAffine(maskU8, maskR, R, bbox.size(), cv::INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

    //    // crop to bounding box of largest contour in rotated image
    //    std::vector<std::vector<cv::Point>> cntR;
    //    cv::findContours(maskR.clone(), cntR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //    if (cntR.empty()) return feat;
    //    int imaxR = 0; double amaxR = 0;
    //    for (int i = 0; i < (int)cntR.size(); ++i) {
    //        double a = cv::contourArea(cntR[i]);
    //        if (a > amaxR) { amaxR = a; imaxR = i; }
    //    }
    //    cv::Rect bb = cv::boundingRect(cntR[imaxR]);
    //    cv::Mat maskRc = maskR(bb);
    //    cv::Mat maskRbin = (maskRc > 0);

    //    // scale normalize
    //    double scale = normTargetSize / (double)std::max(maskRbin.rows, maskRbin.cols);
    //    if (!std::isfinite(scale) || scale <= 0) scale = 1.0;
    //    cv::Mat maskResized;
    //    cv::resize(maskRbin, maskResized, cv::Size(), scale, scale, cv::INTER_NEAREST);
    //    cv::Mat maskN = (maskResized > 0);

    //    // ---------------- 1) region props ----------------
    //    // contours on maskN
    //    std::vector<std::vector<cv::Point>> cntsN;
    //    cv::findContours(maskN.clone(), cntsN, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    //    if (cntsN.empty()) return feat;
    //    int idxN = 0; double aNmax = 0;
    //    for (int i = 0; i < (int)cntsN.size(); ++i) {
    //        double a = cv::contourArea(cntsN[i]);
    //        if (a > aNmax) { aNmax = a; idxN = i; }
    //    }
    //    double A = cv::countNonZero(maskN);
    //    double P = std::max(cv::arcLength(cntsN[idxN], true), 1e-9);
    //    double AreaNorm = A / maskN.total();
    //    double PerimNorm = P / std::max(2.0 * (maskN.rows + maskN.cols), 1.0);
    //    double Circularity = 4.0 * CV_PI * A / (P * P + 1e-9);

    //    cv::Rect bbN = cv::boundingRect(cntsN[idxN]);
    //    double Extent = A / std::max(1.0, (double)bbN.area());

    //    std::vector<cv::Point> hull;
    //    cv::convexHull(cntsN[idxN], hull);
    //    double convexA = std::max(1e-9, cv::contourArea(hull));
    //    double Solidity = A / convexA;

    //    double maj = std::max(bbN.width, bbN.height);
    //    double mino = std::min(bbN.width, bbN.height);
    //    double AspectRatio = maj / std::max(1.0, mino);
    //    double Eccentricity = sqrt(std::max(0.0, 1.0 - (mino * mino) / (maj * maj)));

    //    // Euler + holes
    //    cv::Mat filled = maskN.clone();
    //    // fill holes
    //    {
    //        Mat temp = maskN.clone();
    //        temp.convertTo(temp, CV_8U, 255);
    //        Mat fl = temp.clone();
    //        floodFill(fl, Point(0, 0), Scalar(255));
    //        Mat flInv; bitwise_not(fl, flInv);
    //        Mat filledU8 = temp | flInv;
    //        filled = (filledU8 > 0);
    //    }
    //    cv::Mat holesN = (filled & (~maskN));
    //    Mat labels;
    //    int nlabels = cv::connectedComponents(holesN, labels);
    //    int HolesCount = std::max(0, nlabels - 1);
    //    double HolesAreaFrac = cv::countNonZero(holesN) / std::max(1.0, A);
    //    double EulerNumber = 1 - HolesCount;

    //    // ---------------- 2) skeleton ----------------
    //    // morphological thinning (Zhang-Suen like) via iterative approach
    //    cv::Mat skel = cv::Mat::zeros(maskN.size(), CV_8U);
    //    cv::Mat m = maskN.clone();
    //    cv::Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    //    while (true) {
    //        cv::Mat eroded; cv::erode(m, eroded, element);
    //        cv::Mat tempOpen; cv::morphologyEx(eroded, tempOpen, MORPH_OPEN, element);
    //        cv::Mat diff = eroded - tempOpen;
    //        cv::bitwise_or(skel, diff, skel);
    //        m = eroded.clone();
    //        if (countNonZero(m) == 0) break;
    //    }
    //    double SkelLenNorm = cv::countNonZero(skel) / std::max(1.0, std::sqrt(A));

    //    int SkelEndpoints = 0, SkelBranchpoints = 0;
    //    for (int r = 1; r < skel.rows - 1; ++r) {
    //        for (int c = 1; c < skel.cols - 1; ++c) {
    //            if (!skel.at<uchar>(r, c)) continue;
    //            int n = 0;
    //            for (int rr = -1; rr <= 1; ++rr)
    //                for (int cc = -1; cc <= 1; ++cc)
    //                    if (rr != 0 || cc != 0)
    //                        n += skel.at<uchar>(r + rr, c + cc) ? 1 : 0;
    //            if (n == 1) SkelEndpoints++;
    //            else if (n >= 3) SkelBranchpoints++;
    //        }
    //    }

    //    // ---------------- 3) projections ----------------
    //    std::vector<double> projV(maskN.cols), projH(maskN.rows);
    //    for (int c = 0; c < maskN.cols; c++) projV[c] = countNonZero(maskN.col(c));
    //    for (int r = 0; r < maskN.rows; r++) projH[r] = countNonZero(maskN.row(r));

    //    auto entropy = [&](const std::vector<double>& v) {
    //        double s = std::accumulate(v.begin(), v.end(), 0.0), e = 0;
    //        if (s < 1e-12) return 0.0;
    //        for (double x : v) { double p = x / s; if (p > 1e-12) e -= p * log(p); }
    //        return e;
    //        };

    //    int ProjV_peaks = countPeaks(projV);
    //    int ProjH_peaks = countPeaks(projH);
    //    double ProjV_entropy = entropy(projV);
    //    double ProjH_entropy = entropy(projH);

    //    // ---------------- 4) grid ----------------
    //    cv::Mat G = gridOccupancy(maskN, gridN);
    //    std::vector<double> g;
    //    g.reserve(G.total());
    //    for (int r = 0; r < G.rows; ++r) for (int c = 0; c < G.cols; ++c) g.push_back(G.at<double>(r, c));
    //    double GridOccFrac = std::count_if(g.begin(), g.end(), [](double v) {return v > 0.15; }) / (double)g.size();
    //    double GridOccGini = giniCoeff(g);
    //    double GridOccDiagDiff = std::abs(G.at<double>(0, 0) + G.at<double>(1, 1) + G.at<double>(2, 2)
    //        - (G.at<double>(0, 2) + G.at<double>(1, 1) + G.at<double>(2, 0)));

    //    // ---------------- 5) studs ----------------
    //    std::vector<cv::Vec3f> circles;
    //    // Hough requires 8U gray image; maskN already CV_8U with 0/255 values
    //    cv::Mat maskForHough;
    //    maskN.convertTo(maskForHough, CV_8U, 255);
    //    cv::HoughCircles(maskForHough, circles, cv::HOUGH_GRADIENT, 1.2, 15, 100, 20, studs_rmin, studs_rmax);
    //    std::vector<double> radii;
    //    for (auto& c : circles) radii.push_back(c[2]);

    //    double StudsCount = static_cast<double>(radii.size());
    //    double StudsCountNormArea = StudsCount / std::max(1.0, A / 1e4);
    //    double StudsMeanRadius = radii.empty() ? 0.0 : std::accumulate(radii.begin(), radii.end(), 0.0) / radii.size();
    //    double StudsRadiusStd = 0.0;
    //    if (!radii.empty()) {
    //        double ss = 0.0;
    //        for (double r : radii) ss += (r - StudsMeanRadius) * (r - StudsMeanRadius);
    //        StudsRadiusStd = std::sqrt(ss / radii.size());
    //    }

    //    // ---------------- assemble ----------------
    //    feat = {
    //        AreaNorm, PerimNorm, Circularity, Extent, Solidity,
    //        Eccentricity, AspectRatio, EulerNumber,
    //        (double)HolesCount, HolesAreaFrac,
    //        SkelLenNorm, (double)SkelEndpoints, (double)SkelBranchpoints,
    //        (double)ProjV_peaks, (double)ProjH_peaks,
    //        ProjV_entropy, ProjH_entropy,
    //        GridOccFrac, GridOccGini, GridOccDiagDiff,
    //        StudsCount, StudsCountNormArea, StudsMeanRadius, StudsRadiusStd
    //    };

    //    return feat;
    //}

    // ----------------- Shape features (14) --------------------------------------
    static std::vector<double> local_extractShapeFeatures(const Mat& I_float01) {
        const double tBlackMin = 0.03;
        const int minObjArea = 300;
        const int holeSmallMaxArea = 200;
        const int closeRadius = 3;
        const int openRadius = 2;
        const int Nboundary = 128;
        const int Kfourier = 5;

        std::vector<double> feat(14, 0.0);
        if (I_float01.empty()) return feat;

        Mat Ig;
        if (I_float01.channels() == 3) cvtColor(I_float01, Ig, COLOR_BGR2GRAY);
        else Ig = I_float01.clone();

        // mask
        Mat mask = (Ig > tBlackMin);

        // remove small objects 
        {
            std::vector<std::vector<Point>> contours;
            findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            Mat cleaned = Mat::zeros(mask.size(), CV_8U);
            for (const auto& c : contours) {
                double a = contourArea(c);
                if (a >= minObjArea) {
                    drawContours(cleaned, std::vector<std::vector<Point>>{c}, 0, Scalar(255), FILLED);
                }
            }
            mask = (cleaned > 0);
        }

        // Morph close and open
        Mat seClose = getStructuringElement(MORPH_ELLIPSE, Size(2 * closeRadius + 1, 2 * closeRadius + 1));
        Mat seOpen = getStructuringElement(MORPH_ELLIPSE, Size(2 * openRadius + 1, 2 * openRadius + 1));
        morphologyEx(mask, mask, MORPH_CLOSE, seClose);
        morphologyEx(mask, mask, MORPH_OPEN, seOpen);

        // imfill (fill holes)
        Mat temp;
        mask.convertTo(temp, CV_8U, 255);
        Mat im_flood = temp.clone();
        // flood fill from borders: ensure border pixels are background
        floodFill(im_flood, Point(0, 0), Scalar(255));
        Mat im_flood_inv;
        bitwise_not(im_flood, im_flood_inv);
        Mat maskFilled = temp | im_flood_inv; // filled

        // holes = maskFilled & ~mask
        Mat holes;
        Mat maskU8 = temp;
        bitwise_and(maskFilled, (~maskU8), holes);

        // holesSmall = bwareaopen(holes, holeSmallMaxArea) -> we will keep holes >= holeSmallMaxArea
        // holesToFill = holes & ~holesSmall -> i.e. holes smaller than threshold
        Mat holesToFill = Mat::zeros(holes.size(), CV_8U);
        {
            std::vector<std::vector<Point>> contours;
            findContours(holes.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (const auto& c : contours) {
                double a = contourArea(c);
                if (a < holeSmallMaxArea) {
                    drawContours(holesToFill, std::vector<std::vector<Point>>{c}, 0, Scalar(255), FILLED);
                }
            }
        }
        Mat mask2;
        bitwise_or(maskU8, holesToFill, mask2);
        mask = (mask2 > 0);

        // Connected components: keep largest component if multiple
        {
            std::vector<std::vector<Point>> contours;
            findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (contours.empty()) return feat;
            if (contours.size() > 1) {
                double maxA = 0; int imax = 0;
                for (size_t i = 0; i < contours.size(); ++i) {
                    double a = contourArea(contours[i]);
                    if (a > maxA) { maxA = a; imax = static_cast<int>(i); }
                }
                Mat keep = Mat::zeros(mask.size(), CV_8U);
                drawContours(keep, contours, imax, Scalar(255), FILLED);
                mask = (keep > 0);
            }
        }

        // Rotate to principal orientation and crop bounding box
        std::vector<std::vector<Point>> cnts;
        findContours(mask.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (cnts.empty()) return feat;
        std::vector<Point> contour = cnts[0];
        // compute orientation using PCA on contour points
        Mat dataPts(static_cast<int>(contour.size()), 2, CV_64F);
        for (size_t i = 0; i < contour.size(); ++i) {
            dataPts.at<double>((int)i, 0) = contour[i].x;
            dataPts.at<double>((int)i, 1) = contour[i].y;
        }
        // get covariance and eigenvectors
        Mat mean;
        reduce(dataPts, mean, 0, REDUCE_AVG);
        Mat centered = dataPts - repeat(mean, dataPts.rows, 1);
        Mat cov = (centered.t() * centered) / static_cast<double>(dataPts.rows);
        Mat evals, evecs;
        eigen(cov, evals, evecs); // evecs: rows are eigenvectors
        double angle = atan2(evecs.at<double>(0, 1), evecs.at<double>(0, 0)); // principal direction
        double angleDeg = angle * 180.0 / CV_PI;
        // rotate mask
        Mat maskU8_forRot;
        mask.convertTo(maskU8_forRot, CV_8U, 255);
        Mat maskR;
        // rotate around center, use warpAffine with sufficient bounding size by using warpAffine with flags and get rotation matrix
        Point2f center((float)maskU8_forRot.cols / 2.0f, (float)maskU8_forRot.rows / 2.0f);
        Mat Rm = getRotationMatrix2D(center, -angleDeg, 1.0);
        // compute bounding box of rotated image
        Rect bbox = RotatedRect(center, maskU8_forRot.size(), -angleDeg).boundingRect();
        // adjust transformation
        Rm.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        Rm.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        warpAffine(maskU8_forRot, maskR, Rm, bbox.size(), INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

        // crop to bounding box of maskR
        std::vector<std::vector<Point>> cntR;
        findContours(maskR.clone(), cntR, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (cntR.empty()) return feat;
        Rect bb = boundingRect(cntR[0]);
        Mat maskRc = maskR(bb);
        Mat maskRbin = (maskRc > 0);

        // regionprops: area, perimeter, ecc, solidity, extent, major/minor axis length, convex area, euler
        double A = contourArea(cntR[0]);
        double P = arcLength(cntR[0], true);
        if (P < 1e-9) P = 1e-9;
        double circ = (4.0 * CV_PI * A) / (P * P);

        // fitEllipse for major/minor
        double maj = 1.0, mino = 1.0;
        if (cntR[0].size() >= 5) {
            RotatedRect ell = fitEllipse(cntR[0]);
            maj = std::max(ell.size.width, ell.size.height);
            mino = std::min(ell.size.width, ell.size.height);
        }
        else {
            // fallback: use bounding box
            maj = std::max((double)maskRbin.cols, (double)maskRbin.rows);
            mino = std::min((double)maskRbin.cols, (double)maskRbin.rows);
        }
        double aspect = maj / std::max(mino, 1.0);

        // convex area
        std::vector<Point> hull;
        convexHull(cntR[0], hull);
        double convexA = std::max(1e-9, contourArea(hull));
        double sol = A / convexA;
        double ext = A / std::max(1.0, bb.width * (double)bb.height);
        double ecc = sqrt(std::max(0.0, 1.0 - (mino * mino) / (maj * maj)));
        double conv = A / convexA;

        // Euler number: components - holes ; compute using RETR_CCOMP
        Mat maskForEuler = maskRbin.clone();
        std::vector<std::vector<Point>> contoursAll;
        std::vector<Vec4i> hierarchy;
        findContours(maskForEuler, contoursAll, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        int numExternal = 0, numHoles = 0;
        for (size_t i = 0; i < contoursAll.size(); ++i) {
            if (hierarchy[i][3] == -1) numExternal++;
            else numHoles++;
        }
        int euler = numExternal - numHoles;

        // skeletonization (by morphological thinning using morphological operations)
        Mat skel = Mat::zeros(maskRbin.size(), CV_8U);
        Mat tempErode, tempOpen;
        Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
        Mat m = maskRbin.clone();
        while (true) {
            erode(m, tempErode, element);
            morphologyEx(tempErode, tempOpen, MORPH_OPEN, element);
            Mat diff = tempErode - tempOpen;
            bitwise_or(skel, diff, skel);
            m = tempErode.clone();
            if (countNonZero(m) == 0) break;
        }
        int skelLen = countNonZero(skel);
        double skelLenNorm = skelLen / std::max(1.0, sqrt(A));

        // endpoints and branchpoints: count neighbor sums
        int nEnd = 0, nBranch = 0;
        Mat skelU8 = (skel > 0);
        for (int r = 0; r < skelU8.rows; ++r) {
            for (int c = 0; c < skelU8.cols; ++c) {
                if (!skelU8.at<uchar>(r, c)) continue;
                int nbrs = 0;
                for (int rr = -1; rr <= 1; ++rr) {
                    for (int cc = -1; cc <= 1; ++cc) {
                        if (rr == 0 && cc == 0) continue;
                        int nr = r + rr, nc = c + cc;
                        if (nr >= 0 && nr < skelU8.rows && nc >= 0 && nc < skelU8.cols) {
                            if (skelU8.at<uchar>(nr, nc)) nbrs++;
                        }
                    }
                }
                if (nbrs == 1) nEnd++;
                else if (nbrs >= 3) nBranch++;
            }
        }

        // boundary and Fourier descriptors
        std::vector<double> fd(4, 0.0);
        std::vector<std::vector<Point>> bcont;
        findContours(maskRbin.clone(), bcont, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (!bcont.empty()) {
            std::vector<Point> b = bcont[0];
            int n = static_cast<int>(b.size());
            if (n >= 2) {
                // create complex vector z = x + i*y (matching MATLAB ordering: col + i*row -> x + i*y)
                std::vector<std::complex<double>> z;
                z.reserve(n);
                for (int i = 0; i < n; ++i) z.emplace_back(static_cast<double>(b[i].x), static_cast<double>(b[i].y));
                // resample linearly by index to Nboundary
                std::vector<std::complex<double>> zres(Nboundary);
                for (int k = 0; k < Nboundary; ++k) {
                    double pos = ((double)k * (n - 1)) / (Nboundary - 1);
                    int i0 = static_cast<int>(floor(pos));
                    int i1 = std::min(i0 + 1, n - 1);
                    double frac = pos - i0;
                    zres[k] = z[i0] * (1.0 - frac) + z[i1] * frac;
                }
                // subtract mean
                std::complex<double> meanz(0, 0);
                for (auto& cpx : zres) meanz += cpx;
                meanz /= (double)Nboundary;
                std::vector<std::complex<double>> zc(Nboundary);
                for (int i = 0; i < Nboundary; ++i) zc[i] = zres[i] - meanz;
                // compute DFT (use cv::dft on two-channel real/imag)
                Mat planes[2];
                planes[0] = Mat::zeros(Nboundary, 1, CV_64F);
                planes[1] = Mat::zeros(Nboundary, 1, CV_64F);
                for (int i = 0; i < Nboundary; ++i) {
                    planes[0].at<double>(i, 0) = zc[i].real();
                    planes[1].at<double>(i, 0) = zc[i].imag();
                }
                Mat complexI;
                merge(std::vector<Mat>{planes[0], planes[1]}, complexI);
                Mat spectrum;
                dft(complexI, spectrum, DFT_ROWS);
                // compute magnitude array
                std::vector<double> mag(Nboundary, 0.0);
                for (int i = 0; i < Nboundary; ++i) {
                    Vec2d v = spectrum.at<Vec2d>(i, 0);
                    mag[i] = std::hypot(v[0], v[1]);
                }
                double den = std::max(mag.size() > 1 ? mag[1] : 0.0, 1e-12);
                // select indices 2..(1+Kfourier) in 0-based (matching MATLAB idx=3:(2+Kfourier))
                std::vector<double> magSel;
                for (int k = 2; k <= 1 + Kfourier && k < (int)mag.size(); ++k) magSel.push_back(mag[k] / den);
                // FD2..FD5 -> first 4 entries of magSel (if missing pad with zeros)
                for (int i = 0; i < 4; ++i) {
                    if (i < (int)magSel.size()) fd[i] = magSel[i];
                    else fd[i] = 0.0;
                }
            }
        }

        // assemble feat vector in the same order as MATLAB
        feat[0] = circ;
        feat[1] = aspect;
        feat[2] = ext;
        feat[3] = sol;
        feat[4] = conv;
        feat[5] = ecc;
        feat[6] = static_cast<double>(euler);
        feat[7] = skelLenNorm;
        feat[8] = static_cast<double>(nEnd);
        feat[9] = static_cast<double>(nBranch);
        feat[10] = fd[0];
        feat[11] = fd[1];
        feat[12] = fd[2];
        feat[13] = fd[3];

        return feat;
    }


    // Extrae las 12 caracteristicas de color y forma 
    void ExtractColorShapeFeatures(const Mat& I_in, std::vector<double>& feat, std::vector<std::string>& featNames) {
        feat.clear();
        featNames.clear();
        if (I_in.empty()) {
            feat.assign(12, 0.0);
            featNames = { "Extent","Solidity","V_mean","Eccentricity","SkelLenNorm","Circularity",
                          "H_mean_circ","S_mean","V_IQR","S_median","FD5","EulerNumber" };
            return;
        }

        Mat I = toFloat01(I_in); // BGR float 0..1
        // Ensure 3 channels
        if (I.channels() == 1) cvtColor(I, I, COLOR_GRAY2BGR);

        std::vector<double> featColor = local_extractColorFeatures(I);   // 8
<<<<<<< HEAD
        std::vector<double> featShape = local_extractShapeFeatures(I);   // 14 o 24
=======
        std::vector<double> featShape = local_extractShapeFeatures_14(I);   // 14 (MATLAB-like)
>>>>>>> main

        // Map values with MATLAB layout
        double H_mean_circ = featColor[0];
        double S_median = featColor[2];
        double V_IQR = featColor[5];
        double S_mean = featColor[6];
        double V_mean = featColor[7];

        double Circularity = featShape[0];
        double Extent = featShape[2];
        double Solidity = featShape[3];
        double Eccentricity = featShape[5];
        double EulerNumber = featShape[6];
        double SkelLenNorm = featShape[7];
<<<<<<< HEAD
        // FD5 no está en la versión estructural -> sustituimos por StudsCountNormArea
        double StudsCountNormArea = featShape[13];
=======
        double FD5 = featShape[13]; // FD5 en la posición 14 (1-based) -> índice 13
>>>>>>> main

        feat = {
            Extent, Solidity, V_mean, Eccentricity, SkelLenNorm, Circularity,
            H_mean_circ, S_mean, V_IQR, S_median, FD5, EulerNumber
        };

        featNames = {
            "Extent","Solidity","V_mean","Eccentricity","SkelLenNorm","Circularity",
            "H_mean_circ","S_mean","V_IQR","S_median","FD5","EulerNumber"
        };
    }

//    // Extrae las 24 caracteristicas de forma (actualizada)
//    void ExtractShapeFeatures(const Mat& I_in, std::vector<double>& feat, std::vector<std::string>& featNames) {
//        feat.clear();
//        featNames.clear();
//        if (I_in.empty()) {
//            feat.assign(24, 0.0);
//            featNames = {
//                "AreaNorm","PerimNorm","Circularity","Extent","Solidity","Eccentricity","AspectRatio","EulerNumber",
//                "HolesCount","HolesAreaFrac","SkelLenNorm","SkelEndpoints","SkelBranchpoints",
//                "ProjV_peaks","ProjH_peaks","ProjV_entropy","ProjH_entropy",
//                "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3",
//                "StudsCount","StudsCountNormArea","StudsMeanRadius","StudsRadiusStd"
//            };
//            return;
//        }
//
//        Mat I = toFloat01(I_in); // BGR float 0..1
//        if (I.channels() == 1) cvtColor(I, I, COLOR_GRAY2BGR);
//
//        std::vector<double> sfeat = local_extractShapeFeatures(I); // 24
//        feat = sfeat;
//
//        featNames = {
//            "AreaNorm","PerimNorm","Circularity","Extent","Solidity","Eccentricity","AspectRatio","EulerNumber",
//            "HolesCount","HolesAreaFrac","SkelLenNorm","SkelEndpoints","SkelBranchpoints",
//            "ProjV_peaks","ProjH_peaks","ProjV_entropy","ProjH_entropy",
//            "GridOccFrac_3x3","GridOccGini_3x3","GridOccDiagDiff_3x3",
//            "StudsCount","StudsCountNormArea","StudsMeanRadius","StudsRadiusStd"
//        };
//    }
//

<<<<<<< HEAD
=======
        Mat I = toFloat01(I_in); // BGR float 0..1
        if (I.channels() == 1) cvtColor(I, I, COLOR_GRAY2BGR);

        std::vector<double> sfeat = local_extractShapeFeatures_14(I); // 24
        feat = sfeat;
>>>>>>> main

// Extrae las 14 caracteristicas de forma 
void ExtractShapeFeatures(const Mat& I_in, std::vector<double>& feat, std::vector<std::string>& featNames) {
    feat.clear();
    featNames.clear();
    if (I_in.empty()) {
        feat.assign(14, 0.0);
        featNames = {
            "Circularity","AspectRatio","Extent","Solidity","ConvexFrac","Eccentricity","EulerNumber",
            "SkelLenNorm","SkelEndpoints","SkelBranchpoints","FD2","FD3","FD4","FD5"
        };
        return;
    }

    Mat I = toFloat01(I_in); // BGR float 0..1
    if (I.channels() == 1) cvtColor(I, I, COLOR_GRAY2BGR);

    std::vector<double> sfeat = local_extractShapeFeatures(I); // 14
    feat = sfeat;

    featNames = {
        "Circularity","AspectRatio","Extent","Solidity","ConvexFrac","Eccentricity","EulerNumber",
        "SkelLenNorm","SkelEndpoints","SkelBranchpoints","FD2","FD3","FD4","FD5"
    };
}





} // namespace FeatureExtractor