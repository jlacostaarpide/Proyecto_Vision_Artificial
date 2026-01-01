#include "ExtractCaracteristicas.h"
#include <opencv2/opencv.hpp>
#include <complex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>

using namespace cv;


//Script para obtener vectores de caracter�sticas

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
        if (in.depth() == CV_32F || in.depth() == CV_64F) {
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
		feat[3] = S_iqr; // no la usamos
        feat[4] = V_median; // no la usamos
        feat[5] = V_iqr;
        feat[6] = S_mean;
        feat[7] = V_mean;
        return feat;
    }



   
    static cv::Mat bwareaopen_u8(const cv::Mat& binU8, int minArea)
    {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(binU8.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat out = cv::Mat::zeros(binU8.size(), CV_8U);
        for (auto& c : cnts) {
            if (cv::contourArea(c) >= minArea) {
                cv::drawContours(out, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
            }
        }
        return out;
    }

    static void fillSmallHoles(cv::Mat& maskU8, int holeSmallMaxArea)
    {
        // maskU8: 0/255
        cv::Mat flood = maskU8.clone();
        cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
        cv::Mat floodInv; cv::bitwise_not(flood, floodInv);

        cv::Mat filled = maskU8 | floodInv;
        cv::Mat holes = filled & (~maskU8); // agujeros

        // queremos rellenar agujeros "pequeños" (< holeSmallMaxArea)
        std::vector<std::vector<cv::Point>> hc;
        cv::findContours(holes.clone(), hc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat holesToFill = cv::Mat::zeros(maskU8.size(), CV_8U);
        for (auto& c : hc) {
            if (cv::contourArea(c) < holeSmallMaxArea) {
                cv::drawContours(holesToFill, std::vector<std::vector<cv::Point>>{c}, 0, 255, cv::FILLED);
            }
        }
        cv::bitwise_or(maskU8, holesToFill, maskU8);
    }

    static double eccentricityFromMoments(const cv::Mat& binU8)
    {
        // binU8: 0/255, 1 componente principal ya recortada/rotada idealmente.
        // Eccentricity = sqrt(1 - (b^2/a^2)) con a>=b (ejes de la elipse de inercia)
        cv::Moments mu = cv::moments(binU8, true);
        if (mu.m00 <= 1e-9) return 0.0;

        // momentos centrales normalizados (covarianza)
        double cx = mu.m10 / mu.m00;
        double cy = mu.m01 / mu.m00;

        double mu20 = mu.mu20 / mu.m00;
        double mu02 = mu.mu02 / mu.m00;
        double mu11 = mu.mu11 / mu.m00;

        // matriz de covarianza 2x2:
        // [mu20  mu11
        //  mu11  mu02]
        double tr = mu20 + mu02;
        double det = mu20 * mu02 - mu11 * mu11;
        double disc = std::max(0.0, tr * tr - 4.0 * det);
        double s = std::sqrt(disc);

        // autovalores (>=0)
        double l1 = 0.5 * (tr + s);
        double l2 = 0.5 * (tr - s);
        if (l1 < l2) std::swap(l1, l2);
        if (l1 <= 1e-12) return 0.0;

        // semiejes proporcionales a sqrt(lambda). Como el factor común se cancela, ecc solo depende del ratio.
        double a2 = l1; // ~ a^2
        double b2 = std::max(l2, 0.0); // ~ b^2
        double ecc = std::sqrt(std::max(0.0, 1.0 - (b2 / a2)));
        return ecc;
    }

    static cv::Mat morphologicalSkeleton(const cv::Mat& binU8)
    {
        // skeleton aproximado (si no tienes thinning)
        cv::Mat skel(binU8.size(), CV_8U, cv::Scalar(0));
        cv::Mat m = binU8.clone();
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

        while (true) {
            cv::Mat eroded, opened, temp;
            cv::erode(m, eroded, element);
            cv::morphologyEx(eroded, opened, cv::MORPH_OPEN, element);
            cv::subtract(eroded, opened, temp);
            cv::bitwise_or(skel, temp, skel);
            m = eroded;
            if (cv::countNonZero(m) == 0) break;
        }
        return skel;
    }

    static double computeFD5_fromMask(const cv::Mat& maskU8, int Nboundary = 128)
    {
        // FD5 = abs(Z(6)) / abs(Z(2)) (siguiendo tu MATLAB: den=Z(2), idx=3..6 -> FD2..FD5)
        // En 0-based: den=mag[1], FD5 corresponde a mag[5]/den.
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(maskU8.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (cnts.empty()) return 0.0;

        // mayor contorno
        int imax = 0;
        double amax = 0.0;
        for (int i = 0; i < (int)cnts.size(); ++i) {
            double a = std::abs(cv::contourArea(cnts[i]));
            if (a > amax) { amax = a; imax = i; }
        }
        const auto& b = cnts[imax];
        if (b.size() < 2) return 0.0;

        // z = x + i*y, re-muestreo a Nboundary (como interp1 lineal)
        int M = (int)b.size();
        std::vector<std::complex<double>> z0(M);
        for (int i = 0; i < M; ++i) z0[i] = { (double)b[i].x, (double)b[i].y };

        std::vector<std::complex<double>> z(Nboundary);
        for (int k = 0; k < Nboundary; ++k) {
            double idx = k * (M - 1.0) / (Nboundary - 1.0);
            int i0 = (int)std::floor(idx);
            int i1 = (int)std::ceil(idx);
            if (i1 >= M) i1 = M - 1;
            double frac = idx - i0;
            z[k] = z0[i0] * (1.0 - frac) + z0[i1] * frac;
        }

        // quitar media
        std::complex<double> mean(0, 0);
        for (auto& v : z) mean += v;
        mean /= (double)z.size();
        for (auto& v : z) v -= mean;

        // DFT con OpenCV (CV_64FC2)
        cv::Mat dftIn(Nboundary, 1, CV_64FC2);
        for (int i = 0; i < Nboundary; ++i) {
            dftIn.at<cv::Vec2d>(i, 0)[0] = z[i].real();
            dftIn.at<cv::Vec2d>(i, 0)[1] = z[i].imag();
        }
        cv::Mat dftOut;
        cv::dft(dftIn, dftOut, cv::DFT_ROWS);

        std::vector<double> mag(Nboundary, 0.0);
        for (int i = 0; i < Nboundary; ++i) {
            double re = dftOut.at<cv::Vec2d>(i, 0)[0];
            double im = dftOut.at<cv::Vec2d>(i, 0)[1];
            mag[i] = std::hypot(re, im);
        }

        double den = std::max(mag.size() > 1 ? mag[1] : 0.0, 1e-12);
        int idxFD5 = 5; // 0-based -> Z(6)
        if (idxFD5 >= (int)mag.size()) return 0.0;
        return mag[idxFD5] / den;
    }

    // ============================================================
    // local_extractShapeFeatures -> SOLO 7 features como en MATLAB
    // ============================================================
    static std::vector<double> local_extractShapeFeatures(const cv::Mat& I_in)
    {
        // Parámetros MATLAB
        const double tBlackMin = 0.03;
        const int minObjArea = 300;
        const int holeSmallMaxArea = 200;
        const int closeRadius = 3;
        const int openRadius = 2;
        const int Nboundary = 128;

        std::vector<double> feat(7, 0.0);

        // 1) Asegurar I en double y en rango 0..1
        cv::Mat I;
        if (I_in.empty()) return feat;

        // Pasar a float (no double) para cvtColor
        if (I_in.depth() == CV_8U) {
            I_in.convertTo(I, CV_32F, 1.0 / 255.0);
        }
        else {
            I_in.convertTo(I, CV_32F);
            double mn, mx;
            cv::minMaxLoc(I, &mn, &mx);
            if (mx > 1.0) I *= (1.0f / 255.0f);
        }

        cv::Mat Ig;
        if (I.channels() == 3) {
            cv::cvtColor(I, Ig, cv::COLOR_BGR2GRAY);
        }
        else if (I.channels() == 4) {
            cv::cvtColor(I, Ig, cv::COLOR_BGRA2GRAY);
        }
        else {
            Ig = I.clone();
        }

        // Si tu resto de código quiere CV_64F, conviertes aquí:
        Ig.convertTo(Ig, CV_64F);

        // 2) mask = Ig > tBlackMin
        cv::Mat mask = (Ig > tBlackMin);
        mask.convertTo(mask, CV_8U, 255);

        // 3) bwareaopen
        mask = bwareaopen_u8(mask, minObjArea);

        // 4) close + open (disk)
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * closeRadius + 1, 2 * closeRadius + 1)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * openRadius + 1, 2 * openRadius + 1)));

        // 5) imfill holes + rellenar agujeros pequeños
        fillSmallHoles(mask, holeSmallMaxArea);

        // 6) quedarnos con la CC más grande
        {
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(mask.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (cnts.empty()) return feat;

            int imax = 0; double amax = 0.0;
            for (int i = 0; i < (int)cnts.size(); ++i) {
                double a = std::abs(cv::contourArea(cnts[i]));
                if (a > amax) { amax = a; imax = i; }
            }
            cv::Mat keep = cv::Mat::zeros(mask.size(), CV_8U);
            cv::drawContours(keep, cnts, imax, 255, cv::FILLED);
            mask = keep;
        }

        // 7) orientación (aprox MATLAB regionprops Orientation) usando momentos
        //    (para rotar a "horizontal" como tu MATLAB)
        double angleDeg = 0.0;
        {
            cv::Moments mu = cv::moments(mask, true);
            if (mu.m00 > 1e-9) {
                double mu20 = mu.mu20 / mu.m00;
                double mu02 = mu.mu02 / mu.m00;
                double mu11 = mu.mu11 / mu.m00;
                // ángulo del eje principal (rad): 0.5*atan2(2*mu11, mu20-mu02)
                double theta = 0.5 * std::atan2(2.0 * mu11, (mu20 - mu02));
                angleDeg = theta * 180.0 / CV_PI;
            }
        }

        // 8) rotar (loose)
        cv::Point2f ctr(mask.cols / 2.f, mask.rows / 2.f);
        cv::Mat R = cv::getRotationMatrix2D(ctr, -angleDeg, 1.0);
        cv::Rect bbox = cv::RotatedRect(ctr, mask.size(), (float)-angleDeg).boundingRect();
        R.at<double>(0, 2) += bbox.width / 2.0 - ctr.x;
        R.at<double>(1, 2) += bbox.height / 2.0 - ctr.y;

        cv::Mat maskR;
        cv::warpAffine(mask, maskR, R, bbox.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // 9) crop a bounding box de la región
        {
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(maskR.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (cnts.empty()) return feat;

            int imax = 0; double amax = 0.0;
            for (int i = 0; i < (int)cnts.size(); ++i) {
                double a = std::abs(cv::contourArea(cnts[i]));
                if (a > amax) { amax = a; imax = i; }
            }
            cv::Rect bb = cv::boundingRect(cnts[imax]);
            maskR = maskR(bb).clone();
        }

        // A partir de aquí, maskR es la equivalente a maskR de MATLAB (binaria, recortada)
        double A = (double)cv::countNonZero(maskR);
        if (A <= 1.0) return feat;

        // Perimeter P
        double P = 0.0;
        {
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(maskR.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            if (!cnts.empty()) {
                int imax = 0; double amax = 0.0;
                for (int i = 0; i < (int)cnts.size(); ++i) {
                    double a = std::abs(cv::contourArea(cnts[i]));
                    if (a > amax) { amax = a; imax = i; }
                }
                P = std::max(cv::arcLength(cnts[imax], true), 1e-9);
            }
            else {
                P = 1e-9;
            }
        }

        // Circularity
        double Circularity = (4.0 * CV_PI * A) / (P * P);

        // Extent: A / area(bounding box)
        cv::Rect bb = cv::boundingRect(maskR);
        double Extent = A / std::max(1.0, (double)bb.area());

        // Solidity: A / ConvexArea
        double ConvexArea = 0.0;
        {
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(maskR.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            if (!cnts.empty()) {
                int imax = 0; double amax = 0.0;
                for (int i = 0; i < (int)cnts.size(); ++i) {
                    double a = std::abs(cv::contourArea(cnts[i]));
                    if (a > amax) { amax = a; imax = i; }
                }
                std::vector<cv::Point> hull;
                cv::convexHull(cnts[imax], hull);
                ConvexArea = std::max(std::abs(cv::contourArea(hull)), 1e-9);
            }
            else {
                ConvexArea = 1e-9;
            }
        }
        double Solidity = A / ConvexArea;

        // Eccentricity (moments-based, equivalente a regionprops)
        double Eccentricity = eccentricityFromMoments(maskR);

        // EulerNumber = 1 - numHoles
        double EulerNumber = 1.0;
        {
            // rellenar fuera para detectar agujeros
            cv::Mat flood = maskR.clone();
            cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
            cv::Mat floodInv; cv::bitwise_not(flood, floodInv);
            cv::Mat filled = maskR | floodInv;
            cv::Mat holes = filled & (~maskR);

            cv::Mat labels;
            int nlabels = cv::connectedComponents(holes, labels, 8, CV_32S);
            int holesCount = std::max(0, nlabels - 1);
            EulerNumber = 1.0 - (double)holesCount;
        }

        // Skeleton length normalized: skelLen / sqrt(A)
        double SkelLenNorm = 0.0;
        {
            cv::Mat skel;

            // Si tienes ximgproc thinning (mejor), úsalo:
            // cv::ximgproc::thinning(maskR, skel, cv::ximgproc::THINNING_ZHANGSUEN);

            // Fallback:
            skel = morphologicalSkeleton(maskR);

            double skelLen = (double)cv::countNonZero(skel);
            SkelLenNorm = skelLen / std::max(1e-9, std::sqrt(A));
        }

        // FD5
        double FD5 = computeFD5_fromMask(maskR, Nboundary);

        feat[0] = Circularity;
        feat[1] = Extent;
        feat[2] = Solidity;
        feat[3] = Eccentricity;
        feat[4] = EulerNumber;
        feat[5] = SkelLenNorm;
        feat[6] = FD5;

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
        std::vector<double> featShape = local_extractShapeFeatures(I);   // 7

        // Map values with MATLAB layout
        double H_mean_circ = featColor[0];
        double S_median = featColor[2];
        double V_IQR = featColor[5];
        double S_mean = featColor[6];
        double V_mean = featColor[7];

        double Circularity = featShape[0];
        double Extent = featShape[1];
        double Solidity = featShape[2];
        double Eccentricity = featShape[3];
        double EulerNumber = featShape[4];
        double SkelLenNorm = featShape[5];
        // FD5 no est� en la versi�n estructural -> sustituimos por StudsCountNormArea
       // double StudsCountNormArea = featShape[13];
        double FD5 = featShape[6];

        feat = {
            Extent, Solidity, V_mean, Eccentricity, SkelLenNorm, Circularity,
            H_mean_circ, S_mean, V_IQR, S_median, FD5, EulerNumber
        };

        featNames = {
            "Extent","Solidity","V_mean","Eccentricity","SkelLenNorm","Circularity",
            "H_mean_circ","S_mean","V_IQR","S_median","FD5","EulerNumber"
        };
    }

} // namespace FeatureExtractor