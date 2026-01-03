#include "Segmentacion.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <QDebug>

using namespace cv;
using namespace std;

// --- MÉTODO PRINCIPAL ---
vector<ResultadoPieza> Segmentacion::Segmentar(const Mat& inputBGR, DebugInfo* debug)
{
    vector<ResultadoPieza> resultados;
    if (inputBGR.empty()) return resultados;

    // Guardar original si se pide debug
    if (debug) debug->I_orig = inputBGR.clone();

    // =========================================================
    // 1. PRE-PROCESAMIENTO: CORRECCIÓN DE COLOR Y FONDO
    // =========================================================

    // Convertir a float [0..1]
    Mat imgFloat;
    inputBGR.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    // Separar canales B, G, R
    vector<Mat> channelsBGR;
    split(imgFloat, channelsBGR);
    Mat B = channelsBGR[0];
    Mat G = channelsBGR[1];
    Mat R = channelsBGR[2];

    // Calcular medias de cada canal
    Scalar meanScalar = mean(imgFloat);
    double mean_B = meanScalar[0];
    double mean_G = meanScalar[1];
    double mean_R = meanScalar[2];

    // Balance de blancos (referencia: G)
    Mat R_bal, G_bal, B_bal;

    // Evitar división por cero
    double scale_R = (mean_R > 0.001) ? (mean_G / mean_R) : 1.0;
    double scale_B = (mean_B > 0.001) ? (mean_G / mean_B) : 1.0;

    multiply(R, scale_R, R_bal);
    G_bal = G.clone();
    multiply(B, scale_B, B_bal);

    // Reconstruir imagen balanceada
    vector<Mat> channelsBalanced = { B_bal, G_bal, R_bal };
    Mat I_balanced;
    merge(channelsBalanced, I_balanced);

    // Clampear valores > 1.0
    threshold(I_balanced, I_balanced, 1.0, 1.0, THRESH_TRUNC);

    // =========================================================
    // 2. TRANSFORMACIÓN Y MEJORA HSV
    // =========================================================

    Mat hsv_temp;
    cvtColor(I_balanced, hsv_temp, COLOR_BGR2HSV);
    vector<Mat> channelsHSV;
    split(hsv_temp, channelsHSV);
    Mat H_raw = channelsHSV[0]; // 0..360
    Mat S_raw = channelsHSV[1]; // 0..1
    Mat V_raw = channelsHSV[2]; // 0..1

    // --- A. Boost de Saturación (S * 1.55) ---
    Mat S_boosted;
    multiply(S_raw, 1.55, S_boosted);
    threshold(S_boosted, S_boosted, 1.0, 1.0, THRESH_TRUNC); // Clamp a 1.0

    // --- B. Corrección de Iluminación en V (División por Gaussiana) ---
    Mat V_filt;
    GaussianBlur(V_raw, V_filt, Size(0, 0), 120);

    double minVal, maxValFilt;
    minMaxLoc(V_filt, &minVal, &maxValFilt);

    Mat V_corrected;
    if (maxValFilt > 0.0001) {
        divide(V_raw, maxValFilt, V_corrected);
    }
    else {
        V_corrected = V_raw.clone();
    }
    threshold(V_corrected, V_corrected, 1.0, 1.0, THRESH_TRUNC);

    // --- Reconstruir HSV corregido y volver a RGB ---
    channelsHSV[0] = H_raw;
    channelsHSV[1] = S_boosted;
    channelsHSV[2] = V_corrected;
    merge(channelsHSV, hsv_temp);

    Mat I_corrected;
    cvtColor(hsv_temp, I_corrected, COLOR_HSV2BGR);

    // GUARDAR DEBUG: Normalizada
    if (debug) {
        Mat debugNorm;
        I_corrected.convertTo(debugNorm, CV_8U, 255.0);
        debug->I_norm = debugNorm;
    }

    // --- Obtener canales finales para segmentación ---
    Mat I_hsv_final;
    cvtColor(I_corrected, I_hsv_final, COLOR_BGR2HSV);
    split(I_hsv_final, channelsHSV);
    Mat H = channelsHSV[0];
    Mat S_final = channelsHSV[1]; // Este es el S que usaremos
    Mat V = channelsHSV[2];

    // GUARDAR DEBUG: Canales HSV
    if (debug) {
        H.convertTo(debug->H, CV_8U, 1.0);
        S_final.convertTo(debug->S, CV_8U, 255.0);
        V.convertTo(debug->V, CV_8U, 255.0);
    }

    // =========================================================
    // 3. SEGMENTACIÓN (OTSU EN CANAL S)
    // =========================================================

    // Convertir S a 8-bit [0..255] para usar Otsu de OpenCV
    Mat S_8u;
    S_final.convertTo(S_8u, CV_8U, 255.0);
    if (debug) debug->S_proc = S_8u.clone();

    Mat mask_S;
    // THRESH_OTSU calcula automáticamente el umbral óptimo
    threshold(S_8u, mask_S, 0, 255, THRESH_BINARY | THRESH_OTSU);
    if (debug) debug->mask_otsu = mask_S.clone();

    // =========================================================
    // 4. MORFOLOGÍA
    // =========================================================

    // A. Sutura inicial (imclose disk 3 -> Size 7x7)
    Mat se_suture = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    Mat mask_morph;
    morphologyEx(mask_S, mask_morph, MORPH_CLOSE, se_suture);

    // B. Relleno de huecos (imfill)
    mask_morph = ImFillHoles(mask_morph);
    if (debug) debug->mask_fill = mask_morph.clone();

    // C. Limpieza de ruido (imopen disk 3 -> Size 7x7)
    Mat se_noise = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask_morph, mask_morph, MORPH_OPEN, se_noise);
    if (debug) debug->mask_clean = mask_morph.clone();

    // D. Eliminar bordes (imclearborder)
    mask_morph = ImClearBorder(mask_morph);
    if (debug) debug->mask_border = mask_morph.clone();

    // E. Cierre grande (imclose disk 14 -> Size 29x29)
    Mat se_merge = getStructuringElement(MORPH_ELLIPSE, Size(29, 29));
    morphologyEx(mask_morph, mask_morph, MORPH_CLOSE, se_merge);

    // F. Relleno final
    mask_morph = ImFillHoles(mask_morph);
    if (debug) debug->mask_close = mask_morph.clone();

    // G. Suavizado final (imopen disk 4 -> Size 9x9)
    Mat se_smooth = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    Mat mask_final;
    morphologyEx(mask_morph, mask_final, MORPH_OPEN, se_smooth);
    if (debug) debug->mask_final = mask_final.clone();

    // =========================================================
    // 5. EXTRACCIÓN Y FILTRADO
    // =========================================================

    vector<vector<Point>> contours;
    findContours(mask_final, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        qDebug() << "ALERTA: No se encontraron contornos tras la morfología.";
        return resultados;
    }

    // Calcular área máxima para filtro relativo
    double max_area = 0;
    for (const auto& cnt : contours) {
        double a = contourArea(cnt);
        if (a > max_area) max_area = a;
    }

    // Umbrales definidos en MATLAB
    double umbral_area_rel = 0.15 * max_area;
    double umbral_area_abs = 1000.0;
    double umbral_ratio_max = 4.0;
    double umbral_saturacion = 0.30;

    int id_counter = 1;
    qDebug() << "--- INICIO SEGMENTACIÓN ---";
    qDebug() << "Max Area:" << max_area << " | Umbral Relativo:" << umbral_area_rel;

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);

        // 1. Filtro Área
        if (area <= umbral_area_rel || area <= umbral_area_abs) {
            // qDebug() << "  [Descartado] Area insuficiente:" << area;
            continue;
        }

        Rect bbox = boundingRect(cnt);

        // 2. Filtro Aspect Ratio
        double w = (double)bbox.width;
        double h = (double)bbox.height;
        double ratio = std::max(w, h) / std::min(w, h); // max(width, height) / min(width, height)

        if (ratio > umbral_ratio_max) {
            qDebug() << "  [Descartado] Ratio alargado:" << ratio;
            continue;
        }

        // 3. Filtro Saturación Promedio
        // Creamos una máscara local solo para este objeto para calcular la media de S
        Mat maskObj = Mat::zeros(S_final.size(), CV_8U);
        drawContours(maskObj, vector<vector<Point>>{cnt}, 0, Scalar(255), FILLED);

        // Calcular media dentro de la máscara
        Scalar meanSatScalar = mean(S_final, maskObj); // S_final es float [0..1]
        double meanSat = meanSatScalar[0];

        if (meanSat <= umbral_saturacion) {
            qDebug() << "  [Descartado] Saturacion baja:" << meanSat << "(Umbral:" << umbral_saturacion << ")";
            continue;
        }

        // --- OBJETO ACEPTADO ---
        qDebug() << "  [ACEPTADO] ID:" << id_counter << " | Area:" << area << " | Sat:" << meanSat;

        // Calcular métricas adicionales
        Moments mu = moments(cnt);
        Point2f mc(mu.m10 / (mu.m00 + 1e-5), mu.m01 / (mu.m00 + 1e-5));

        double perimeter = arcLength(cnt, true);
        double circularity = 0.0;
        if (perimeter > 0) circularity = (4 * CV_PI * area) / (perimeter * perimeter);

        // Generar Crop Final (Fondo Negro)
        Mat cropBGR = inputBGR(bbox).clone();

        // Máscara local para el crop
        Mat maskLocalCrop = Mat::zeros(cropBGR.size(), CV_8U);
        vector<Point> cntShifted = cnt;
        for (auto& pt : cntShifted) {
            pt.x -= bbox.x;
            pt.y -= bbox.y;
        }
        vector<vector<Point>> cntsShifted = { cntShifted };
        drawContours(maskLocalCrop, cntsShifted, 0, Scalar(255), FILLED);

        // Aplicar máscara
        Mat cropMasked;
        cropBGR.copyTo(cropMasked, maskLocalCrop);

        // Mejora de contraste final en V (igual que antes)
        MejorarContrasteV(cropMasked);

        ResultadoPieza pieza;
        pieza.valida = true;
        pieza.id = id_counter++;
        pieza.boundingBox = bbox;
        pieza.centroide = mc;
        pieza.area = area;
        pieza.circularidad = circularity;
        pieza.imagenRecortada = cropMasked;
        pieza.mascara = maskLocalCrop;

        resultados.push_back(pieza);
    }

    // =========================================================
    // 6. ORDENAR RESULTADOS POR ÁREA
    // =========================================================
    std::sort(resultados.begin(), resultados.end(),
        [](const ResultadoPieza& a, const ResultadoPieza& b) {
            return a.area > b.area;
        });

    // Reasignar IDs en orden (opcional, para que 1 sea el más grande)
    for (size_t i = 0; i < resultados.size(); ++i) {
        resultados[i].id = (int)(i + 1);
    }

    return resultados;
}

// --- IMPLEMENTACIONES AUXILIARES---
Mat Segmentacion::ImFillHoles(const Mat& mask)
{
    Mat mask_padded;
    copyMakeBorder(mask, mask_padded, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    Mat flood = mask_padded.clone();
    floodFill(flood, Point(0, 0), Scalar(255));

    Mat invertido;
    bitwise_not(flood, invertido);

    Mat filled_padded;
    bitwise_or(mask_padded, invertido, filled_padded);

    // Volver al tamaño original
    Rect roi(1, 1, mask.cols, mask.rows);
    return filled_padded(roi).clone();
}

Mat Segmentacion::ImClearBorder(const Mat& mask)
{
    Mat cleaned = mask.clone();
    vector<vector<Point>> contours;
    findContours(cleaned, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int h = mask.rows;
    int w = mask.cols;

    for (const auto& cnt : contours) {
        bool touches = false;
        for (const auto& pt : cnt) {
            if (pt.x <= 1 || pt.x >= w - 2 || pt.y <= 1 || pt.y >= h - 2) {
                touches = true;
                break;
            }
        }
        if (touches) {
            drawContours(cleaned, vector<vector<Point>>{cnt}, 0, Scalar(0), FILLED);
        }
    }
    return cleaned;
}

void Segmentacion::MejorarContrasteV(cv::Mat& imgBGR)
{
    if (imgBGR.empty()) return;

    // Pasar a HSV en float [0..1]
    cv::Mat imgFloat;
    imgBGR.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    cv::Mat hsv;
    cv::cvtColor(imgFloat, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> chans;
    cv::split(hsv, chans);
    cv::Mat& V = chans[2]; // float [0..1]

    // Recolectar TODOS los valores (incluye ceros, como MATLAB)
    std::vector<float> values;
    values.reserve(V.total());
    for (int r = 0; r < V.rows; ++r) {
        const float* p = V.ptr<float>(r);
        for (int c = 0; c < V.cols; ++c) {
            values.push_back(p[c]);
        }
    }

    if (values.empty()) return;

    float p1 = percentileMatlabLike(values, 1.f);
    float p95 = percentileMatlabLike(values, 95.f);

    float denom = p95 - p1;
    if (std::abs(denom) < 1e-6f) denom = 1e-6f;

    // Ecualización idéntica a MATLAB
    for (int r = 0; r < V.rows; ++r) {
        float* p = V.ptr<float>(r);
        for (int c = 0; c < V.cols; ++c) {
            float v = (p[c] - p1) / denom;
            p[c] = std::min(1.f, std::max(0.f, v));
        }
    }

    // Volver a BGR uint8
    cv::merge(chans, hsv);
    cv::cvtColor(hsv, imgFloat, cv::COLOR_HSV2BGR);
    imgFloat.convertTo(imgBGR, CV_8U, 255.0);
}

static float percentileMatlabLike(std::vector<float>& v, float p)
{
    if (v.empty()) return 0.f;

    float idx = p / 100.f * (v.size() - 1);
    size_t i0 = static_cast<size_t>(std::floor(idx));
    size_t i1 = static_cast<size_t>(std::ceil(idx));
    float frac = idx - i0;

    std::nth_element(v.begin(), v.begin() + i0, v.end());
    float v0 = v[i0];

    if (i1 == i0) return v0;

    std::nth_element(v.begin(), v.begin() + i1, v.end());
    float v1 = v[i1];

    return v0 * (1.f - frac) + v1 * frac;
}