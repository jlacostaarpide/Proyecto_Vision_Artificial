#include "Segmentacion.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// --- MÉTODO PRINCIPAL ---
vector<ResultadoPieza> Segmentacion::Segmentar(const Mat& inputBGR)
{
    vector<ResultadoPieza> resultados;
    if (inputBGR.empty()) return resultados;

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
    // R_bal = R * (mean_G / mean_R)
    // B_bal = B * (mean_G / mean_B)
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
    // MATLAB: V_filt = imgaussfilt(V_raw, 120);
    Mat V_filt;
    GaussianBlur(V_raw, V_filt, Size(0, 0), 120);

    // MATLAB: V_corrected = V_raw ./ max(V_filt(:))
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
    // Esto imita: I_corrected = hsv2rgb(I_hsv_temp) en MATLAB
    channelsHSV[0] = H_raw;
    channelsHSV[1] = S_boosted;
    channelsHSV[2] = V_corrected;
    merge(channelsHSV, hsv_temp);

    Mat I_corrected;
    cvtColor(hsv_temp, I_corrected, COLOR_HSV2BGR);

    // --- Obtener canales finales para segmentación ---
    // MATLAB: I_hsv = rgb2hsv(I_corrected);
    Mat I_hsv_final;
    cvtColor(I_corrected, I_hsv_final, COLOR_BGR2HSV);
    split(I_hsv_final, channelsHSV);
    Mat S_final = channelsHSV[1]; // Este es el S que usaremos

    // =========================================================
    // 3. SEGMENTACIÓN (OTSU EN CANAL S)
    // =========================================================

    // Convertir S a 8-bit [0..255] para usar Otsu de OpenCV
    Mat S_8u;
    S_final.convertTo(S_8u, CV_8U, 255.0);

    Mat mask_S;
    // THRESH_OTSU calcula automáticamente el umbral óptimo
    threshold(S_8u, mask_S, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // =========================================================
    // 4. MORFOLOGÍA
    // =========================================================

    // A. Sutura inicial (imclose disk 3 -> Size 7x7)
    Mat se_suture = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    Mat mask_morph;
    morphologyEx(mask_S, mask_morph, MORPH_CLOSE, se_suture);

    // B. Relleno de huecos (imfill)
    mask_morph = ImFillHoles(mask_morph);

    // C. Limpieza de ruido (imopen disk 3 -> Size 7x7)
    Mat se_noise = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask_morph, mask_morph, MORPH_OPEN, se_noise);

    // D. Eliminar bordes (imclearborder)
    mask_morph = ImClearBorder(mask_morph);

    // E. Cierre grande (imclose disk 14 -> Size 29x29)
    Mat se_merge = getStructuringElement(MORPH_ELLIPSE, Size(29, 29));
    morphologyEx(mask_morph, mask_morph, MORPH_CLOSE, se_merge);

    // F. Relleno final
    mask_morph = ImFillHoles(mask_morph);

    // G. Suavizado final (imopen disk 4 -> Size 9x9)
    Mat se_smooth = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    Mat mask_final;
    morphologyEx(mask_morph, mask_final, MORPH_OPEN, se_smooth);

    // =========================================================
    // 5. EXTRACCIÓN Y FILTRADO
    // =========================================================

    vector<vector<Point>> contours;
    findContours(mask_final, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return resultados;

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
    double umbral_saturacion = 0.30; // 0.30 sobre 1.0

    int id_counter = 1;

    for (const auto& cnt : contours) {
        double area = contourArea(cnt);

        // 1. Filtro Área
        if (area <= umbral_area_rel || area <= umbral_area_abs) continue;

        Rect bbox = boundingRect(cnt);

        // 2. Filtro Aspect Ratio
        double w = (double)bbox.width;
        double h = (double)bbox.height;
        double ratio = std::max(w, h) / std::min(w, h); // max(width, height) / min(width, height)

        if (ratio > umbral_ratio_max) {
            // Descartado por forma alargada
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
            // Descartado por baja saturación
            continue;
        }

        // --- OBJETO VÁLIDO ---

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
    Mat flood = mask.clone();
    floodFill(flood, Point(0, 0), Scalar(255));
    Mat invertido;
    bitwise_not(flood, invertido);
    Mat filled;
    bitwise_or(mask, invertido, filled);
    return filled;
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

void Segmentacion::MejorarContrasteV(Mat& imgBGR)
{
    if (imgBGR.empty()) return;
    Mat hsv;
    cvtColor(imgBGR, hsv, COLOR_BGR2HSV);
    vector<Mat> chans;
    split(hsv, chans);
    Mat V = chans[2];

    vector<uchar> values;
    values.reserve(V.total());
    for (int i = 0; i < V.rows; ++i) {
        uchar* p = V.ptr<uchar>(i);
        for (int j = 0; j < V.cols; ++j) {
            if (p[j] > 0) values.push_back(p[j]);
        }
    }

    if (values.empty()) return;

    size_t n = values.size();
    size_t idx1 = (size_t)(0.01 * n);
    size_t idx95 = (size_t)(0.95 * n);

    std::nth_element(values.begin(), values.begin() + idx1, values.end());
    uchar p1 = values[idx1];

    std::nth_element(values.begin(), values.begin() + idx95, values.end());
    uchar p95 = values[idx95];

    Mat maskValid = (V > 0);
    float scale = (p95 > p1) ? 255.0f / (p95 - p1) : 1.0f;

    for (int i = 0; i < V.rows; ++i) {
        uchar* p = V.ptr<uchar>(i);
        for (int j = 0; j < V.cols; ++j) {
            if (p[j] > 0) {
                float val = (float)p[j];
                val = (val - p1) * scale;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                p[j] = (uchar)val;
            }
        }
    }
    merge(chans, hsv);
    cvtColor(hsv, imgBGR, COLOR_HSV2BGR);
}