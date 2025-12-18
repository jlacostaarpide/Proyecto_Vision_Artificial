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

    // 1. PRE-PROCESAMIENTO Y CONVERSIÓN
    // MATLAB: im2double(I) -> Rango [0, 1]
    // OpenCV: Convertimos a CV_32F y normalizamos a [0, 1] para usar los mismos umbrales
    Mat imgFloat;
    inputBGR.convertTo(imgFloat, CV_32F, 1.0 / 255.0);

    // MATLAB: rgb2hsv
    // OpenCV BGR2HSV con 32F: H[0..360], S[0..1], V[0..1]
    Mat hsv;
    cvtColor(imgFloat, hsv, COLOR_BGR2HSV);

    // Separar canales
    vector<Mat> channels;
    split(hsv, channels);
    Mat H = channels[0]; // Ojo: En OpenCV float H va de 0 a 360.
    Mat S = channels[1];
    Mat V = channels[2];

    // Normalizamos H a [0, 1] para coincidir EXACTAMENTE con el código de MATLAB
    H = H / 360.0f;

    // ---------------------------------------------------------
    // 2. ANÁLISIS CANAL S (Multi-level Otsu)
    // ---------------------------------------------------------

    // Gamma correction: S .^ 1.4
    Mat S_proc;
    pow(S, 1.4, S_proc);

    // Calculamos 2 umbrales (3 clases)
    vector<float> thresh_vals = CalcularMultilevelOtsu2(S_proc);
    float t1 = thresh_vals[0];
    float t2 = thresh_vals[1];

    // Clasificación y cálculo de ratio de clase media
    // Clase 2 (Media) está entre t1 y t2
    Mat mask_S_mid = (S_proc > t1) & (S_proc <= t2);

    int num_pixels = S_proc.rows * S_proc.cols;
    int count_mid = countNonZero(mask_S_mid);
    double ratio_mid = (double)count_mid / num_pixels;

    // Decisión por rangos (Lógica exacta de MATLAB)
    double umbral_inferior = 0.055;
    double umbral_superior = 0.17;
    bool use_lower_thresh = false;

    if (ratio_mid < umbral_inferior) {
        use_lower_thresh = true;
    }
    else if (ratio_mid > umbral_superior) {
        use_lower_thresh = false;
    }
    else {
        // Zona gris: Análisis de solidez del objeto más grande en la máscara media
        vector<vector<Point>> contours;
        findContours(mask_S_mid, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            // Buscar el contorno más grande
            size_t max_idx = 0;
            double max_area = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double a = contourArea(contours[i]);
                if (a > max_area) {
                    max_area = a;
                    max_idx = i;
                }
            }

            // Calcular solidez: Area / ConvexArea
            vector<Point> hull;
            convexHull(contours[max_idx], hull);
            double hull_area = contourArea(hull);

            double solidez_mid = 0.0;
            if (hull_area > 0) solidez_mid = max_area / hull_area;

            if (solidez_mid > 0.6) use_lower_thresh = true;
            else use_lower_thresh = false;
        }
        else {
            use_lower_thresh = false;
        }
    }

    float final_thresh = use_lower_thresh ? t1 : t2;
    Mat mask_S;
    threshold(S_proc, mask_S, final_thresh, 1.0, THRESH_BINARY); // mask_S es float 0.0/1.0
    mask_S.convertTo(mask_S, CV_8U, 255.0); // Convertir a 0/255

    // ---------------------------------------------------------
    // 3. ANÁLISIS CANAL H (Rescate Rosa/Morado)
    // ---------------------------------------------------------
    float min_sat_H_purple = 0.4f * final_thresh;
    float min_sat_H_pink = 1.0f * final_thresh;

    // Máscara Morada: H [0.58, 0.92] & S > min
    Mat mask_H_purple;
    inRange(H, 0.58, 0.92, mask_H_purple); // H range
    Mat mask_sat_p;
    threshold(S, mask_sat_p, min_sat_H_purple, 255, THRESH_BINARY);
    mask_sat_p.convertTo(mask_sat_p, CV_8U);
    bitwise_and(mask_H_purple, mask_sat_p, mask_H_purple);

    // Máscara Rosa: H [0.01, 0.065] & S > min
    Mat mask_H_pink;
    inRange(H, 0.01, 0.065, mask_H_pink);
    Mat mask_sat_pk;
    threshold(S, mask_sat_pk, min_sat_H_pink, 255, THRESH_BINARY);
    mask_sat_pk.convertTo(mask_sat_pk, CV_8U);
    bitwise_and(mask_H_pink, mask_sat_pk, mask_H_pink);

    Mat mask_H;
    bitwise_or(mask_H_purple, mask_H_pink, mask_H);

    // ---------------------------------------------------------
    // 4. ANÁLISIS CANAL V (Oscuros)
    // ---------------------------------------------------------
    // MATLAB: mask_V_dark(:) = 0; (Desactivado explícitamente)
    Mat mask_V_dark = Mat::zeros(mask_S.size(), CV_8U);

    // ---------------------------------------------------------
    // 5. FUSIÓN Y MORFOLOGÍA
    // ---------------------------------------------------------
    Mat mask_combined;
    bitwise_or(mask_S, mask_H, mask_combined);
    bitwise_or(mask_combined, mask_V_dark, mask_combined);

    // A. SUTURA INICIAL (Cerrar grietas)
    // se_suture = strel('disk', 3); -> OpenCV Size(7,7)
    Mat mask_sutured;
    Mat se_suture = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask_combined, mask_sutured, MORPH_CLOSE, se_suture);

    // B. RELLENO DE HUECOS
    Mat mask_filled = ImFillHoles(mask_sutured);

    // C. LIMPIEZA DE RUIDO (Apertura)
    // se_noise = strel('disk', 3); -> OpenCV Size(7,7)
    Mat mask_clean;
    Mat se_noise = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask_filled, mask_clean, MORPH_OPEN, se_noise);

    // D. ELIMINAR BORDES
    Mat mask_noborder = ImClearBorder(mask_clean);

    // E. OPERACIÓN DE CIERRE (MERGE)
    // radio_disco = 14; -> OpenCV Size(29, 29)
    Mat mask_merged;
    Mat se_merge = getStructuringElement(MORPH_ELLIPSE, Size(29, 29));
    morphologyEx(mask_noborder, mask_merged, MORPH_CLOSE, se_merge);

    // F. RELLENO FINAL
    mask_merged = ImFillHoles(mask_merged);

    // G. SUAVIZADO FINAL (Apertura)
    // se_smooth = strel('disk', 4); -> OpenCV Size(9, 9)
    Mat mask_final;
    Mat se_smooth = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(mask_merged, mask_final, MORPH_OPEN, se_smooth);

    // ---------------------------------------------------------
    // 6. EXTRACCIÓN DE RESULTADOS Y FILTRADO
    // ---------------------------------------------------------
    vector<vector<Point>> contoursFinal;
    findContours(mask_final, contoursFinal, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contoursFinal.empty()) return resultados;

    // Calcular área máxima para el filtro relativo
    double max_area = 0;
    for (const auto& cnt : contoursFinal) {
        double area = contourArea(cnt);
        if (area > max_area) max_area = area;
    }
    double umbral_area_relativo = 0.15 * max_area;

    int id_counter = 1;
    for (const auto& cnt : contoursFinal) {
        double area = contourArea(cnt);

        // Filtro de área
        if (area <= umbral_area_relativo) continue;

        // Calcular métricas
        Rect bbox = boundingRect(cnt);

        // Momentos para centroide
        Moments mu = moments(cnt);
        Point2f mc(mu.m10 / mu.m00, mu.m01 / mu.m00);

        // Circularidad
        double perimeter = arcLength(cnt, true);
        double circularity = 0.0;
        if (perimeter > 0) {
            circularity = (4 * CV_PI * area) / (perimeter * perimeter);
        }

        // --- GENERAR CROP FINAL ---
        // 1. Recortar imagen original
        Mat cropBGR = inputBGR(bbox).clone();

        // 2. Crear máscara local para el crop (para poner fondo negro)
        Mat maskLocal = Mat::zeros(cropBGR.size(), CV_8U);
        vector<Point> cntShifted = cnt; // Ajustar contorno a coordenadas del crop
        for (auto& pt : cntShifted) {
            pt.x -= bbox.x;
            pt.y -= bbox.y;
        }
        vector<vector<Point>> cntsShifted = { cntShifted };
        drawContours(maskLocal, cntsShifted, 0, Scalar(255), FILLED);

        // 3. Aplicar fondo negro
        Mat cropMasked;
        cropBGR.copyTo(cropMasked, maskLocal);

        // 4. Mejora de contraste en V (igual que MATLAB)
        MejorarContrasteV(cropMasked);

        // Guardar resultado
        ResultadoPieza pieza;
        pieza.valida = true;
        pieza.id = id_counter++;
        pieza.boundingBox = bbox;
        pieza.centroide = mc;
        pieza.area = area;
        pieza.circularidad = circularity;
        pieza.imagenRecortada = cropMasked;
        pieza.mascara = maskLocal; // Guardamos la máscara local por si acaso

        resultados.push_back(pieza);
    }

    return resultados;
}

// --- IMPLEMENTACIONES AUXILIARES ---

vector<float> Segmentacion::CalcularMultilevelOtsu2(const Mat& src)
{
    // Implementación rápida de Otsu para 2 umbrales (3 clases)
    // src debe ser float o uchar. Asumimos float [0,1] de la entrada.
    // Convertimos a 8-bit [0..255] para histograma rápido
    Mat src8;
    src.convertTo(src8, CV_8U, 255.0);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&src8, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

    // Normalizar histograma
    vector<double> p(256);
    double total_pixels = src8.total();
    for (int i = 0; i < 256; i++) p[i] = hist.at<float>(i) / total_pixels;

    // Tablas precalculadas (Probabilidad acumulada y Media acumulada)
    vector<double> omega(256, 0.0);
    vector<double> mu(256, 0.0);

    omega[0] = p[0];
    mu[0] = 0.0;
    for (int i = 1; i < 256; i++) {
        omega[i] = omega[i - 1] + p[i];
        mu[i] = mu[i - 1] + i * p[i];
    }
    double mu_t = mu[255]; // Media total

    double max_sigma_b = -1.0;
    int t1_opt = 0;
    int t2_opt = 0;

    // Búsqueda exhaustiva optimizada
    // t1 desde 0 hasta 253, t2 desde t1+1 hasta 254
    for (int t1 = 0; t1 < 254; t1++) {
        double w0 = omega[t1];
        double m0 = mu[t1] / w0; // Media clase 0

        for (int t2 = t1 + 1; t2 < 255; t2++) {
            double w1 = omega[t2] - omega[t1];
            double w2 = 1.0 - omega[t2];

            if (w1 <= 0 || w2 <= 0) continue; // Evitar división por cero

            double m1 = (mu[t2] - mu[t1]) / w1;
            double m2 = (mu_t - mu[t2]) / w2;

            // Varianza entre clases
            double sigma_b = w0 * (m0 - mu_t) * (m0 - mu_t) +
                w1 * (m1 - mu_t) * (m1 - mu_t) +
                w2 * (m2 - mu_t) * (m2 - mu_t);

            if (sigma_b > max_sigma_b) {
                max_sigma_b = sigma_b;
                t1_opt = t1;
                t2_opt = t2;
            }
        }
    }

    // Convertir de vuelta a rango float [0, 1]
    return { t1_opt / 255.0f, t2_opt / 255.0f };
}

Mat Segmentacion::ImFillHoles(const Mat& mask)
{
    // Rellenar huecos: Floodfill desde el fondo (0,0) sobre imagen invertida
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
    // Eliminar componentes que tocan el borde
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
    // Estirar el histograma del canal V (Percentiles 1% y 95%)
    // Solo en la zona que no es fondo negro
    if (imgBGR.empty()) return;

    Mat hsv;
    cvtColor(imgBGR, hsv, COLOR_BGR2HSV);
    vector<Mat> chans;
    split(hsv, chans);
    Mat V = chans[2]; // Rango 0..255

    // Recoger píxeles válidos (V > 0)
    vector<uchar> values;
    values.reserve(V.total());
    for (int i = 0; i < V.rows; ++i) {
        uchar* p = V.ptr<uchar>(i);
        for (int j = 0; j < V.cols; ++j) {
            if (p[j] > 0) values.push_back(p[j]);
        }
    }

    if (values.empty()) return;

    // Calcular percentiles
    size_t n = values.size();
    size_t idx1 = (size_t)(0.01 * n);
    size_t idx95 = (size_t)(0.95 * n);

    // Quickselect (nth_element) es más rápido que sort total
    std::nth_element(values.begin(), values.begin() + idx1, values.end());
    uchar p1 = values[idx1];

    std::nth_element(values.begin(), values.begin() + idx95, values.end());
    uchar p95 = values[idx95];

    // Estirar contraste (Normalize min-max)
    // V_eq = (V - p1) * (255 / (p95 - p1))
    // Usamos normalize de OpenCV con máscara
    Mat maskValid = (V > 0);
    // Para simplificar y robustez manual:
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