#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Estructura para extraer los pasos intermedios
struct DebugInfo {
    cv::Mat I_orig;
    cv::Mat I_norm;
    cv::Mat H, S, V;
    cv::Mat S_proc;      // Canal S usado para Otsu
    cv::Mat mask_otsu;   // 1. Binaria Original
    cv::Mat mask_fill;   // 2. Relleno
    cv::Mat mask_clean;  // 3. Limpieza Ruido
    cv::Mat mask_border; // 4. Sin Bordes
    cv::Mat mask_close;  // 5. Cierre
    cv::Mat mask_final;  // 6. Final
};

// Estructura que contiene toda la información de un LEGO detectado
struct ResultadoPieza {
    int id;
    bool valida;
    cv::Rect boundingBox;
    cv::Point2f centroide;
    double area;
    double circularidad;
    cv::Mat imagenRecortada;
    cv::Mat mascara;
};

class Segmentacion
{
public:
    // Método principal: Recibe la imagen BGR (High Res) y devuelve lista de piezas
    static std::vector<ResultadoPieza> Segmentar(const cv::Mat& inputBGR, DebugInfo* debug = nullptr);

private:
	// Métodos auxiliares
    static cv::Mat ImFillHoles(const cv::Mat& mask);
    static cv::Mat ImClearBorder(const cv::Mat& mask);
    static void MejorarContrasteV(cv::Mat& imgBGR);
};