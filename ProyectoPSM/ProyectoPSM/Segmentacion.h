#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Estructura que contiene toda la información de un LEGO detectado
struct ResultadoPieza {
    bool valida;             // True si es un objeto válido
    int id;                 // Identificador (1, 2...)
    cv::Rect boundingBox;    // Caja delimitadora en la imagen original
    cv::Point2f centroide;   // Centroide (x, y)
    double area;             // Área en píxeles
    double circularidad;     // Métrica de circularidad (0..1)

    cv::Mat imagenRecortada; // El crop de la pieza (fondo negro, alta calidad)
    cv::Mat mascara;         // La máscara binaria local de la pieza
};

class Segmentacion
{
public:
    // Método principal: Recibe la imagen BGR (High Res) y devuelve lista de piezas
    static std::vector<ResultadoPieza> Segmentar(const cv::Mat& inputBGR);

private:
    // --- MÉTODOS AUXILIARES INTERNOS (Traducción de MATLAB) ---

    // Implementación de imclearborder (elimina objetos que tocan el borde)
    static cv::Mat ImClearBorder(const cv::Mat& mask);

    // Implementación de imfill('holes')
    static cv::Mat ImFillHoles(const cv::Mat& mask);

    // Mejora de contraste del canal V para el crop final (Percentiles 1% - 95%)
    static void MejorarContrasteV(cv::Mat& imgBGR);
};