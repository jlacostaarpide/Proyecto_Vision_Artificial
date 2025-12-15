#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>

class Segmentacion : public QObject
{
    Q_OBJECT
public:
    explicit Segmentacion(QObject* parent = nullptr);
    ~Segmentacion();
    // Recibe imagen BGR y devuelve imagen BGR con bounding boxes y etiquetas dibujadas
    static cv::Mat Segment(const cv::Mat& src);

    // Devuelve una máscara binaria (CV_8U, 0/255) para la imagen de entrada (BGR).
    // Hace pipeline de segmentación y devuelve la máscara,
    // sin dibujar anotaciones ni calcular bounding boxes.
    static cv::Mat SegmentMask(const cv::Mat &src);

public slots:
    void processImage(const cv::Mat &input);

signals:
    void segmentedImage(const cv::Mat &result);

private:
    cv::Mat createMask(const cv::Mat &gray);
};