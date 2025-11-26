#include "segmentacion.h"
#include <QDebug>
#include <opencv2/imgproc.hpp>

Segmentacion::Segmentacion(QObject* parent)
    : QObject(parent)
{
    // Registrar para poder usar cv::Mat en conexiones queued entre hilos
    qRegisterMetaType<cv::Mat>("cv::Mat");
}

Segmentacion::~Segmentacion() = default;

void Segmentacion::processImage(const cv::Mat &input)
{
    if (input.empty()) return;

    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    cv::Mat mask = createMask(gray);

    cv::Mat result;
    if (input.channels() == 3) {
        // copiar color original donde la máscara sea no cero
        input.copyTo(result, mask);
    } else {
        result = mask;
    }

    emit segmentedImage(result);
}

cv::Mat Segmentacion::createMask(const cv::Mat &gray)
{
    cv::Mat blurred, thresh;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);
    cv::adaptiveThreshold(blurred, thresh, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);

    return thresh;
}