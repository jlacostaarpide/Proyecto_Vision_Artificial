#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>

class Segmentacion : public QObject
{
    Q_OBJECT
public:
    explicit Segmentacion(QObject* parent = nullptr);
    ~Segmentacion();

public slots:
    void processImage(const cv::Mat &input);

signals:
    void segmentedImage(const cv::Mat &result);

private:
    cv::Mat createMask(const cv::Mat &gray);
};