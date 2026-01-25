#ifndef CLASSIFICATIONVISUALIZER_H
#define CLASSIFICATIONVISUALIZER_H

#include <QObject>
#include <QImage>
#include <QVector>
#include <QStringList>
#include <QColor>
#include <opencv2/core.hpp>
#include <QPainter>
#include <QDebug>
#include <algorithm>
#include <opencv2/core.hpp> 
#include <cmath>

class ClassificationVisualizer : public QObject
{
    Q_OBJECT
public:
    explicit ClassificationVisualizer(QObject* parent = nullptr);

    QImage generateConfusionMatrix(const QVector<int>& trueLabels,
        const QVector<int>& predictedLabels,
        const QStringList& classNames,
        int imageSize = 600);

    QImage generateScatterPlot(const cv::Mat& features,
        const std::vector<int>& labels,
        const QStringList& classNames,
        int imageSize = 800,
        bool usePCA = true,
        QStringList axisLabels = QStringList());

private:
    QColor m_baseColor;
    QColor m_textColor;

    QColor interpolateColor(float ratio);
    QColor getClassColor(int classIdx, int totalClasses);
};

#endif // CLASSIFICATIONVISUALIZER_H