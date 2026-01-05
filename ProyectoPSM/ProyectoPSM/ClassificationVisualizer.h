#ifndef CLASSIFICATIONVISUALIZER_H
#define CLASSIFICATIONVISUALIZER_H

#include <QObject>
#include <QImage>
#include <QVector>
#include <QStringList>
#include <QColor>

class ClassificationVisualizer : public QObject
{
    Q_OBJECT
public:
    explicit ClassificationVisualizer(QObject* parent = nullptr);

    /**
     * @brief Genera una imagen de la matriz de confusión.
     * @param trueLabels Vector con las etiquetas reales (ground truth).
     * @param predictedLabels Vector con las etiquetas predichas por el modelo.
     * @param classNames Lista con los nombres de las clases (en orden de índice 0, 1, 2...).
     * @param imageSize Tamaño cuadrado de la imagen de salida (por defecto 600x600).
     * @return QImage con el gráfico renderizado.
     */
    QImage generateConfusionMatrix(const QVector<int>& trueLabels,
        const QVector<int>& predictedLabels,
        const QStringList& classNames,
        int imageSize = 600);

    // TODO: Aquí añadiremos generateScatterPlot() en el futuro

private:
    // Configuración de estilo
    QColor m_baseColor;     // Color base para el heatmap (ej: azul)
    QColor m_textColor;     // Color del texto

    // Métodos auxiliares
    QColor interpolateColor(float ratio);
};

#endif // CLASSIFICATIONVISUALIZER_H