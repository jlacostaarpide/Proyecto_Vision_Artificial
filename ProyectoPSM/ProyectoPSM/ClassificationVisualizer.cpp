#include "classificationvisualizer.h"
#include <QPainter>
#include <QDebug>
#include <algorithm> // para std::max_element

ClassificationVisualizer::ClassificationVisualizer(QObject* parent)
    : QObject(parent)
{
    // Configuración por defecto: Azul oscuro para intensidad máxima
    m_baseColor = QColor(41, 128, 185);
    m_textColor = Qt::black;
}

QImage ClassificationVisualizer::generateConfusionMatrix(const QVector<int>& trueLabels,
    const QVector<int>& predictedLabels,
    const QStringList& classNames,
    int imageSize)
{
    // Aseguramos que siempre haya 12 clases si la lista tiene 12 nombres
    int numClasses = classNames.size();
    if (numClasses == 0) return QImage();

    // 1. Matriz de Conteo
    QVector<QVector<int>> matrix(numClasses, QVector<int>(numClasses, 0));
    int maxCount = 0;

    for (int i = 0; i < trueLabels.size(); ++i) {
        int t = trueLabels[i];
        int p = predictedLabels[i];
        // Proteccion de rangos
        if (t >= 0 && t < numClasses && p >= 0 && p < numClasses) {
            matrix[t][p]++;
            if (matrix[t][p] > maxCount) maxCount = matrix[t][p];
        }
    }

    // 2. Configuración de Dibujo
    QImage image(imageSize, imageSize, QImage::Format_ARGB32);
    image.fill(Qt::white);

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    // Márgenes más amplios para los textos
    int leftMargin = 80;
    int bottomMargin = 80;
    int topMargin = 40;   // Título eje X
    int rightMargin = 20;

    // Área útil para la cuadrícula
    int gridW = imageSize - leftMargin - rightMargin;
    int gridH = imageSize - topMargin - bottomMargin;

    // Tamaño de cada celda
    double cellW = (double)gridW / numClasses;
    double cellH = (double)gridH / numClasses;

    // Fuentes
    QFont numberFont = painter.font();
    // Ajuste dinámico de fuente para que quepa en la celda
    numberFont.setPixelSize(std::max(10, (int)(cellH * 0.4)));
    numberFont.setBold(true);

    QFont labelFont = painter.font();
    labelFont.setPixelSize(std::max(10, (int)(cellH * 0.25))); // Fuente más pequeña para ejes

    // 3. Dibujar Celdas
    for (int t = 0; t < numClasses; ++t) {      // Filas (True)
        for (int p = 0; p < numClasses; ++p) {  // Columnas (Predicted)

            int count = matrix[t][p];
            float ratio = (maxCount > 0) ? (float)count / maxCount : 0.0f;

            // Coordenadas precisas
            QRectF rect(leftMargin + p * cellW, topMargin + t * cellH, cellW, cellH);

            // Color
            QColor cellColor = interpolateColor(ratio);
            painter.fillRect(rect, cellColor);

            // Borde celda
            painter.setPen(QPen(Qt::lightGray, 1));
            painter.drawRect(rect);

            // Número
            if (count > 0) {
                painter.setFont(numberFont);
                painter.setPen(ratio > 0.5 ? Qt::white : Qt::black);
                painter.drawText(rect, Qt::AlignCenter, QString::number(count));
            }
        }
    }

    // 4. Dibujar Ejes y Etiquetas
    painter.setFont(labelFont);
    painter.setPen(Qt::black);

    for (int i = 0; i < numClasses; ++i) {
        QString name = classNames[i];

        // EJE Y (True Labels) - Izquierda
        // Centrado verticalmente respecto a la celda
        QRectF yRect(0, topMargin + i * cellH, leftMargin - 5, cellH);
        painter.drawText(yRect, Qt::AlignRight | Qt::AlignVCenter, name);

        // EJE X (Predicted Labels) - Abajo
        // Centrado horizontalmente respecto a la celda
        QRectF xRect(leftMargin + i * cellW, imageSize - bottomMargin + 5, cellW, 30);

        // Guardar estado para rotar texto si es necesario (opcional, aquí lo pongo recto centrado)
        painter.drawText(xRect, Qt::AlignHCenter | Qt::AlignTop, name);
    }

    // 5. Títulos de los Ejes (Grandes)
    QFont titleFont = painter.font();
    titleFont.setPixelSize(16);
    titleFont.setBold(true);
    painter.setFont(titleFont);

    // Título Superior (Eje X)
    painter.drawText(QRect(leftMargin, 0, gridW, topMargin), Qt::AlignCenter, "Clase Predicha");

    // Título Lateral (Eje Y) - Rotado
    painter.save();
    painter.translate(20, topMargin + gridH / 2);
    painter.rotate(-90);
    painter.drawText(QRect(-gridH / 2, -20, gridH, 20), Qt::AlignCenter, "Clase Real");
    painter.restore();

    return image;
}

// Interpola entre Blanco (0) y el Color Base (1)
QColor ClassificationVisualizer::interpolateColor(float ratio)
{
    // Empezamos en blanco (255, 255, 255) y vamos hacia m_baseColor
    int r = 255 + (m_baseColor.red() - 255) * ratio;
    int g = 255 + (m_baseColor.green() - 255) * ratio;
    int b = 255 + (m_baseColor.blue() - 255) * ratio;
    return QColor(r, g, b);
}