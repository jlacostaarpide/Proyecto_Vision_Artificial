#include "classificationvisualizer.h"
#include <QPainter>
#include <QDebug>
#include <algorithm>
#include <opencv2/core.hpp> 
#include <cmath>

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

QImage ClassificationVisualizer::generateScatterPlot(const cv::Mat& features,
    const std::vector<int>& labels,
    const QStringList& classNames,
    int imageSize)
{
    if (features.empty() || features.rows != labels.size()) {
        qWarning() << "Visualizer: Datos vacíos o desajuste features/labels.";
        return QImage();
    }

    // 1. CALCULAR PCA (Reducción a 2 dimensiones)
    // Usamos OpenCV para proyectar los datos N-dimensionales a 2D
    int nComponents = 2;
    cv::PCA pca(features, cv::Mat(), cv::PCA::DATA_AS_ROW, nComponents);
    cv::Mat projection = pca.project(features); // Matriz de Nx2

    // 2. BUSCAR MÍNIMOS Y MÁXIMOS (Para escalar a la imagen)
    double minX = 1e9, maxX = -1e9;
    double minY = 1e9, maxY = -1e9;

    for (int i = 0; i < projection.rows; ++i) {
        float x = projection.at<float>(i, 0);
        float y = projection.at<float>(i, 1);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    // Añadir un margen del 10% para que los puntos no toquen el borde
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    if (rangeX < 1e-6) rangeX = 1.0; // Evitar división por cero
    if (rangeY < 1e-6) rangeY = 1.0;

    minX -= rangeX * 0.1; maxX += rangeX * 0.1;
    minY -= rangeY * 0.1; maxY += rangeY * 0.1;
    rangeX = maxX - minX;
    rangeY = maxY - minY;

    // 3. PREPARAR LIENZO
    QImage image(imageSize, imageSize, QImage::Format_ARGB32);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    // Espacio para la leyenda a la derecha (200px)
    int legendWidth = 200;
    int plotWidth = imageSize - legendWidth;
    int plotHeight = imageSize;

    // Dibujar Ejes (Cruz central)
    painter.setPen(QPen(Qt::lightGray, 1, Qt::DashLine));
    // Eje X (Y=0)
    if (minY < 0 && maxY > 0) {
        int y0 = plotHeight - (int)((0.0 - minY) / rangeY * plotHeight);
        painter.drawLine(0, y0, plotWidth, y0);
    }
    // Eje Y (X=0)
    if (minX < 0 && maxX > 0) {
        int x0 = (int)((0.0 - minX) / rangeX * plotWidth);
        painter.drawLine(x0, 0, x0, plotHeight);
    }

    // 4. DIBUJAR PUNTOS
    int pointRadius = 4; // Tamaño del punto
    painter.setPen(Qt::NoPen);

    for (int i = 0; i < projection.rows; ++i) {
        float xVal = projection.at<float>(i, 0);
        float yVal = projection.at<float>(i, 1);
        int label = labels[i];

        // Mapear coordenadas al píxel
        int px = (int)((xVal - minX) / rangeX * plotWidth);
        int py = plotHeight - (int)((yVal - minY) / rangeY * plotHeight); // Y invertido en pantalla

        // Obtener color según la clase (0..11)
        QColor color = getClassColor(label, classNames.size());

        // Hacer el punto semi-transparente para ver solapamientos
        color.setAlpha(180);
        painter.setBrush(color);
        painter.drawEllipse(QPoint(px, py), pointRadius, pointRadius);
    }

    // 5. DIBUJAR LEYENDA (Derecha)
    // Fondo de la leyenda
    painter.fillRect(plotWidth, 0, legendWidth, plotHeight, QColor(245, 245, 245));
    painter.setPen(Qt::black);
    painter.drawLine(plotWidth, 0, plotWidth, plotHeight); // Separador

    QFont legendFont = painter.font();
    legendFont.setPixelSize(12);
    painter.setFont(legendFont);

    // Título Leyenda
    painter.drawText(QRect(plotWidth + 10, 10, legendWidth - 20, 20), Qt::AlignLeft, "Clases:");

    // Lista de clases
    int itemHeight = 20;
    int startY = 40;

    // Detectar qué clases están presentes para pintarlas
    std::vector<bool> present(classNames.size(), false);
    for (int l : labels) if (l >= 0 && l < (int)present.size()) present[l] = true;

    for (int i = 0; i < classNames.size(); ++i) {
        // Solo pintamos en la leyenda si la clase existe en los datos (o pintamos todas si prefieres)
        // Pintamos todas para mantener consistencia de colores

        QColor c = getClassColor(i, classNames.size());

        // Rectangulito de color
        QRect colorRect(plotWidth + 10, startY + i * itemHeight, 15, 10);
        painter.setBrush(c);
        painter.setPen(Qt::black);
        painter.drawRect(colorRect);

        // Nombre
        // Si no está presente, lo pintamos gris claro
        if (!present[i]) painter.setPen(Qt::gray);
        else painter.setPen(Qt::black);

        painter.drawText(QRect(plotWidth + 35, startY + i * itemHeight - 5, legendWidth - 40, 20),
            Qt::AlignLeft | Qt::AlignVCenter, classNames[i]);
    }

    // Título del Gráfico
    painter.setPen(Qt::black);
    QFont titleFont = painter.font();
    titleFont.setBold(true);
    titleFont.setPixelSize(14);
    painter.setFont(titleFont);
    painter.drawText(QRect(0, 0, plotWidth, 30), Qt::AlignCenter, "Analisis de Componentes Principales (PCA)");

    return image;
}

// Genera un color distinto para cada clase usando el espacio HSV
QColor ClassificationVisualizer::getClassColor(int classIdx, int totalClasses)
{
    if (totalClasses < 1) return Qt::black;
    // Repartir el matiz (hue) en el círculo cromático (0..359)
    // 0=Rojo, 120=Verde, 240=Azul...
    double hue = (classIdx * 360.0) / totalClasses;
    return QColor::fromHsvF(hue / 360.0, 0.85, 0.90); // Saturación alta, Valor alto
}
