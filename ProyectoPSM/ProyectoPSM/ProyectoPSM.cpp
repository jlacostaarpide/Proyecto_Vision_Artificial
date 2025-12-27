#include "ProyectoPSM.h"
#include <filesystem>
#include <QFileDialog>
#include <QFile>
#include <QComboBox>
#include "Segmentacion.h"
#include <chrono>
#include <QMetaType>
#include <QDebug>
#include <QPainter>
#include <QApplication>

Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<QRectF>)
Q_DECLARE_METATYPE(std::vector<QImage>)

// Recibe la imagen y obtiene el bounding box y el thumbnail segmentado
void SegmentationWorker::process(std::shared_ptr<cv::Mat> snapshotPtr)
{
    std::vector<QRectF> outBoxes;
    std::vector<QImage> outThumbs;

    try {
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finishedResult(outBoxes, outThumbs);
            return;
        }

        // 1. Ejecutar Segmentación (Ya devuelve ordenado por área)
        std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(*snapshotPtr);

        double iw = static_cast<double>(snapshotPtr->cols);
        double ih = static_cast<double>(snapshotPtr->rows);

        // 2. Procesar las N mejores piezas (Mínimo las 2 primeras para la UI)
        //    Guardamos TODAS las cajas, pero solo 2 thumbnails para las etiquetas.
        int max_thumbs = 2;

        for (size_t i = 0; i < resultados.size(); i++) {
            const ResultadoPieza& pieza = resultados[i];

            // A. Guardar Caja Normalizada
            if (iw > 0 && ih > 0) {
                outBoxes.push_back(QRectF(
                    static_cast<double>(pieza.boundingBox.x) / iw,
                    static_cast<double>(pieza.boundingBox.y) / ih,
                    static_cast<double>(pieza.boundingBox.width) / iw,
                    static_cast<double>(pieza.boundingBox.height) / ih
                ));
            }

            // B. Generar Thumbnail (Solo para las primeras 'max_thumbs')
            if (outThumbs.size() < max_thumbs) {
                cv::Mat crop = pieza.imagenRecortada;
                if (!crop.empty()) {
                    const int thumbW = 160; // Ancho fijo para la UI
                    int srcW = crop.cols;
                    int srcH = crop.rows;
                    int thumbH = max(1, static_cast<int>((double)thumbW * srcH / max(1, srcW)));

                    cv::Mat thumb;
                    cv::resize(crop, thumb, cv::Size(thumbW, thumbH), 0, 0, cv::INTER_LINEAR);

                    cv::Mat thumb_rgb;
                    if (thumb.channels() == 3)
                        cv::cvtColor(thumb, thumb_rgb, cv::COLOR_BGR2RGB);
                    else
                        cv::cvtColor(thumb, thumb_rgb, cv::COLOR_GRAY2RGB);

                    QImage qthumb(reinterpret_cast<const uchar*>(thumb_rgb.data),
                        thumb_rgb.cols, thumb_rgb.rows,
                        static_cast<int>(thumb_rgb.step),
                        QImage::Format_RGB888);

                    outThumbs.push_back(qthumb.copy()); // .copy() es vital para detachar memoria
                }
            }
        }
    }
    catch (const std::exception& e) {
        qDebug() << "SegmentationWorker exception:" << e.what();
    }
    catch (...) {
        qDebug() << "SegmentationWorker unknown exception";
    }

    emit finishedResult(outBoxes, outThumbs);
}

ProyectoPSM::ProyectoPSM(QWidget* parent) : QMainWindow(parent)
{
    ui.setupUi(this);

    // Registros necesarios para pasar datos entre hilos
    qRegisterMetaType<shared_ptr<Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::vector<QRectF>>("std::vector<QRectF>");
    qRegisterMetaType<std::vector<QImage>>("std::vector<QImage>");

    if (!filesystem::exists("Database")) {
        filesystem::create_directory("Database");
    }

    NameList = NameHelper::GenerarNombres();

    // Inicialización
    LiveSegmentationEnabled = false;
    SegProcessing = false;
    segInFlight = 0;

    SegmentationIntervalMs = 40;
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);

    Camera = new CVideoAcquisition();

    // Configurar Worker
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);

    // CONEXIÓN UNIFICADA:
    connect(segWorker, &SegmentationWorker::finishedResult, this, &ProyectoPSM::UpdateSegmentationResults, Qt::QueuedConnection);
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);

    segThread->start();

    segTimer = new QTimer(this);
    segTimer->setInterval(SegmentationIntervalMs);
    connect(segTimer, &QTimer::timeout, this, &ProyectoPSM::onSegmentationTimer);
    segTimer->start();

    // --- Gestion de archivos ---
    ui.pbtnGuardar->setEnabled(false);
    ui.pbtnDescartar->setEnabled(false);

    connect(ui.pbtnAbrirImag, SIGNAL(clicked()), this, SLOT(SegmentarImagDisco()));
    connect(ui.comboSegMode, SIGNAL(activated(int)), this, SLOT(SegmentationMode(int)));
    connect(ui.pbtnDescartar, SIGNAL(clicked()), this, SLOT(ReturnTab()));
    connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));

    // --- CÁMARA ---
    if (Camera->CameraOK) {
        ui.pbtnEncender->setEnabled(true);
        ui.pbtnCapturar->setEnabled(false);
        ui.pbtnSegmentar->setEnabled(true);
        if(ui.pbtnClasificar) ui.pbtnClasificar->setEnabled(true);

        ImageIndex = 0;
        SavedImageIndex = 1;
        ui.boxImageNumber->setValue(SavedImageIndex);
        Camera->SetCameraAutoExposure();

        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        connect(ui.pbtnCapturar, SIGNAL(clicked()), this, SLOT(VisualizeImage()));
        connect(ui.pbtnSegmentar, SIGNAL(toggled(bool)), this, SLOT(EnableLiveSegmentation(bool)));
    }
    else {
        ui.lblImagen->setText("AVISO: Cámara no detectada.");
        ui.lblImagen->setAlignment(Qt::AlignCenter);
        ui.pbtnEncender->setEnabled(false);
        ui.pbtnCapturar->setEnabled(false);
        ui.pbtnSegmentar->setEnabled(false);
    }
}

ProyectoPSM::~ProyectoPSM()
{
    if (segThread) {
        segThread->quit();
        segThread->wait();
    }
}

void ProyectoPSM::EnableButtons(bool StartCapture)
{
    if (!StartCapture) {
        ui.pbtnEncender->setText("Encender");
        ui.pbtnCapturar->setEnabled(false);
        ui.lblImagen->clear();
    } else {
        ui.pbtnCapturar->setEnabled(StartCapture);
        ui.pbtnEncender->setText("Apagar");
    }
}

void ProyectoPSM::ShowImage()
{
    if (LastImage.empty()) return;

    cv::Mat rgb;
    if (LastImage.channels() == 3) cv::cvtColor(LastImage, rgb, cv::COLOR_BGR2RGB);
    else cv::cvtColor(LastImage, rgb, cv::COLOR_GRAY2RGB);

    QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());

    QSize labelSize = ui.lblImagen->size();
    QPixmap scaled = pix.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // --- DIBUJAR TODAS LAS CAJAS DETECTADAS ---
    if (!lastBoxesNormalized.empty()) {
        QPainter p(&scaled);
        QPen pen(Qt::green);
        pen.setWidthF(max(1.0, scaled.width() * 0.005));
        pen.setStyle(Qt::SolidLine);
        p.setPen(pen);
        p.setRenderHint(QPainter::Antialiasing, true);

        for (const auto& box : lastBoxesNormalized) {
            int x = static_cast<int>(box.x() * scaled.width());
            int y = static_cast<int>(box.y() * scaled.height());
            int w = static_cast<int>(box.width() * scaled.width());
            int h = static_cast<int>(box.height() * scaled.height());

            // Limites seguros
            x = max(0, min(x, scaled.width() - 1));
            y = max(0, min(y, scaled.height() - 1));
            if (x + w > scaled.width()) w = scaled.width() - x;
            if (y + h > scaled.height()) h = scaled.height() - y;

            if (w > 0 && h > 0) p.drawRect(x, y, w, h);
        }
    }

    ui.lblImagen->setPixmap(scaled);
    ui.lblImagen->setAlignment(Qt::AlignCenter);
}

// muestra la imagen capturada (para la base de datos) escalada al label
void ProyectoPSM::VisualizeImage()
{
    ui.tabWidget->setCurrentIndex(1);
    SavedImageIndex = ui.boxImageNumber->value();

    string texto = "";
    if (SavedImageIndex < 1 || SavedImageIndex > static_cast<int>(NameList.size())) {
        texto = "Índice de imagen fuera de rango. Valores válidos: 1 - " + to_string(NameList.size());
        ui.pbtnGuardar->setEnabled(false);
    } else {
        texto = "Guardar siguiente imagen como: " + (NameList[SavedImageIndex - 1]);
    }
    ui.txtImageName->setText(QString::fromStdString(texto));

    if (!LastImage.empty()) {
        CapturedImage = LastImage.clone();

        cv::Mat rgb;
        if (CapturedImage.channels() == 3) cv::cvtColor(CapturedImage, rgb, cv::COLOR_BGR2RGB);
        else cv::cvtColor(CapturedImage, rgb, cv::COLOR_GRAY2RGB);

        QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        QPixmap pix = QPixmap::fromImage(qimg.copy());
        if (ui.lblImagenCapturada) {
            ui.lblImagenCapturada->setPixmap(pix.scaled(ui.lblImagenCapturada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            ui.lblImagenCapturada->setAlignment(Qt::AlignCenter);
        } else {
            ui.lblImagenCapturada->setPixmap(pix);
        }
    }
}

void ProyectoPSM::SaveImage()
{
    if (!CapturedImage.empty()) {
        string Name = NameList[SavedImageIndex-1];
        string Path = "C:/Users/Lenovo/Desktop/imagenes/" + Name + ".jpg";
        qDebug("LLEGAAAAAAAAAAAAAAAA");
        cv::imwrite(Path, CapturedImage);
        ui.txtImageName->setText(QString::fromStdString("Image saved!"));
        SavedImageIndex++;
        ui.boxImageNumber->setValue(SavedImageIndex);
        ReturnTab();
    }
}

void ProyectoPSM::ReturnTab()
{
    ui.tabWidget->setCurrentIndex(0);
}

void ProyectoPSM::NewImage(cv::Mat Img)
{
    if (Img.empty()) return;
    LastImage = std::move(Img);
    ShowImage();
    ++ImageIndex;
}

void ProyectoPSM::onSegmentationTimer()
{
    if (!LiveSegmentationEnabled || SegProcessing.load() || LastImage.empty()) return;

    // Evitar saturación (max 1 frame en vuelo)
    int prev = segInFlight.fetch_add(1);
    if (prev >= 1) {
        segInFlight.fetch_sub(1);
        return;
    }

    SegProcessing = true;
    auto snapshotPtr = std::make_shared<cv::Mat>(LastImage.clone());
    emit requestSegmentation(snapshotPtr);
}

// slot que activa/desactiva la segmentación en vivo
void ProyectoPSM::EnableLiveSegmentation(bool enabled)
{
    LiveSegmentationEnabled = enabled;
    if (!enabled) {
        SegProcessing = false;
        lastBoxesNormalized.clear();
        segInFlight.store(0);
        // Limpiar labels
        if (ui.lblSeg1) ui.lblSeg1->clear();
        if (ui.lblSeg2) ui.lblSeg2->clear();
    }
}

void ProyectoPSM::UpdateSegmentationResults(const std::vector<QRectF>& boxes, const std::vector<QImage>& thumbnails)
{
    lastBoxesNormalized = boxes;
    ShowImage(); // Redibuja con las cajas verdes

    // Actualizar Thumbnails (Imagen 1 y Imagen 2)
    if (thumbnails.size() > 0 && ui.lblSeg1) {
        ui.lblSeg1->setPixmap(QPixmap::fromImage(thumbnails[0]).scaled(ui.lblSeg1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    else if (ui.lblSeg1) {
        ui.lblSeg1->clear();
    }

    if (thumbnails.size() > 1 && ui.lblSeg2) {
        ui.lblSeg2->setPixmap(QPixmap::fromImage(thumbnails[1]).scaled(ui.lblSeg2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    else if (ui.lblSeg2) {
        ui.lblSeg2->clear();
    }

    SegProcessing = false;
    segInFlight.fetch_sub(1);
}

//Slot para modo de segmentación offline
void ProyectoPSM::SegmentationMode(int index)
{
    if (index == 0) SegmentarImagCapturada();
    else if (index == 1) SegmentarImagDisco();
    ui.comboSegMode->setCurrentIndex(0);
}

//Segmentación de la imagen capturada
void ProyectoPSM::SegmentarImagCapturada()
{
    // 1. Preparamos el entorno
    ui.tabWidget->setCurrentIndex(2);

    // 2. Decidimos qué imagen usar (Capturada o Último Frame)
    if (!CapturedImage.empty()) {
        LastImage = CapturedImage.clone();
    }
    else if (!LastImage.empty()) {
        // Ya tenemos LastImage lista, no hacemos nada
    }
    else {
        ui.lblImagNoSegmentada->setText("No hay imagen capturada para segmentar.");
        ui.lblImagSegmentada->clear();
        return;
    }

    // 3. ¡Truco! Llamamos a la función de disco
    // Como ya hemos puesto la imagen en 'LastImage', la función de disco
    // la procesará y creará el COLLAGE con todas las piezas automáticamente.
    SegmentarImagDisco();
}

// Segmentacion de una imagen desde disco
void ProyectoPSM::SegmentarImagDisco() {
    ui.tabWidget->setCurrentIndex(2);

    // Si no venimos de capturar (modo 0), abrir dialogo
    if (sender() == ui.pbtnAbrirImag) OpenPicture();

    if (LastImage.empty()) return;

    // Mostrar original
    cv::Mat rgb;
    cv::cvtColor(LastImage, rgb, cv::COLOR_BGR2RGB);
    ui.lblImagNoSegmentada->setPixmap(QPixmap::fromImage(QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888)).scaled(ui.lblImagNoSegmentada->size(), Qt::KeepAspectRatio));

    ui.lblImagSegmentada->setText("Procesando...");
    QApplication::processEvents();

    std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(LastImage);

    if (resultados.empty()) {
        ui.lblImagSegmentada->setText("No se encontraron piezas.");
    }
    else {
        // CREAR COLLAGE: Unimos las imágenes horizontalmente (max 3)
        cv::Mat collage = resultados[0].imagenRecortada;

        // Intentar unir hasta 2 piezas más
        for (size_t i = 1; i < resultados.size() && i < 3; i++) {
            cv::Mat nextPiece = resultados[i].imagenRecortada;

            // Redimensionar nextPiece para que tenga la misma altura que el collage
            if (nextPiece.rows != collage.rows) {
                double scale = (double)collage.rows / nextPiece.rows;
                cv::resize(nextPiece, nextPiece, cv::Size(), scale, scale);
            }

            // Unir
            cv::hconcat(collage, nextPiece, collage);
        }

        // Mostrar collage
        cv::Mat rgbSeg;
        cv::cvtColor(collage, rgbSeg, cv::COLOR_BGR2RGB);
        QImage qimgSeg(rgbSeg.data, rgbSeg.cols, rgbSeg.rows, rgbSeg.step, QImage::Format_RGB888);
        ui.lblImagSegmentada->setPixmap(QPixmap::fromImage(qimgSeg).scaled(ui.lblImagSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void ProyectoPSM::OpenPicture()
{
    // 1. Abrir diálogo
    fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp);;All Files (*)"));
    if (fileName.isEmpty()) return;

    // 2. Comprobaciones de seguridad
    if (!QFile::exists(fileName)) {
        qDebug() << "SelectImage: fichero no existe segun QFile()";
        fileName.clear();
        return;
    }

    // 3. Lectura robusta (bytes crudos)
    QFile f(fileName);
    if (!f.open(QIODevice::ReadOnly)) {
        qDebug() << "SelectImage: no se puede abrir el fichero con QFile()";
        fileName.clear();
        return;
    }
    QByteArray fileData = f.readAll();
    f.close();

    // 4. Decodificar a Mat de OpenCV
    std::vector<uchar> vec(fileData.begin(), fileData.end());
    cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR);

    if (image.empty()) {
        qDebug() << "SelectImage: imdecode falló.";
        fileName.clear();
        return;
    }
    LastImage = image.clone();
}