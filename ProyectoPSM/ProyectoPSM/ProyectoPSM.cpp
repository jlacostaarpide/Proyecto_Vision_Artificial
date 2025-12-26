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

// Recibe la imagen y obtiene el bounding box y el thumbnail segmentado
void SegmentationWorker::process(std::shared_ptr<cv::Mat> snapshotPtr)
{
    QRectF normalizedBox(0,0,0,0);
    QImage qthumb;

    try {
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finishedBox(normalizedBox);
            emit finishedThumbnail(qthumb);
            return;
        }

        std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(*snapshotPtr);

        if (!resultados.empty()) {
            const ResultadoPieza& pieza = resultados[0];

            double iw = static_cast<double>(snapshotPtr->cols);
            double ih = static_cast<double>(snapshotPtr->rows);

            if (iw > 0 && ih > 0) {
                normalizedBox = QRectF(
                    static_cast<double>(pieza.boundingBox.x) / iw,
                    static_cast<double>(pieza.boundingBox.y) / ih,
                    static_cast<double>(pieza.boundingBox.width) / iw,
                    static_cast<double>(pieza.boundingBox.height) / ih
                );
            }

            cv::Mat crop = pieza.imagenRecortada;
            if (!crop.empty()) {
                const int thumbW = 160;
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

                qthumb = QImage(reinterpret_cast<const uchar*>(thumb_rgb.data),
                                thumb_rgb.cols, thumb_rgb.rows,
                                static_cast<int>(thumb_rgb.step),
                                QImage::Format_RGB888).copy();
            }
        }
    }
    catch (const std::exception& e) {
        qDebug() << "SegmentationWorker exception:" << e.what();
    }
    catch (...) {
        qDebug() << "SegmentationWorker unknown exception";
    }

    emit finishedBox(normalizedBox);
    emit finishedThumbnail(qthumb);
}

ProyectoPSM::ProyectoPSM(QWidget *parent): QMainWindow(parent)
{
    ui.setupUi(this);

    // registrar tipo para queued connections
    qRegisterMetaType<shared_ptr<Mat>>("std::shared_ptr<cv::Mat>");

	if (!filesystem::exists("Database")) {
		filesystem::create_directory("Database");
	}

	NameList = NameHelper::GenerarNombres();


	qDebug() << "Número de nombres generados: " << static_cast<int>(NameList.size());
	qDebug() << "Primer nombre: " << QString::fromStdString(NameList[0]);
	qDebug() << "Segundo nombre: " << QString::fromStdString(NameList[1]);
	qDebug() << "Último nombre: " << QString::fromStdString(NameList.back());

    Camera = new CVideoAcquisition();
	// inicializar flags de segmentación
	LiveSegmentationEnabled = false;
	SegProcessing = false;
    segInFlight = 0;
    segThumbNext = 0;

	SegmentationIntervalMs = 40; //cada cuanto hacer segmentación (ms)
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);

    // crear worker y thread para segmentación 
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);
    connect(segWorker, &SegmentationWorker::finishedBox, this, &ProyectoPSM::UpdateSegmentationBox, Qt::QueuedConnection);
    connect(segWorker, &SegmentationWorker::finishedThumbnail, this, &ProyectoPSM::EnqueueSegThumbnail, Qt::QueuedConnection);
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);
    segThread->start();

    segTimer = new QTimer(this);
    segTimer->setInterval(SegmentationIntervalMs);
    connect(segTimer, &QTimer::timeout, this, &ProyectoPSM::onSegmentationTimer);
    segTimer->start();

    if (Camera->CameraOK) {
        ui.pbtnEncender->setEnabled(true);
        ui.pbtnCapturar->setEnabled(false);
        ui.pbtnGuardar->setEnabled(true);
        ui.pbtnDescartar->setEnabled(true);

        ImageIndex = 0;
        SavedImageIndex = 1;
        ui.boxImageNumber->setValue(SavedImageIndex);
        Camera->SetCameraAutoExposure();

        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        connect(ui.pbtnCapturar, SIGNAL(clicked()), this, SLOT(VisualizeImage()));
        connect(ui.pbtnDescartar, SIGNAL(clicked()), this, SLOT(ReturnTab()));
        connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));
        connect(ui.pbtnSegmentar, SIGNAL(toggled(bool)), this, SLOT(EnableLiveSegmentation(bool)));
        connect(ui.pbtnAbrirImag, SIGNAL(clicked()), this, SLOT(SegmentarImagDisco()));
        connect(ui.comboSegMode, SIGNAL(activated(int)), this, SLOT(SegmentationMode(int)));
    }
    else {
        ui.lblImagen->setText("ERROR: No se ha podido establecer comunicación con la cámara.");
        ui.pbtnCapturar->setEnabled(false);
    }
}

ProyectoPSM::~ProyectoPSM()
{
    if (segThread) {
        segThread->quit();
        segThread->wait();
        segThread = nullptr;
        segWorker = nullptr;
    }
    if (segTimer) {
        segTimer->stop();
        segTimer = nullptr;
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

// Pinta la imagen actual escalada al tamaño del label y dibuja bbox normalizado
void ProyectoPSM::ShowImage()
{
    if (LastImage.empty() || !ui.pbtnCapturar->isEnabled()) return;

    // Crear QImage desde BGR (usamos BGR -> RGB para QImage)
    cv::Mat rgb;
    if (LastImage.channels() == 3) {
        cv::cvtColor(LastImage, rgb, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(LastImage, rgb, cv::COLOR_GRAY2RGB);
    }

    QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());

    // Escalar el pixmap al tamaño del label manteniendo la proporción
    QSize labelSize = ui.lblImagen->size();
    QPixmap scaled = pix.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Dibujar bbox en coordenadas del pixmap escalado si existe
    if (!lastBoxNormalized.isNull() && lastBoxNormalized.width() > 0 && lastBoxNormalized.height() > 0) {
        QPainter p(&scaled);
        QPen pen(Qt::green);
        pen.setWidthF(max(1.0, scaled.width() * 0.005));
        pen.setStyle(Qt::SolidLine);
        p.setPen(pen);
        p.setRenderHint(QPainter::Antialiasing, true);

        int x = static_cast<int>(lastBoxNormalized.x() * scaled.width());
        int y = static_cast<int>(lastBoxNormalized.y() * scaled.height());
        int w = static_cast<int>(lastBoxNormalized.width() * scaled.width());
        int h = static_cast<int>(lastBoxNormalized.height() * scaled.height());

        x = max(0, min(x, scaled.width()-1));
        y = max(0, min(y, scaled.height()-1));
        if (w <= 0) w = 1;
        if (h <= 0) h = 1;
        if (x + w > scaled.width()) w = scaled.width() - x;
        if (y + h > scaled.height()) h = scaled.height() - y;

        p.drawRect(x, y, w, h);
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

// timer slot: tomar un frame cada intervalo y enviarlo al worker si está libre
void ProyectoPSM::onSegmentationTimer()
{
    if (!LiveSegmentationEnabled) return;
    if (SegProcessing.load()) return;
    if (LastImage.empty()) return;

    int prev = segInFlight.fetch_add(1, std::memory_order_relaxed);
    if (prev >= maxSegInFlight) {
        segInFlight.fetch_sub(1, std::memory_order_relaxed);
        return;
    }

    SegProcessing = true;

    // Enviamos la imagen completa (puedes cambiar a resize si quieres reducir trabajo)
    auto snapshotPtr = std::make_shared<cv::Mat>(LastImage.clone());
    emit requestSegmentation(snapshotPtr);
}

// slot que activa/desactiva la segmentación en vivo
void ProyectoPSM::EnableLiveSegmentation(bool enabled)
{
    LiveSegmentationEnabled = enabled;
    if (enabled) {
        ui.lblImagNoSegmentada->setText(QString::fromStdString("Segmentacion en vivo ACTIVADA"));
    } else {
        ui.lblImagSegmentada->clear();
        SegProcessing = false;
        lastBoxNormalized = QRectF();
        segInFlight.store(0);
    }
}

// slot llamado por el worker con el bbox normalizado (ejecuta en GUI)
void ProyectoPSM::UpdateSegmentationBox(const QRectF &box)
{
    lastBoxNormalized = box;
    ShowImage();
    SegProcessing = false;
}

// recibe miniaturas desde el worker y los muestra en dos QLabel (circular)
void ProyectoPSM::EnqueueSegThumbnail(const QImage &thumb)
{
    if (thumb.isNull()) {
        segInFlight.fetch_sub(1, std::memory_order_relaxed);
        return;
    }

    int idx = segThumbNext % maxSegInFlight;
    QPixmap pix = QPixmap::fromImage(thumb);

    if (idx == 0) {
        if (ui.lblSeg1) ui.lblSeg1->setPixmap(pix.scaled(ui.lblSeg1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else if (idx == 1) {
        if (ui.lblSeg2) ui.lblSeg2->setPixmap(pix.scaled(ui.lblSeg2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    segThumbNext = (segThumbNext + 1) % maxSegInFlight;
    segInFlight.fetch_sub(1, std::memory_order_relaxed);
}

//Slot para modo de segmentación offline
void ProyectoPSM::SegmentationMode(int index)
{
    if (index == 0) {
        // reseteo al índice 0 para permitir volver a seleccionar
        SegmentarImagCapturada();
        if (ui.comboSegMode) ui.comboSegMode->setCurrentIndex(0);

    }
    else if (index == 1) {
        SegmentarImagDisco();
        if (ui.comboSegMode) ui.comboSegMode->setCurrentIndex(0);
    }
}
//Segmentación de la imagen capturada
void ProyectoPSM::SegmentarImagCapturada()
{
    ui.tabWidget->setCurrentIndex(2);

    // Preferir la imagen capturada; si no existe, usar el último frame recibido
    cv::Mat img;
    if (!CapturedImage.empty()) {
        img = CapturedImage.clone();
    }
    else if (!LastImage.empty()) {
        img = LastImage.clone();
    }
    else {
        ui.lblImagNoSegmentada->setText(QString::fromStdString("No hay imagen capturada."));
        ui.lblImagSegmentada->clear();
        return;
    }

    // Mostrar la imagen original escalada en el label
    cv::Mat rgb;
    if (img.channels() == 3) cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    else cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);

    QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());
    ui.lblImagNoSegmentada->setPixmap(pix.scaled(ui.lblImagNoSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui.lblImagNoSegmentada->setAlignment(Qt::AlignCenter);
   

    // Ejecutar segmentación (sin hilo, como en disco)
    std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(img);
    if (resultados.empty()) {
        ui.lblImagSegmentada->setText(QString::fromStdString("No se pudo segmentar la imagen."));
        return;
    }

    cv::Mat segmented = resultados[0].imagenRecortada;
    if (segmented.empty()) {
        ui.lblImagSegmentada->setText(QString::fromStdString("No se pudo generar la región segmentada."));
        return;
    }

    // Mostrar la región segmentada escalada al label
    cv::Mat rgbSeg;
    if (segmented.channels() == 3) cv::cvtColor(segmented, rgbSeg, cv::COLOR_BGR2RGB);
    else cv::cvtColor(segmented, rgbSeg, cv::COLOR_GRAY2RGB);

    QImage qimgSeg(reinterpret_cast<const uchar*>(rgbSeg.data), rgbSeg.cols, rgbSeg.rows, static_cast<int>(rgbSeg.step), QImage::Format_RGB888);
    QPixmap pixSeg = QPixmap::fromImage(qimgSeg.copy());
    ui.lblImagSegmentada->setPixmap(pixSeg.scaled(ui.lblImagSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui.lblImagSegmentada->setAlignment(Qt::AlignCenter);
}


// Segmentacion de una imagen desde disco
void ProyectoPSM::SegmentarImagDisco() {
    ui.tabWidget->setCurrentIndex(2);

    OpenPicture();
    if (fileName.isEmpty()) return;
    if (LastImage.empty()) {
        ui.lblImagNoSegmentada->setText("Error al cargar la imagen.");
        return;
    }

    cv::Mat img = LastImage.clone();

    // Mostrar original escalada
    {
        cv::Mat rgb;
        if (img.channels() == 3) cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        else cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);

        QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        QPixmap pix = QPixmap::fromImage(qimg.copy());
        ui.lblImagNoSegmentada->setPixmap(pix.scaled(ui.lblImagNoSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        ui.lblImagNoSegmentada->setAlignment(Qt::AlignCenter);
    }

    // limpiar posible resultado anterior y mostrar indicador
    ui.lblImagSegmentada->clear();
    ui.lblImagSegmentada->setText(tr("Procesando..."));
    QApplication::processEvents(); // permite mostrar el texto antes de la operación costosa

    // realizar siempre la segmentación para la imagen abierta desde disco
    std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(img);
    if (resultados.empty()) {
        ui.lblImagSegmentada->setText(tr("No se pudo segmentar la imagen."));
    }
    else {
        cv::Mat segmented = resultados[0].imagenRecortada;
        if (segmented.empty()) {
            ui.lblImagSegmentada->setText(tr("No se pudo generar la región segmentada."));
        }
        else {
            cv::Mat rgbSeg;
            if (segmented.channels() == 3) cv::cvtColor(segmented, rgbSeg, cv::COLOR_BGR2RGB);
            else cv::cvtColor(segmented, rgbSeg, cv::COLOR_GRAY2RGB);

            QImage qimgSeg(reinterpret_cast<const uchar*>(rgbSeg.data), rgbSeg.cols, rgbSeg.rows, static_cast<int>(rgbSeg.step), QImage::Format_RGB888);
            QPixmap pixSeg = QPixmap::fromImage(qimgSeg.copy());
            ui.lblImagSegmentada->setPixmap(pixSeg.scaled(ui.lblImagSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            ui.lblImagSegmentada->setAlignment(Qt::AlignCenter);
        }
    }
}

void ProyectoPSM::OpenPicture()
{
    fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp);;All Files (*)"));
    if (fileName.isEmpty()) return;

    if (!QFile::exists(fileName)) {
        qDebug() << "SelectImage: fichero no existe segun QFile()";
        fileName.clear();
        return;
    }

    QFile f(fileName);
    if (!f.open(QIODevice::ReadOnly)) {
        qDebug() << "SelectImage: no se puede abrir el fichero con QFile()";
        fileName.clear();
        return;
    }
    QByteArray fileData = f.readAll();
    f.close();

    std::vector<uchar> vec(fileData.begin(), fileData.end());
    cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR);
    if (image.empty()) {
        qDebug() << "SelectImage: imdecode falló.";
        fileName.clear();
        return;
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());
    if (pix.isNull()) {
        qDebug() << "SelectImage: QPixmap nulo tras la conversion.";
        fileName.clear();
        return;
    }

    ui.lblImagNoSegmentada->setPixmap(pix.scaled(ui.lblImagNoSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui.lblImagNoSegmentada->setAlignment(Qt::AlignCenter);

    LastImage = image.clone();
}
