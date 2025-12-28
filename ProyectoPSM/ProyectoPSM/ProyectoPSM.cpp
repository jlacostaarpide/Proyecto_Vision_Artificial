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

// Necesario para pasar datos entre hilos
Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<QRectF>)
Q_DECLARE_METATYPE(std::vector<QImage>)


// Segmentación en Segundo Plano
void SegmentationWorker::process(std::shared_ptr<cv::Mat> snapshotPtr)
{
    std::vector<QRectF> outBoxes;
    std::vector<QImage> outThumbs;

    try {
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finishedResult(outBoxes, outThumbs);
            return;
        }

        // 1. Ejecutar Segmentación
        std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(*snapshotPtr);

        double iw = static_cast<double>(snapshotPtr->cols);
        double ih = static_cast<double>(snapshotPtr->rows);

        // 2. Procesar las N mejores piezas (Ahora 3 para la UI)
        int max_thumbs = 3;

        for (size_t i = 0; i < resultados.size(); i++) {
            const ResultadoPieza& pieza = resultados[i];

            // A. Guardar Caja Normalizada para pintar recuadro
            if (iw > 0 && ih > 0) {
                outBoxes.push_back(QRectF(
                    static_cast<double>(pieza.boundingBox.x) / iw,
                    static_cast<double>(pieza.boundingBox.y) / ih,
                    static_cast<double>(pieza.boundingBox.width) / iw,
                    static_cast<double>(pieza.boundingBox.height) / ih
                ));
            }

            // B. Generar Thumbnail
            if (outThumbs.size() < max_thumbs) {
                cv::Mat crop = pieza.imagenRecortada;
                if (!crop.empty()) {
                    // Escalamos a 200px para la nueva UI
                    const int thumbW = 200;
                    int srcW = crop.cols;
                    int srcH = crop.rows;
                    int thumbH = max(1, static_cast<int>((double)thumbW * srcH / max(1, srcW)));

                    cv::Mat thumb;
                    cv::resize(crop, thumb, cv::Size(thumbW, thumbH), 0, 0, cv::INTER_LINEAR);

                    cv::Mat thumb_rgb;
                    if (thumb.channels() == 3) cv::cvtColor(thumb, thumb_rgb, cv::COLOR_BGR2RGB);
                    else cv::cvtColor(thumb, thumb_rgb, cv::COLOR_GRAY2RGB);

                    QImage qthumb(reinterpret_cast<const uchar*>(thumb_rgb.data),
                        thumb_rgb.cols, thumb_rgb.rows, static_cast<int>(thumb_rgb.step),
                        QImage::Format_RGB888);

                    outThumbs.push_back(qthumb.copy());
                }
            }
        }
    }
    catch (...) {
        qDebug() << "Excepcion en SegmentationWorker";
    }

    emit finishedResult(outBoxes, outThumbs);
}


// Clase Principal
ProyectoPSM::ProyectoPSM(QWidget* parent) : QMainWindow(parent)
{
    ui.setupUi(this);

    qRegisterMetaType<shared_ptr<Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::vector<QRectF>>("std::vector<QRectF>");
    qRegisterMetaType<std::vector<QImage>>("std::vector<QImage>");

    if (!filesystem::exists("Database")) filesystem::create_directory("Database");

    NameList = NameHelper::GenerarNombres();
    LiveSegmentationEnabled = false;
    SegProcessing = false;
    segInFlight = 0;
    SegmentationIntervalMs = 40;
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);

    // 1. Inicializar Cámara
    Camera = new CVideoAcquisition();

    // 2. Configurar Worker
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);
    connect(segWorker, &SegmentationWorker::finishedResult, this, &ProyectoPSM::UpdateSegmentationResults, Qt::QueuedConnection);
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);
    segThread->start();

    // 3. Timers
    segTimer = new QTimer(this);
    segTimer->setInterval(SegmentationIntervalMs);
    connect(segTimer, &QTimer::timeout, this, &ProyectoPSM::onSegmentationTimer);
    segTimer->start();

    // Watchdog Timer
    statusTimer = new QTimer(this);
    statusTimer->setInterval(2000);
    connect(statusTimer, &QTimer::timeout, this, &ProyectoPSM::CheckCameraStatus);
    statusTimer->start();

    // 4. Conexiones UI
    connect(ui.pbtnEncender, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
    connect(ui.chkLiveSeg, SIGNAL(toggled(bool)), this, SLOT(EnableLiveSegmentation(bool)));
    connect(ui.btnCapturarAnalizar, SIGNAL(clicked()), this, SLOT(CapturarYAnalizar()));

    // Botón Reconectar
    connect(ui.btnReconectar, SIGNAL(clicked()), this, SLOT(ReconectarCamara()));

    connect(ui.btnCargarDisco, SIGNAL(clicked()), this, SLOT(CargarImagenDisco()));
    connect(ui.btnRecalcSeg, SIGNAL(clicked()), this, SLOT(RecalcularSegmentacion()));
    connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));
    connect(ui.btnGuardarComo, SIGNAL(clicked()), this, SLOT(SaveImageAs()));
    connect(ui.boxImageNumber, SIGNAL(valueChanged(int)), this, SLOT(UpdateFileNameLabel()));

    ui.pbtnGuardar->setEnabled(false);

    // 5. Configurar estado inicial
    bool camOk = (Camera && Camera->CameraOK);
    SetCameraStatusUI(camOk);
    if (camOk) {
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        Camera->SetCameraAutoExposure();
    }

    ImageIndex = 0;
    SavedImageIndex = 1;
    ui.boxImageNumber->setValue(SavedImageIndex);
    UpdateFileNameLabel();
}

ProyectoPSM::~ProyectoPSM()
{
    if (segThread) {
        segThread->quit();
        segThread->wait();
    }
    if (Camera) {
        delete Camera;
    }
}

// --- LÓGICA DE RECONEXIÓN ---

void ProyectoPSM::ReconectarCamara()
{
    // Desactivar botón para evitar pulsaciones múltiples
    ui.btnReconectar->setEnabled(false);
    ui.lblStatusCamara->setText("Estado: Conectando...");
    ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: orange;");
    QApplication::processEvents();

    // 1. Destruir objeto antiguo si existe
    if (Camera) {
        // Desconectar señales viejas
        disconnect(Camera, 0, 0, 0);

        // Importante: Asegurar que el hilo de captura ha muerto antes de borrar
        delete Camera;
        Camera = nullptr;
    }

    // 2. Crear nuevo objeto
    try {
        Camera = new CVideoAcquisition();
    }
    catch (...) {
        Camera = nullptr;
    }

    // 3. Verificar éxito
    bool success = (Camera && Camera->CameraOK);

    // Configurar interfaz según resultado
    SetCameraStatusUI(success);

    if (success) {
        // Reconectar señales al nuevo objeto
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        Camera->SetCameraAutoExposure();
    }
    else {
        // Si falla, volver a habilitar el botón para reintentar
        ui.btnReconectar->setEnabled(true);
    }
}

void ProyectoPSM::SetCameraStatusUI(bool isConnected)
{
    if (isConnected) {
        ui.lblStatusCamara->setText("Estado: Listo");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: green;");

        ui.pbtnEncender->setEnabled(true);
        ui.pbtnEncender->setChecked(false);
        ui.pbtnEncender->setText("Encender Cámara");
        ui.btnReconectar->setEnabled(true);
    }
    else {
        ui.lblStatusCamara->setText("Estado: Desconectado");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: red;");

        ui.pbtnEncender->setEnabled(false);
        ui.pbtnEncender->setChecked(false);
        ui.pbtnEncender->setText("No Disponible");

        ui.btnReconectar->setEnabled(true);
        ui.btnCapturarAnalizar->setEnabled(false);
    }
}

void ProyectoPSM::CheckCameraStatus()
{
    // Watchdog: Si detectamos que CameraOK pasó a false inesperadamente
    if (Camera) {
        if (!Camera->CameraOK && ui.pbtnEncender->isEnabled()) {
            // Forzar apagado UI
            if (ui.pbtnEncender->isChecked()) {
                ui.pbtnEncender->setChecked(false);
            }
            SetCameraStatusUI(false);
        }
    }
}

void ProyectoPSM::EnableButtons(bool StartCapture)
{
    if (StartCapture) {
        // Se ha encendido
        ui.pbtnEncender->setText("Apagar Cámara");
        ui.btnCapturarAnalizar->setEnabled(true);
        ui.lblStatusCamara->setText("Estado: Capturando");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: blue;");
        ui.btnReconectar->setEnabled(false);
    }
    else {
        // Se ha apagado
        ui.pbtnEncender->setText("Encender Cámara");
        ui.btnCapturarAnalizar->setEnabled(false);

        if (Camera && Camera->CameraOK) {
            ui.lblStatusCamara->setText("Estado: Listo");
            ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: green;");
            ui.btnReconectar->setEnabled(true);
        }
        else {
            SetCameraStatusUI(false);
        }

        ui.lblVideoLive->clear();
        ui.lblVideoLive->setText("Cámara Pausada");
    }
}

void ProyectoPSM::NewImage(cv::Mat Img)
{
    if (Img.empty()) return;
    LastImage = std::move(Img);
    ShowImage(); // Muestra en lblVideoLive
    ++ImageIndex;
}

void ProyectoPSM::ShowImage()
{
    if (LastImage.empty()) return;

    // Convertir para mostrar en Qt
    cv::Mat rgb;
    if (LastImage.channels() == 3) cv::cvtColor(LastImage, rgb, cv::COLOR_BGR2RGB);
    else cv::cvtColor(LastImage, rgb, cv::COLOR_GRAY2RGB);

    QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(qimg.copy());

    // Escalar al label de Video
    QSize labelSize = ui.lblVideoLive->size();
    if (labelSize.width() < 10) labelSize = QSize(640, 480); // Protección inicio

    QPixmap scaled = pix.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // DIBUJAR CAJAS VERDES (Segmentación en vivo)
    if (LiveSegmentationEnabled && !lastBoxesNormalized.empty()) {
        QPainter p(&scaled);
        QPen pen(Qt::green);
        pen.setWidth(3);
        p.setPen(pen);

        QFont font = p.font();
        font.setPixelSize(std::max<double>(12, scaled.height() / 25));
        font.setBold(true);
        p.setFont(font);

        for (size_t i = 0; i < lastBoxesNormalized.size(); ++i) {
            const auto& boxNorm = lastBoxesNormalized[i];
            int x = static_cast<int>(boxNorm.x() * scaled.width());
            int y = static_cast<int>(boxNorm.y() * scaled.height());
            int w = static_cast<int>(boxNorm.width() * scaled.width());
            int h = static_cast<int>(boxNorm.height() * scaled.height());

            p.drawRect(x, y, w, h);

            QString text = QString::number(i + 1);
            int textY = y - 5;
            if (textY < font.pixelSize()) textY = y + font.pixelSize() + 5;
            p.drawText(x, textY, text);
        }
    }

    ui.lblVideoLive->setPixmap(scaled);
}

void ProyectoPSM::onSegmentationTimer()
{
    if (!LiveSegmentationEnabled || SegProcessing.load() || LastImage.empty()) return;

    int prev = segInFlight.fetch_add(1);
    if (prev >= 1) {
        segInFlight.fetch_sub(1);
        return;
    }
    SegProcessing = true;
    auto snapshotPtr = std::make_shared<cv::Mat>(LastImage.clone());
    emit requestSegmentation(snapshotPtr);
}

void ProyectoPSM::EnableLiveSegmentation(bool enabled)
{
    LiveSegmentationEnabled = enabled;
    if (!enabled) {
        SegProcessing = false;
        lastBoxesNormalized.clear();
        segInFlight.store(0);
        // Limpiar thumbnails en vivo
        if (ui.lblLiveThumb1) ui.lblLiveThumb1->clear();
        if (ui.lblLiveThumb2) ui.lblLiveThumb2->clear();
        if (ui.lblLiveThumb3) ui.lblLiveThumb3->clear();
    }
}

void ProyectoPSM::UpdateSegmentationResults(const std::vector<QRectF>& boxes, const std::vector<QImage>& thumbnails)
{
    lastBoxesNormalized = boxes;
    // NewImage se encarga de llamar a ShowImage para pintar las cajas

    // Actualizar Thumbnails en Vivo (derecha)
    QLabel* labels[] = { ui.lblLiveThumb1, ui.lblLiveThumb2, ui.lblLiveThumb3 };
    int numLabels = 3;

    for (int i = 0; i < numLabels; i++) {
        if (i < thumbnails.size()) {
            labels[i]->setPixmap(QPixmap::fromImage(thumbnails[i]).scaled(labels[i]->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        else {
            labels[i]->clear();
            labels[i]->setText(QString("Pieza %1").arg(i + 1));
        }
    }

    SegProcessing = false;
    segInFlight.fetch_sub(1);
}


// CAPTURA Y ANÁLISIS OFFLINE (BRIDGE)
void ProyectoPSM::CapturarYAnalizar()
{
    // 1. Verificar imagen
    if (LastImage.empty()) return;

    // 2. Congelar imagen actual
    CapturedImage = LastImage.clone();

    // 3. Parar segmentación en vivo para ahorrar recursos
    //if (ui.chkLiveSeg->isChecked()) {
    //    ui.chkLiveSeg->setChecked(false);
    //}

    // 4. Cambiar a la pestaña de Análisis
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // 5. Procesar automáticamente la imagen capturada
    ProcesarImagenOffline(CapturedImage);
}

void ProyectoPSM::CargarImagenDisco()
{
    // 1. Abrir diálogo
    QString fileName = QFileDialog::getOpenFileName(this, tr("Abrir Imagen"), "", tr("Images (*.png *.jpg *.bmp);;All (*)"));
    if (fileName.isEmpty()) return;

    // 2. Cargar con QFile (robusto)
    QFile f(fileName);
    if (!f.open(QIODevice::ReadOnly)) return;
    QByteArray fileData = f.readAll();
    f.close();

    std::vector<uchar> vec(fileData.begin(), fileData.end());
    cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR);

    if (image.empty()) return;

    // 3. Guardar como imagen capturada y procesar
    CapturedImage = image.clone();

    // Asegurar que estamos en la pestaña correcta
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    ProcesarImagenOffline(CapturedImage);
}

void ProyectoPSM::RecalcularSegmentacion()
{
    // Re-ejecutar sobre la imagen que ya tenemos
    if (!CapturedImage.empty()) {
        ProcesarImagenOffline(CapturedImage);
    }
}

// Lógica central de Análisis
void ProyectoPSM::ProcesarImagenOffline(const cv::Mat& img)
{
    if (img.empty()) return;

    // A. Asegurar que estamos en la sub-pestaña de Resultados
    ui.tabWidgetAnalysis->setCurrentWidget(ui.subTabResultados);
    ui.lblOfflineMain->setText("Procesando...");
    QApplication::processEvents();

    // B. Ejecutar Segmentación
    std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(img);

    // C. Preparar la Imagen Central (Original + Cajas Verdes)
    cv::Mat displayImg = img.clone();

    // Dibujamos los recuadros sobre la imagen completa
    for (const auto& res : resultados) {
        cv::rectangle(displayImg, res.boundingBox, cv::Scalar(0, 255, 0), 3);
        cv::putText(displayImg, std::to_string(res.id),
            cv::Point(res.boundingBox.x, res.boundingBox.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }

    // Mostrar en el visor central (lblOfflineMain)
    cv::Mat rgbDisplay;
    if (displayImg.channels() == 3) cv::cvtColor(displayImg, rgbDisplay, cv::COLOR_BGR2RGB);
    else cv::cvtColor(displayImg, rgbDisplay, cv::COLOR_GRAY2RGB);

    QImage qDisplay(rgbDisplay.data, rgbDisplay.cols, rgbDisplay.rows, rgbDisplay.step, QImage::Format_RGB888);
    ui.lblOfflineMain->setPixmap(QPixmap::fromImage(qDisplay).scaled(ui.lblOfflineMain->size(), Qt::KeepAspectRatio));

    // D. Rellenar las Miniaturas Laterales (Recortes)
    if (resultados.empty()) {
        ui.lblOfflineThumb1->clear();
        ui.lblOfflineThumb2->clear();
        ui.lblOfflineThumb3->clear();
        ui.pbtnGuardar->setEnabled(false);
    }
    else {
        ui.pbtnGuardar->setEnabled(true);

        QLabel* thumbs[] = { ui.lblOfflineThumb1, ui.lblOfflineThumb2, ui.lblOfflineThumb3 };

        for (int i = 0; i < 3; i++) {
            if (i < resultados.size()) {
                cv::Mat p = resultados[i].imagenRecortada;
                if (!p.empty()) {
                    cv::Mat pRGB;
                    if (p.channels() == 3) cv::cvtColor(p, pRGB, cv::COLOR_BGR2RGB);
                    else cv::cvtColor(p, pRGB, cv::COLOR_GRAY2RGB);

                    QImage qp(pRGB.data, pRGB.cols, pRGB.rows, pRGB.step, QImage::Format_RGB888);
                    thumbs[i]->setPixmap(QPixmap::fromImage(qp).scaled(thumbs[i]->size(), Qt::KeepAspectRatio));
                }
            }
            else {
                thumbs[i]->clear();
                thumbs[i]->setText("---");
            }
        }
    }
}

void ProyectoPSM::UpdateFileNameLabel()
{
    int idx = ui.boxImageNumber->value();
    if (idx > 0 && idx <= NameList.size()) {
        QString name = QString::fromStdString(NameList[idx - 1]);
        ui.lblImageName->setText("Nombre: " + name + ".jpg");
    }
    else {
        ui.lblImageName->setText("Nombre: [Fuera de Rango]");
    }
}

void ProyectoPSM::SaveImageAs()
{
    if (CapturedImage.empty()) return;

    // 1. Abrir diálogo obligando a extensión .jpg
    QString fileName = QFileDialog::getSaveFileName(this, tr("Guardar Imagen"), "", tr("JPEG Image (*.jpg);;All Files (*)"));
    if (fileName.isEmpty()) return;

    // 2. AUTO-CORRECCIÓN: Si el usuario no escribió ".jpg", se lo ponemos nosotros
    if (!fileName.endsWith(".jpg", Qt::CaseInsensitive) && !fileName.endsWith(".jpeg", Qt::CaseInsensitive)) {
        fileName += ".jpg";
    }

    // 3. GUARDADO ROBUSTO (Buffer OpenCV -> QFile Qt)
    // Esto evita problemas con tildes, ñ o rutas largas en Windows que hacen fallar a imwrite
    std::vector<uchar> buffer;
    try {
        // Codificar a JPG en memoria (Calidad 95)
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        cv::imencode(".jpg", CapturedImage, buffer, params);

        // Escribir a disco usando Qt (que maneja bien las rutas)
        QFile file(fileName);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            file.close();

            // 4. Feedback Visual (Cambiar texto del botón)
            ui.btnGuardarComo->setText("¡Guardado!");
            ui.btnGuardarComo->setEnabled(false);
            QTimer::singleShot(1500, [this]() {
                ui.btnGuardarComo->setText("Guardar Como...");
                ui.btnGuardarComo->setEnabled(true);
                });
        }
        else {
            qDebug() << "Error: No se pudo escribir en el archivo (Permisos?).";
        }
    }
    catch (...) {
        qDebug() << "Error al codificar la imagen.";
    }
}

void ProyectoPSM::SaveImage()
{
    if (!CapturedImage.empty()) {
        // Generar nombre basado en DB
        int idx = ui.boxImageNumber->value();
        std::string Name = (idx <= NameList.size() && idx > 0) ? NameList[idx - 1] : "captura_extra_" + std::to_string(idx);

        // Crear ruta segura
        std::string Path = "Database/" + Name + ".jpg";

        cv::imwrite(Path, CapturedImage);

        // Feedback visual
        ui.pbtnGuardar->setText("¡Guardado!");
        ui.pbtnGuardar->setEnabled(false);
        QTimer::singleShot(1000, [this]() {
            ui.pbtnGuardar->setText("Guardar");
            ui.pbtnGuardar->setEnabled(true);
            });

        // Avanzar índice y actualizar etiqueta
        if (idx < 9999) {
            ui.boxImageNumber->setValue(idx + 1);
            // El setValue disparará el signal valueChanged que llamará a UpdateFileNameLabel
        }
    }
}