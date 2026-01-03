#include "ProyectoPSM.h"
#include "SegmentarGuardar.h"
#include <filesystem>
#include <QFileDialog>
#include <QFile>
#include <QComboBox>
#include <chrono>
#include <QMetaType>
#include <QDebug>
#include <QPainter>
#include <QApplication>
#include <iostream>

#include <QMessageBox>
#include <QFileInfo>
#include <QRegularExpression>
#include <QDateTime>

namespace fs = std::filesystem;

// Necesario para pasar datos entre hilos
Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)
Q_DECLARE_METATYPE(std::vector<QRectF>)
Q_DECLARE_METATYPE(std::vector<QImage>)
Q_DECLARE_METATYPE(std::vector<cv::Mat>)

// Función auxiliar para convertir cv::Mat a QPixmap y ponerlo en un Label
void DisplayMat(QLabel* lbl, const cv::Mat& mat, bool isBinary = false) {
    if (mat.empty()) { lbl->clear(); return; }

    cv::Mat disp;
    if (isBinary || mat.type() == CV_8UC1) {
        // Si es gris/binaria, convertir a RGB para Qt
        cv::cvtColor(mat, disp, cv::COLOR_GRAY2RGB);
    }
    else {
        // Si es BGR, convertir a RGB
        cv::cvtColor(mat, disp, cv::COLOR_BGR2RGB);
    }

    QImage qimg(disp.data, disp.cols, disp.rows, disp.step, QImage::Format_RGB888);
    lbl->setPixmap(QPixmap::fromImage(qimg).scaled(lbl->size(), Qt::KeepAspectRatio));
}

void DrawHistogram(QLabel* lbl, const cv::Mat& src) {
    if (src.empty()) return;

    // Calcular histograma
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Calcular Otsu localmente para saber dónde pintar la línea
    cv::Mat dummy;
    double otsuThresh = cv::threshold(src, dummy, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Configurar lienzo
    int w = 500; int h = 350;
    int mX = 40;
    int mY = 30;

    cv::Mat histImg(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

    int plotHeight = h - 2 * mY;
    int plotWidth = w - 2 * mX;
    cv::normalize(hist, hist, 0, plotHeight, cv::NORM_MINMAX);

    // Dibujar Ejes (Marco Negro)
    cv::rectangle(histImg, cv::Point(mX, mY), cv::Point(w - mX, h - mY), cv::Scalar(0, 0, 0), 2);

    // Dibujar Gráfica (Línea Roja)
    for (int i = 1; i < histSize; i++) {
        // Mapear índice 'i' (0-255) a coordenadas X de la gráfica
        int x1 = mX + cvRound((i - 1) * ((double)plotWidth / 256));
        int x2 = mX + cvRound((i) * ((double)plotWidth / 256));

        // Mapear valor del histograma a coordenadas Y (invertido porque Y=0 es arriba)
        int y1 = h - mY - cvRound(hist.at<float>(i - 1));
        int y2 = h - mY - cvRound(hist.at<float>(i));

        cv::line(histImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Dibujar Línea de Otsu (Azul)
    int xTh = mX + cvRound(otsuThresh * ((double)plotWidth / 256));
    cv::line(histImg, cv::Point(xTh, mY), cv::Point(xTh, h - mY), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    // Texto con el valor
    std::string text = "T: " + std::to_string((int)otsuThresh);
    cv::putText(histImg, text, cv::Point(xTh + 5, mY + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 0, 0), 2);

    DisplayMat(lbl, histImg);
}

// Segmentación en Segundo Plano
void SegmentationWorker::process(std::shared_ptr<cv::Mat> snapshotPtr)
{
    std::vector<QRectF> outBoxes;
    std::vector<QImage> outThumbs;
    std::vector<cv::Mat> outCrops;

    try {
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finishedResult(outBoxes, outThumbs, outCrops);
            return;
        }

        // 1. Ejecutar Segmentación
        std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(*snapshotPtr, nullptr);

        double iw = static_cast<double>(snapshotPtr->cols);
        double ih = static_cast<double>(snapshotPtr->rows);

        // 2. Procesar las N mejores piezas (Ahora 3 para la UI)
        int max_thumbs = 3;

        for (size_t i = 0; i < resultados.size(); i++) {
            const ResultadoPieza& pieza = resultados[i];

            // Si el recorte es válido lo guardamos, si no, guardamos uno vacío para mantener índices
            if (!pieza.imagenRecortada.empty()) {
                outCrops.push_back(pieza.imagenRecortada);
            }
            else {
                outCrops.push_back(cv::Mat());
            }

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

    emit finishedResult(outBoxes, outThumbs, outCrops);
}

// Clasificación en Segundo Plano
void ClasificationWorker::process(std::vector<cv::Mat> crops, std::vector<QRectF> boxes) 
{
    std::vector<QString> outLabels;

    try {
        if (crops.empty()) {
            emit finishedResult(boxes, outLabels);
            return;
        }

        for (const auto& crop : crops) {
            if (crop.empty()) {
                outLabels.push_back("Error");
                continue;
            }

            // Clasificación
            QString labelText = "Desc.";
            std::string codigoPieza = "";
            bool svmExito = false;

            // A. SVM
            if (svmClf && svmClf->IsLoaded()) {
                int id = svmClf->Predict(crop);
                if (id > 0) {
                    codigoPieza = std::to_string(id);
                    if (id < 10) codigoPieza = "0" + codigoPieza;
                    svmExito = true;
                }
            }

            // B. Orientación
            if (svmExito) {
                bool orientExito = false;
                int yaw = 0;
                if (orientClf) {
                    try {
                        OrientationResult orr = orientClf->predict(crop, codigoPieza);
                        if (orr.ok) { yaw = orr.yaw; orientExito = true; }
                    }
                    catch (...) {}
                }

                if (orientExito) labelText = QString("Cód:%1 Orientación:%2").arg(QString::fromStdString(codigoPieza)).arg(yaw);
                else labelText = QString("Cód:%1").arg(QString::fromStdString(codigoPieza));
            }
            outLabels.push_back(labelText);
        }
    }
    catch (...) {
        qDebug() << "Excepcion en ClasificationWorker";
    }
    emit finishedResult(boxes, outLabels);
}

// Clase Principal
ProyectoPSM::ProyectoPSM(QWidget* parent) : QMainWindow(parent)
{
    ui.setupUi(this);

    svmClf_ = std::make_unique<Clasificador>();

    // Inicializar pestañas
    ui.tabWidget->setCurrentIndex(0);
    ui.tabWidgetAnalysis->setCurrentIndex(0);
    ui.tabWidgetDebug->setCurrentIndex(0);

    // Entrena si no hay modelo de clasificacion
    //maybeTrain();
    //runEvalAmarillas();
    //runEvalGlobal();

    qRegisterMetaType<shared_ptr<Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::vector<QRectF>>("std::vector<QRectF>");
    qRegisterMetaType<std::vector<QImage>>("std::vector<QImage>");
    qRegisterMetaType<std::vector<cv::Mat>>("std::vector<cv::Mat>");

    if (!filesystem::exists("Database")) filesystem::create_directory("Database");

    NameList = NameHelper::GenerarNombres();
    LiveSegmentationEnabled = false;
    SegProcessing = false;
    LiveClassificationEnabled = false;
    ClassProcessing = false;
    segInFlight = 0;
    SegmentationIntervalMs = 40;
    ClasificationIntervalMs = 150;
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);


    // CLASIFICACION ORIENTACION:
    // Ruta RELATIVA (corregida)
    orientTemplatesDir_ = "Templates";
    // Crea el clasificador con esa carpeta
    orientClf_ = std::make_unique<ClasificadorOrientacion>(orientTemplatesDir_.toStdString(), 128);
    orientTemplatesLoaded_ = false;
    EnsureOrientTemplatesLoaded();


    // 1. Inicializar Cámara
    Camera = new CVideoAcquisition();

    // 2. Configurar Segmentation Worker
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);
    connect(segWorker, &SegmentationWorker::finishedResult, this, &ProyectoPSM::UpdateSegmentationResults, Qt::QueuedConnection);
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);
    segThread->start();

	// Configurar Clasification Worker
    classWorker = new ClasificationWorker(svmClf_.get(), orientClf_.get());
    classThread = new QThread(this);
    classWorker->moveToThread(classThread);
    connect(classThread, &QThread::finished, classWorker, &QObject::deleteLater);
    connect(this, &ProyectoPSM::requestClassification, classWorker, &ClasificationWorker::process);
    connect(classWorker, &ClasificationWorker::finishedResult, this, &ProyectoPSM::UpdateClassificationResults);
    classThread->start();

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

    // 4. Conexiones UI Principales
    connect(ui.pbtnEncender, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
    connect(ui.chkLiveSeg, SIGNAL(toggled(bool)), this, SLOT(EnableLiveSegmentation(bool)));
    connect(ui.chkLiveClass, &QCheckBox::toggled, this, &ProyectoPSM::onCheckLiveClass);
    connect(ui.btnCapturarAnalizar, SIGNAL(clicked()), this, SLOT(CapturarYAnalizar()));

    // Botón Reconectar
    connect(ui.btnReconectar, SIGNAL(clicked()), this, SLOT(ReconectarCamara()));

    connect(ui.btnCargarDisco, SIGNAL(clicked()), this, SLOT(CargarImagenDisco()));
    connect(ui.btnRecalcSeg, SIGNAL(clicked()), this, SLOT(RecalcularSegmentacion()));
    connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));
    connect(ui.btnGuardarComo, SIGNAL(clicked()), this, SLOT(SaveImageAs()));
    connect(ui.boxImageNumber, SIGNAL(valueChanged(int)), this, SLOT(UpdateFileNameLabel()));
    connect(ui.btnRecalcClass, SIGNAL(clicked()), this, SLOT(ProcesarClasificacionOffline()));
    connect(ui.btnDB, SIGNAL(clicked()), this, SLOT(OnBatchSegmentar()));

    ui.pbtnGuardar->setEnabled(false);

    ui.chkLiveSeg->setEnabled(false);
    ui.chkLiveClass->setEnabled(false);
    ui.chkLiveSeg->setChecked(false);
    ui.chkLiveClass->setChecked(false);

    // 5. Configurar estado inicial Cámara
    bool camOk = (Camera && Camera->CameraOK);
    SetCameraStatusUI(camOk);
    if (camOk) {
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        Camera->SetCameraAutoExposure();
    }

    // 6. Conectar boton Clasificador Orientacion (Legacy):
    connect(ui.btnClasificarOrientacion, SIGNAL(clicked()), this, SLOT(AbrirYClasificarOrientacion()));

    // 7. CONEXIONES NUEVA PESTAÑA ENTRENAMIENTO
    // Botones de examinar (Browse)
    connect(ui.btnBrowseRaw, &QPushButton::clicked, this, &ProyectoPSM::onBrowseRaw);
    connect(ui.btnBrowseSeg, &QPushButton::clicked, this, &ProyectoPSM::onBrowseSeg);
    connect(ui.btnBrowseFeatures, &QPushButton::clicked, this, &ProyectoPSM::onBrowseFeatures);
    connect(ui.btnBrowseModel, &QPushButton::clicked, this, &ProyectoPSM::onBrowseModel);
    connect(ui.btnBrowseTest, &QPushButton::clicked, this, &ProyectoPSM::onBrowseTest);

    // Checkboxes (Saltar pasos)
    connect(ui.chkSkipSeg, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipSeg);
    connect(ui.chkSkipExtract, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipExtract);
    connect(ui.chkSkipTrain, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipTrain);
    connect(ui.chkSkipEval, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipEval);

    // Botón Iniciar Proceso
    connect(ui.btnStartTraining, &QPushButton::clicked, this, &ProyectoPSM::onStartTrainingClicked);


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
    if (classThread) {
        classThread->quit();
        classThread->wait();
    }
    if (Camera) {
        delete Camera;
    }
}

// Funciones de la pestaña de entrenamiento

void ProyectoPSM::onBrowseRaw() {
    QString dir = QFileDialog::getExistingDirectory(this, "Seleccionar Carpeta de Imágenes Raw", "Database");
    if (!dir.isEmpty()) ui.txtPathRaw->setText(dir);
}

void ProyectoPSM::onBrowseSeg() {
    QString dir = QFileDialog::getExistingDirectory(this, "Seleccionar Carpeta de Destino/Origen Segmentadas", "Database");
    if (!dir.isEmpty()) ui.txtPathSeg->setText(dir);
}

void ProyectoPSM::onBrowseFeatures() {
    QString file = QFileDialog::getSaveFileName(this, "Archivo de Características", "Database/features.yml", "YAML Files (*.yml *.yaml);;XML Files (*.xml)");
    if (!file.isEmpty()) ui.txtPathFeatures->setText(file);
}

void ProyectoPSM::onBrowseModel() {
    QString file = QFileDialog::getOpenFileName(this, "Archivo de Modelo SVM", "Database/modelM.yml", "YAML Files (*.yml *.yaml)");
    if (!file.isEmpty()) ui.txtPathModel->setText(file);
}

void ProyectoPSM::onBrowseTest() {
    QString dir = QFileDialog::getExistingDirectory(this, "Seleccionar Carpeta de Test", "Database");
    if (!dir.isEmpty()) ui.txtPathTest->setText(dir);
}

// Lógica visual de los Checkboxes
void ProyectoPSM::onCheckSkipSeg(bool checked) {
    ui.txtPathRaw->setEnabled(!checked);
    ui.btnBrowseRaw->setEnabled(!checked);
    // Cambiar placeholder o etiqueta para indicar que ahora cargamos desde segmentadas
    if (checked) ui.label_2->setText("Cargar Segmentadas:");
    else ui.label_2->setText("Salida Segmentadas:");
}

void ProyectoPSM::onCheckSkipExtract(bool checked) {
    // Si saltamos extracción, necesitamos cargar features, pero no necesitamos la carpeta de segmentadas
    // Esto depende de cómo quieras encadenarlo.
    // Por simplicidad visual:
    if (checked) ui.label_3->setText("Cargar Features (.xml):");
    else ui.label_3->setText("Guardar Features (.xml):");
}

void ProyectoPSM::onCheckSkipTrain(bool checked) {
    if (checked) ui.label_4->setText("Cargar Modelo (.yml):");
    else ui.label_4->setText("Guardar Modelo (.yml):");
}

void ProyectoPSM::onCheckSkipEval(bool checked) {
    ui.txtPathTest->setEnabled(!checked);
    ui.btnBrowseTest->setEnabled(!checked);
}

void ProyectoPSM::onStartTrainingClicked() {
    // 1. Configurar rutas desde la UI
    TrainingConfig config;
    config.rawFolder = ui.txtPathRaw->text();
    config.segFolder = ui.txtPathSeg->text();
    config.skipExtraction = ui.chkSkipExtract->isChecked();
    config.featuresFile = ui.txtPathFeatures->text();

    // Resetear UI
    ui.txtLogTrain->clear();
    ui.progressBarSeg->setValue(0);
    ui.btnStartTraining->setEnabled(false); // Deshabilitar botón para evitar doble click

    // 2. Crear Worker y Thread
    // Nota: QThread gestiona la memoria si lo configuramos bien
    QThread* thread = new QThread;
    TrainingWorker* worker = new TrainingWorker(config);
    worker->moveToThread(thread);

    // 3. Conectar señales

    // Cuando el hilo arranca -> worker empieza a procesar
    connect(thread, &QThread::started, worker, &TrainingWorker::process);

    // Actualizar barra de progreso
    connect(worker, &TrainingWorker::progressSeg, ui.progressBarSeg, &QProgressBar::setValue);
    connect(worker, &TrainingWorker::progressExtract, ui.progressBarExtract, &QProgressBar::setValue);

    // Logs al cuadro de texto
    connect(worker, &TrainingWorker::logMessage, this, [this](QString msg) {
        ui.txtLogTrain->append(msg);
        });

    // Limpieza al terminar
    connect(worker, &TrainingWorker::finished, thread, &QThread::quit);
    connect(worker, &TrainingWorker::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    // Reactivar botón al terminar
    connect(thread, &QThread::finished, this, [this]() {
        ui.btnStartTraining->setEnabled(true);
        ui.txtLogTrain->append("<b>Proceso finalizado.</b>");
        });

    // 4. Iniciar
    thread->start();
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

        ui.chkLiveSeg->setEnabled(true);
    }
    else {
        // Se ha apagado
        ui.pbtnEncender->setText("Encender Cámara");
        ui.btnCapturarAnalizar->setEnabled(false);

        // 1. Apagar Segmentación
        if (ui.chkLiveSeg->isChecked()) {
            ui.chkLiveSeg->setChecked(false);
        }
        // 2. Deshabilitar el control
        ui.chkLiveSeg->setEnabled(false);

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
        ui.lblVideoLive->setAlignment(Qt::AlignCenter);
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
    QPainter p(&scaled);

    // DIBUJAR CAJAS VERDES (Segmentación en vivo)
    if (LiveSegmentationEnabled && !lastBoxesNormalized.empty()) {
        QPen pen(Qt::green);
        pen.setWidth(3);
        p.setPen(pen);

        for (size_t i = 0; i < lastBoxesNormalized.size(); ++i) {
            const auto& boxNorm = lastBoxesNormalized[i];
            int x = static_cast<int>(boxNorm.x() * scaled.width());
            int y = static_cast<int>(boxNorm.y() * scaled.height());
            int w = static_cast<int>(boxNorm.width() * scaled.width());
            int h = static_cast<int>(boxNorm.height() * scaled.height());

            p.drawRect(x, y, w, h);
        }
    }
    // DIBUJAR CLASIFICACIÓN
    if (LiveClassificationEnabled && !lastClassBoxes.empty()) {
        QPen pen(Qt::green);
        pen.setWidth(2);
        p.setPen(pen);

        QFont font = p.font();
        font.setPixelSize(std::max<double>(14, scaled.height() / 20));
        font.setBold(true);
        p.setFont(font);

        for (size_t i = 0; i < lastClassBoxes.size(); ++i) {
            if (i >= lastClassLabels.size()) break;

            const auto& boxNorm = lastClassBoxes[i];
            int x = static_cast<int>(boxNorm.x() * scaled.width());
            int y = static_cast<int>(boxNorm.y() * scaled.height());
            int w = static_cast<int>(boxNorm.width() * scaled.width());
            int h = static_cast<int>(boxNorm.height() * scaled.height());

            // Dibujar caja
            //p.drawRect(x, y, w, h);

            // Dibujar texto con fondo
            QString text = lastClassLabels[i];
            QFontMetrics fm(font);
            int tw = fm.horizontalAdvance(text);
            int th = fm.height();

            p.fillRect(x, y - th - 4, tw + 4, th + 4, QColor(0, 0, 0, 150));
            p.drawText(x + 2, y - 4, text);
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
    if (enabled) {
        ui.chkLiveClass->setEnabled(true);
    }
    else {
        if (ui.chkLiveClass->isChecked()) {
            ui.chkLiveClass->setChecked(false);
        }
        ui.chkLiveClass->setEnabled(false);
        SegProcessing = false;
        lastBoxesNormalized.clear();
        segInFlight.store(0);
        // Limpiar thumbnails
        if (ui.lblLiveThumb1) ui.lblLiveThumb1->clear();
        if (ui.lblLiveThumb2) ui.lblLiveThumb2->clear();
        if (ui.lblLiveThumb3) ui.lblLiveThumb3->clear();
    }
}

void ProyectoPSM::UpdateSegmentationResults(const std::vector<QRectF>& boxes,
    const std::vector<QImage>& thumbnails,
    const std::vector<cv::Mat>& crops)
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

    // Si la clasificación está activa y el worker está libre, le pasamos los datos
    if (LiveClassificationEnabled && !ClassProcessing.load()) {
        ClassProcessing = true;
        emit requestClassification(crops, boxes);
    }
}

void ProyectoPSM::onCheckLiveClass(bool checked)
{
    LiveClassificationEnabled = checked;
    if (!checked) {
        lastClassBoxes.clear();
        lastClassLabels.clear();
        ClassProcessing = false;
    }
    else {
        // Asegurar que los modelos estén cargados
		// Ojo, ahora hay una nueva funcion: EnsureOrientTemplatesLoaded
        if (!svmClf_->IsLoaded()) {
            // Cargar SVM (mismas rutas que offline)
            svmClf_->Load("../../Matlab/Clasificador/Clasificador C/modelM.yml",
                "../../Matlab/Clasificador/Clasificador C/modelM_scaler.yml");
        }
        if (!orientTemplatesLoaded_) {
            if (orientClf_->loadAllTemplates()) orientTemplatesLoaded_ = true;
        }
    }
}


void ProyectoPSM::UpdateClassificationResults(std::vector<QRectF> boxes, std::vector<QString> labels)
{
    lastClassBoxes = boxes;
    lastClassLabels = labels;
    ClassProcessing = false;
    // ShowImage se actualizará automáticamente en el siguiente frame de video (NewImage)
}

// CAPTURA Y ANÁLISIS OFFLINE
void ProyectoPSM::CapturarYAnalizar()
{
    // 1. Verificar imagen
    if (LastImage.empty()) {
        QMessageBox::warning(this, "Error", "No hay imagen en vivo para capturar.");
        return;
    }

    // 2. Congelar imagen actual
    CapturedImage = LastImage.clone();

    // APAGADO AUTOMÁTICO
        if (ui.pbtnEncender->isChecked()) {
            ui.pbtnEncender->setChecked(false);
        }

    // 3. Cambiar a la pestaña de Análisis
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // 4. Limpiar visualización anterior para evitar confusión
    ui.lblOfflineMain->clear();
    ui.lblOfflineThumb1->clear(); ui.lblOfflineThumb2->clear(); ui.lblOfflineThumb3->clear();

    // Mostrar la imagen capturada TAL CUAL (sin procesar aún)
    DisplayMat(ui.lblOfflineMain, CapturedImage);

    // NO procesamos automáticamente. El usuario debe pulsar los botones.
    // ProcesarImagenOffline(CapturedImage); 
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

    if (image.empty()) {
        QMessageBox::warning(this, "Error", "No se pudo cargar la imagen seleccionada.");
        return;
    }

    // 3. Guardar como imagen capturada
    CapturedImage = image.clone();

    // Asegurar que estamos en la pestaña correcta
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // Limpiar y mostrar imagen cruda
    ui.lblOfflineMain->clear();
    ui.lblOfflineThumb1->clear(); ui.lblOfflineThumb2->clear(); ui.lblOfflineThumb3->clear();
    DisplayMat(ui.lblOfflineMain, CapturedImage);

    // NO procesamos automáticamente
    // ProcesarImagenOffline(CapturedImage);
}

void ProyectoPSM::RecalcularSegmentacion()
{
    // Validación de imagen capturada
    if (CapturedImage.empty()) {
        QMessageBox::warning(this, "Error", "No hay ninguna imagen cargada para clasificar.");
        return;
    }
    ProcesarImagenOffline(CapturedImage);    
}

// Lógica central de Análisis (Segmentación pura)
void ProyectoPSM::ProcesarImagenOffline(const cv::Mat& img)
{
    if (img.empty()) return;

    ui.lblOfflineMain->setText("Procesando...");

    // Actualizar tamaños de las labels sin que el usuario lo note
    this->setUpdatesEnabled(false);

    ui.tabWidgetAnalysis->setCurrentWidget(ui.subTabDebugSeg);
    QApplication::processEvents();

    // Recorrer sub-pestañas
    // Obliga a Qt a calcular el tamaño de los labels
    int originalSubTab = ui.tabWidgetDebug->currentIndex();
    for (int i = 0; i < ui.tabWidgetDebug->count(); i++) {
        ui.tabWidgetDebug->setCurrentIndex(i);
        QApplication::processEvents();
    }
    ui.tabWidgetDebug->setCurrentIndex(originalSubTab);

    // Preparar datos y Segmentar
    DebugInfo debugData;
    std::vector<ResultadoPieza> resultados = Segmentacion::Segmentar(img, &debugData);

    // Guardar resultados para clasificación posterior (botón independiente)
    lastResultados_ = resultados;

    // RELLENAR PESTAÑAS
    DisplayMat(ui.lblHSV_1_Orig, debugData.I_orig);
    DisplayMat(ui.lblHSV_2_Norm, debugData.I_norm);
    DisplayMat(ui.lblHSV_3_H, debugData.H, true);
    DisplayMat(ui.lblHSV_4_S, debugData.S, true);
    DisplayMat(ui.lblHSV_5_V, debugData.V, true);

    DisplayMat(ui.lblOtsu_1_S, debugData.S_proc, true);
    DrawHistogram(ui.lblOtsu_2_Hist, debugData.S_proc);
    DisplayMat(ui.lblOtsu_3_Mask, debugData.mask_otsu, true);

    DisplayMat(ui.lblMorph_1_Bin, debugData.mask_otsu, true);
    DisplayMat(ui.lblMorph_2_Fill, debugData.mask_fill, true);
    DisplayMat(ui.lblMorph_3_Clean, debugData.mask_clean, true);
    DisplayMat(ui.lblMorph_4_Border, debugData.mask_border, true);
    DisplayMat(ui.lblMorph_5_Close, debugData.mask_close, true);
    DisplayMat(ui.lblMorph_6_Final, debugData.mask_final, true);

    ui.tabWidgetAnalysis->setCurrentWidget(ui.subTabResultados);
    this->setUpdatesEnabled(true);

    cv::Mat displayImg = img.clone();
    for (const auto& res : resultados) {
        cv::rectangle(displayImg, res.boundingBox, cv::Scalar(0, 255, 0), 3);
    }
    DisplayMat(ui.lblOfflineMain, displayImg);

    // Miniaturas (sin texto de clasificación; se actualizarán al pulsar Clasificar)
    if (resultados.empty()) {
        ui.lblOfflineThumb1->clear(); ui.lblOfflineThumb2->clear(); ui.lblOfflineThumb3->clear();
        ui.pbtnGuardar->setEnabled(false);
    }
    else {
        ui.pbtnGuardar->setEnabled(true);
        QLabel* thumbs[] = { ui.lblOfflineThumb1, ui.lblOfflineThumb2, ui.lblOfflineThumb3 };
        for (int i = 0; i < 3; i++) {
            if (i < resultados.size()) DisplayMat(thumbs[i], resultados[i].imagenRecortada);
            else { thumbs[i]->clear(); thumbs[i]->setText("---"); }
        }
    }
}

// Clasifica las piezas guardadas en lastResultados_ y actualiza miniaturas y vista principal
void ProyectoPSM::ProcesarClasificacionOffline()
{
    // 1. Validación de imagen capturada
    if (CapturedImage.empty()) {
        QMessageBox::warning(this, "Error", "No hay ninguna imagen cargada para clasificar.");
        return;
    }

    // Si no se ha segmentado aún, forzamos la segmentación primero
    if (lastResultados_.empty()) {
        RecalcularSegmentacion();
        if (lastResultados_.empty()) {
            QMessageBox::information(this, "Clasificar", "No se detectaron piezas en la imagen.");
            return;
        }
    }

    // 2. CARGA DEL SVM
    if (!svmClf_->IsLoaded()) {
        std::string pathModel = "../../Matlab/Clasificador/Clasificador C/modelM.yml";
        std::string pathScaler = "../../Matlab/Clasificador/Clasificador C/modelM_scaler.yml";
        bool ok = svmClf_->Load(pathModel, pathScaler);
        if (!ok) {
            QMessageBox::warning(this, "Error Crítico", "No se pudo cargar el modelo SVM.");
            return;
        }
    }

    // 1. Convertir Mat (BGR) a formato compatible con Qt (RGB)
    cv::Mat rgbMat;
    cv::cvtColor(CapturedImage, rgbMat, cv::COLOR_BGR2RGB);

    // 2. Crear una QImage sobre la que pintaremos
    // Hacemos .copy() para tener una copia profunda y poder modificarla sin tocar la original
    QImage displayImg = QImage(rgbMat.data, rgbMat.cols, rgbMat.rows,
        static_cast<int>(rgbMat.step), QImage::Format_RGB888).copy();

    // 3. Iniciar el pintor
    QPainter p(&displayImg);

    // Configurar fuente dinámica según tamaño de imagen
    QFont font = p.font();
    int pixelSize = std::max<double>(12, displayImg.width() / 40); // Ajusta el divisor para cambiar tamaño
    font.setPixelSize(pixelSize);
    font.setBold(true);
    p.setFont(font);

    // Limpiamos miniaturas
    QLabel* thumbs[] = { ui.lblOfflineThumb1, ui.lblOfflineThumb2, ui.lblOfflineThumb3 };
    for (int k = 0; k < 3; ++k) thumbs[k]->clear();
    
    for (size_t i = 0; i < lastResultados_.size(); ++i) {
        ResultadoPieza& res = lastResultados_[i];
        if (res.imagenRecortada.empty()) continue;

        std::string codigoPieza = "";
        int clasePredicha = -1;
        bool svmExito = false;

        // --- PASO A: SVM ---
        if (svmClf_->IsLoaded()) {
            clasePredicha = svmClf_->Predict(res.imagenRecortada);
            if (clasePredicha > 0) {
                codigoPieza = std::to_string(clasePredicha);
                if (clasePredicha < 10) codigoPieza = "0" + codigoPieza;
                res.id = clasePredicha;
                svmExito = true;
            }
        }

        // --- PASO B: ORIENTACIÓN ---
        QString labelInfo = "Desc."; // Ahora usamos QString directamente

        if (svmExito) {
            OrientationResult orr;
            bool orientExito = false;

            if (orientTemplatesLoaded_) {
                try {
                    orr = orientClf_->predict(res.imagenRecortada, codigoPieza);
                    orientExito = orr.ok;
                }
                catch (...) {}
            }

            // Usamos QString::arg para formatear
            if (orientExito) {
                labelInfo = QString("Cód: %1 Orientación: %2º")
                    .arg(QString::fromStdString(codigoPieza))
                    .arg(orr.yaw);
            }
            else {
                labelInfo = QString("Cód: %1")
                    .arg(QString::fromStdString(codigoPieza));
            }
        }
        else {
            labelInfo = "Desconocido";
        }

        // --- VISUALIZACIÓN ---

        // 1. Miniatura (se mantiene igual usando DisplayMat)
        if (i < 3) {
            DisplayMat(thumbs[i], res.imagenRecortada);
        }

        // 2. Dibujar sobre la imagen principal usando QPainter

        // Convertir coordenadas de OpenCV a Qt
        int x = res.boundingBox.x;
        int y = res.boundingBox.y;
        int w = res.boundingBox.width;
        int h = res.boundingBox.height;

        // A. Dibujar rectángulo verde
        QPen pen(Qt::green);
        pen.setWidth(3);
        p.setPen(pen);
        p.drawRect(x, y, w, h);

        // B. Calcular tamaño del texto para el fondo negro
        QFontMetrics fm(font);
        int textWidth = fm.horizontalAdvance(labelInfo);
        int textHeight = fm.height();
        int padding = 4;

        // Posición del texto (arriba de la caja, o abajo si se sale)
        int textX = x;
        int textY = y - padding;
        if (textY < textHeight) textY = y + h + textHeight + padding;

        // C. Dibujar fondo negro semi-transparente
        p.fillRect(textX, textY - textHeight, textWidth + (padding * 2), textHeight + padding, QColor(0, 0, 0, 180));

        // D. Dibujar texto
        p.setPen(Qt::green);
        p.drawText(textX + padding, textY, labelInfo);
    }

    p.end(); // Finalizar pintura

    // Mostrar resultado final en el Label
    ui.lblOfflineMain->setPixmap(QPixmap::fromImage(displayImg)
        .scaled(ui.lblOfflineMain->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
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


// helper: extrae code del nombre "02_045_090_001" -> "02"
static std::string ExtractCodeFromFilename(const QString& baseName)
{
    // baseName: sin extensión, ej "02_045_090_001"
    // queremos los 2 primeros dígitos antes del primer '_'
    QRegularExpression re(R"(^(\d{2})_)");
    auto m = re.match(baseName);
    if (m.hasMatch()) return m.captured(1).toStdString();
    return "";
}

void ProyectoPSM::AbrirYClasificarOrientacion()
{
    // 1) elegir imagen
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("Abrir imagen de pieza"),
        "",
        tr("Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
    );
    if (fileName.isEmpty()) return;

    // 2) leer con Qt -> cv::Mat (igual que tu CargarImagenDisco)
    QFile f(fileName);
    if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "Error", "No se pudo abrir el archivo.");
        return;
    }
    QByteArray fileData = f.readAll();
    f.close();

    std::vector<uchar> vec(fileData.begin(), fileData.end());
    cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR);
    if (image.empty()) {
        QMessageBox::warning(this, "Error", "La imagen no se pudo decodificar.");
        return;
    }

    // 3) mostrarla en tu visor offline (reutiliza tu pipeline si quieres)
    CapturedImage = image.clone();
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // opcional: muestra la imagen en lblOfflineMain directamente
    {
        cv::Mat rgb;
        cv::cvtColor(CapturedImage, rgb, cv::COLOR_BGR2RGB);
        QImage qimg(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
        ui.lblOfflineMain->setPixmap(QPixmap::fromImage(qimg.copy()).scaled(
            ui.lblOfflineMain->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    // 4) extraer code desde el nombre
    QFileInfo info(fileName);
    QString base = info.completeBaseName();            // "02_045_090_001"
    std::string code = ExtractCodeFromFilename(base); // "02"

    if (code.empty()) {
        QMessageBox::warning(this, "Nombre inválido",
            "No he podido extraer el code del nombre.\n"
            "Ejemplo esperado: 02_045_090_001.jpg");
        return;
    }

    // 5) cargar plantillas (una sola vez)

    if (!QFileInfo::exists(orientTemplatesDir_) || !QFileInfo(orientTemplatesDir_).isDir()) {
        QMessageBox::critical(this, "Error",
            "No existe la carpeta de templates:\n" + orientTemplatesDir_);
        return;
    }
    if (!orientTemplatesLoaded_) {
        if (!orientClf_ || !orientClf_->loadAllTemplates()) {
            QMessageBox::critical(this, "Error",
                "No se pudieron cargar las plantillas .yml/.yaml.\n"
                "Revisa la ruta de templatesFolder_.");
            return;
        }
        orientTemplatesLoaded_ = true;
    }

    // 6) clasificar orientación
    // aquí pasas la imagen de la pieza; si ya vienes con recorte, pásale el recorte.
    // ahora mismo pasamos la imagen completa.
    OrientationResult r = orientClf_->predict(CapturedImage, code);

    if (!r.ok) {
        ui.lblOrientacionResult->setText(
            QString("No se pudo clasificar (code=%1)").arg(QString::fromStdString(code)));
        return;
    }

    ui.lblOrientacionResult->setText(
        QString("code=%1   yaw=%2   pitch=%3   score=%4   gap=%5")
        .arg(QString::fromStdString(r.matchedCode))
        .arg(r.yaw)
        .arg(r.pitch)
        .arg(r.bestScore, 0, 'f', 4)
        .arg(r.gap, 0, 'f', 4)
    );
}

//PRUEBAS DE CLASIFICACIÓN
void ProyectoPSM::runEvalGlobal() {
    //const char* args[] = {
    //    "eval",
    //    R"(C:\Desarrollos\proyectoPSM\SEGMENTED)", // segFolder
    //    R"(C:\Desarrollos\proyectoPSM\eval_out.txt)",      // outTxt
    //    R"(C:\Desarrollos\proyectoPSM\models\modelM.yml)" // modelM.yml
    //};

    //const char* args[] = {
    //    "eval",
    //    R"(C:/Users/jlaco/OneDrive/Escritorio/1/Procesado de Señales Multimedia/Proyecto/ProyectoPSM/Database/SEGMENTED)", // segFolder
    //    R"(C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador C\eval_out.txt)",      // outTxt
    //    R"(C:\Users\jlaco\OneDrive\Escritorio\1\Procesado de Señales Multimedia\Proyecto\ProyectoPSM\Matlab\Clasificador\Clasificador C\modelM.yml)" // modelM.yml
    //};
    const char* args[] = {
        "eval",
        R"(../../Database/SEGMENTED_TEST_C)",
        R"(../../Matlab/Clasificador/Clasificador C/eval_out_nuevaSeg.txt)",
        R"(../../Matlab/Clasificador/Clasificador C/modelM.yml)"
    };
    qDebug() << "RunEval outTxt =" << args[2];
    int rc = RunEval(4, const_cast<char**>(args));

    if (rc == 0) {
        qDebug() << "EVAL OK. TXT guardado en:" << args[2];
    }
    else {
        qDebug() << "EVAL FAIL. rc =" << rc << " | outTxt =" << args[2];
    }
}

void ProyectoPSM::runEvalAmarillas() {
    RunEvalRefinerOnly(
        R"(../../Database/SEGMENTED_TEST_AMARILLAS)",
        R"(../../Matlab/Clasificador/Clasificador C/eval_refiner_only.txt)",
        R"(../../Matlab/Clasificador/Clasificador C/model912.yml)"
    );
}

void ProyectoPSM::maybeTrain() {
    TrainSVM::Options opts;
    opts.inputFolder = R"(../../Database/SEGMENTED_TRAIN_C)";
    opts.outModelPath = R"(../../Matlab/Clasificador/Clasificador C/modelM.yml)";

    opts.csvOut = ""; // opcional
    opts.doScale = true;
    opts.C = 1.0;
    opts.gamma = 0.0;

    if (!std::filesystem::exists(opts.outModelPath)) {
        qDebug("Entrenando modelo...");
        int r = RunTrain(opts); //Clasificador gordo
		//int r = RunTrainRefiner(opts, true); //Clasificador amarillas

        if (r != 0) std::cerr << "RunTrain fallo: " << r << "\n";
    }
    else {
        std::cout << "Modelo ya existe, omitiendo entrenamiento.\n";
    }
}


bool ProyectoPSM::EnsureOrientTemplatesLoaded()
{
    if (orientTemplatesLoaded_) return true;

    if (!orientClf_) {
        QMessageBox::warning(this, "Error", "orientClf_ no está inicializado.");
        return false;
    }

    try {
        if (!orientClf_->loadAllTemplates()) {
            QMessageBox::warning(this, "Error", "No se pudieron cargar las plantillas de orientación.");
            orientTemplatesLoaded_ = false;
            return false;
        }
    }
    catch (...) {
        QMessageBox::warning(this, "Error", "Excepción al cargar plantillas de orientación.");
        orientTemplatesLoaded_ = false;
        return false;
    }

    orientTemplatesLoaded_ = true;
    return true;
}

void ProyectoPSM::OnBatchSegmentar()
{
    std::string in = R"(../../Database/RAW)";
    std::string out = R"(../../Database/SEGMENTED_C)";

    SegmentBatchStats st = SegmentFolderAndSaveCrops(
        in, out,
        false,   // keepSubfolders
        0,       // maxPiecesPerImage (0 = todas)
        true     // verbose
    );
}


