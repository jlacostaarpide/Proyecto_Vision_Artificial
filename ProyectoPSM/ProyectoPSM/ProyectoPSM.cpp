//-----------------------------------------------------------------------------------------
// Script principal para la gestion del procesamiento de imagenes y de la interfaz gráfica
//-----------------------------------------------------------------------------------------
#include "ProyectoPSM.h"
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

//----------------------------------------------------------------------------
//Funciones Auxiliares
//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
// Clase Principal
//----------------------------------------------------------------------------
ProyectoPSM::ProyectoPSM(QWidget* parent) : QMainWindow(parent)
{
    ui.setupUi(this);

    svmClf_ = std::make_unique<Clasificador>();

    // Inicializar pestañas
    ui.tabWidget->setCurrentIndex(0);
    ui.tabWidgetAnalysis->setCurrentIndex(0);
    ui.tabWidgetDebug->setCurrentIndex(0);

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
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);
    m_featureNames = {
        "Extent", "Solidity", "V_mean", "Eccentricity", "SkelLenNorm", "Circularity",
        "H_mean_circ", "S_mean", "V_IQR", "S_median", "FD5", "EulerNumber"
    };

    // Cargar rutas iniciales en los textbox
    LoadDefaultSettings(); 

	// Cargar Clasificador de Orientación
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

	// 3. Configurar Clasification Worker
    classWorker = new ClasificationWorker(svmClf_.get(), orientClf_.get());
    classThread = new QThread(this);
    classWorker->moveToThread(classThread);
    connect(classThread, &QThread::finished, classWorker, &QObject::deleteLater);
    connect(this, &ProyectoPSM::requestClassification, classWorker, &ClasificationWorker::process);
    connect(classWorker, &ClasificationWorker::finishedResult, this, &ProyectoPSM::UpdateClassificationResults);
    classThread->start();

    // 4. Timers
    segTimer = new QTimer(this);
    segTimer->setInterval(SegmentationIntervalMs);
    connect(segTimer, &QTimer::timeout, this, &ProyectoPSM::onSegmentationTimer);
    segTimer->start();

    // Watchdog Timer
    statusTimer = new QTimer(this);
    statusTimer->setInterval(2000);
    connect(statusTimer, &QTimer::timeout, this, &ProyectoPSM::CheckCameraStatus);
    statusTimer->start();

    // 5. Conexiones UI Principales
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

    ui.chkLiveSeg->setEnabled(false);
    ui.chkLiveClass->setEnabled(false);
    ui.chkLiveSeg->setChecked(false);
    ui.chkLiveClass->setChecked(false);

    // 6. Configurar estado inicial Cámara
    bool camOk = (Camera && Camera->CameraOK);
    SetCameraStatusUI(camOk);
    if (camOk) {
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
        Camera->SetCameraAutoExposure();
    }


	// 7. Conexiones Pestaña Entrenamiento
    // Botones de examinar (Browse)
    connect(ui.btnBrowseRaw, &QPushButton::clicked, this, &ProyectoPSM::onBrowseRaw);
    connect(ui.btnBrowseSeg, &QPushButton::clicked, this, &ProyectoPSM::onBrowseSeg);
    connect(ui.btnBrowseFeatures, &QPushButton::clicked, this, &ProyectoPSM::onBrowseFeatures);
    connect(ui.btnBrowseTemplates, &QPushButton::clicked, this, &ProyectoPSM::onBrowseTemplates);
    connect(ui.btnBrowseModel, &QPushButton::clicked, this, &ProyectoPSM::onBrowseModel);
    connect(ui.btnBrowseTest, &QPushButton::clicked, this, &ProyectoPSM::onBrowseTest);

    // Checkboxes (Saltar pasos)
    connect(ui.chkSkipSeg, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipSeg);
    connect(ui.chkSkipExtract, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipExtract);
    connect(ui.chkSkipTrain, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipTrain);
    connect(ui.chkSkipEval, &QCheckBox::toggled, this, &ProyectoPSM::onCheckSkipEval);

    // Botón Iniciar Proceso
    connect(ui.btnStartTraining, &QPushButton::clicked, this, &ProyectoPSM::onStartTrainingClicked);

	// 8. Conexiones Pestaña Análisis
    connect(ui.btnLoadEval, &QPushButton::clicked, this, &ProyectoPSM::onLoadEvaluationFile);
    connect(ui.btnSaveConfusion, &QPushButton::clicked, this, &ProyectoPSM::onSaveConfusionMatrix);

    connect(ui.btnLoadFeaturesPlot, &QPushButton::clicked, this, &ProyectoPSM::onLoadFeaturesPlot);
    connect(ui.btnGenerateScatter, &QPushButton::clicked, this, &ProyectoPSM::onGenerateScatter);
    connect(ui.btnSaveScatter, &QPushButton::clicked, this, &ProyectoPSM::onSaveScatter);
    connect(ui.rbPCAGlobal, &QRadioButton::toggled, this, &ProyectoPSM::onScatterModeChanged);
    connect(ui.rbPCACompare, &QRadioButton::toggled, this, &ProyectoPSM::onScatterModeChanged);
    connect(ui.rbManualFeat, &QRadioButton::toggled, this, &ProyectoPSM::onScatterModeChanged);

    ImageIndex = 0;
    SavedImageIndex = 1;
    ui.boxImageNumber->setValue(SavedImageIndex);
    UpdateFileNameLabel();

    connect(ui.btnSetTemplates, &QPushButton::clicked, this, &ProyectoPSM::onSetBrowseTemplates);
    connect(ui.btnSetModel, &QPushButton::clicked, this, &ProyectoPSM::onSetBrowseModel);
    connect(ui.btnSetScaler, &QPushButton::clicked, this, &ProyectoPSM::onSetBrowseScaler);
    connect(ui.chkUseDbNames, &QCheckBox::toggled, this, &ProyectoPSM::UpdateFileNameLabel);
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

//----------------------------------------------------------------------------
// Workers para procesamiento en segundo plano de SEGMENTACIÓN Y CLASIFICACIÓN
//----------------------------------------------------------------------------
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

                if (orientExito) labelText = QString("Cód: %1 Orientación: %2º").arg(QString::fromStdString(codigoPieza)).arg(yaw);
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

//----------------------------------------------------------------------------
// Funciones de la pestaña de entrenamiento en la interfaz gráfica
//----------------------------------------------------------------------------
QString getSmartStartDir(const QString& currentText, const QString& rutaPorDefecto) {
    if (!currentText.isEmpty()) {
        QFileInfo info(currentText);
        if (info.exists()) return currentText;

        if (info.absoluteDir().exists()) return info.absolutePath();
    }

    if (!rutaPorDefecto.isEmpty() && QDir(rutaPorDefecto).exists()) {
        return rutaPorDefecto;
    }

    if (QDir("Database").exists()) return "Database";
    return QDir::currentPath();
}
void ProyectoPSM::onBrowseRaw() {
    QString defaultDir = "../../Database/RAW";

    QString startPath = getSmartStartDir(ui.txtPathRaw->text(), defaultDir);
    QString dir = QFileDialog::getExistingDirectory(this, "Seleccionar Carpeta RAW", startPath);

    if (!dir.isEmpty()) ui.txtPathRaw->setText(dir);
}

void ProyectoPSM::onBrowseSeg() {
    QString defaultDir = "../../Database/SEGMENTED";

    QString title = ui.chkSkipSeg->isChecked() ? "Seleccionar Carpeta Segmentadas (Origen)"
        : "Seleccionar Carpeta Segmentadas (Destino)";

    QString startPath = getSmartStartDir(ui.txtPathSeg->text(), defaultDir);
    QString dir = QFileDialog::getExistingDirectory(this, title, startPath);

    if (!dir.isEmpty()) ui.txtPathSeg->setText(dir);
}

void ProyectoPSM::onBrowseTest() {
    QString defaultDir = "../../Database";

    QString startPath = getSmartStartDir(ui.txtPathTest->text(), defaultDir);
    QString dir = QFileDialog::getExistingDirectory(this, "Seleccionar Carpeta Test", startPath);

    if (!dir.isEmpty()) ui.txtPathTest->setText(dir);
}

void ProyectoPSM::onBrowseFeatures() {
    QString defaultFile = "Database/Models/features.yml";

    QString startFile = getSmartStartDir(ui.txtPathFeatures->text(), "Database");

    // Si getSmartStartDir devolvió un directorio genérico, le pegamos el nombre de archivo por defecto
    if (QFileInfo(startFile).isDir()) {
        // Si no hay nada escrito, sugerimos la ruta completa por defecto
        if (ui.txtPathFeatures->text().isEmpty()) startFile = defaultFile;
    }

    QString file;
    if (ui.chkSkipExtract->isChecked()) {
        file = QFileDialog::getOpenFileName(this, "Cargar Features Existentes",
            startFile,
            "YAML/XML Files (*.yml *.yaml *.xml)");
    }
    else {
        file = QFileDialog::getSaveFileName(this, "Guardar Nuevas Features",
            startFile,
            "YAML Files (*.yml *.yaml);;XML Files (*.xml)");
    }
    if (!file.isEmpty()) ui.txtPathFeatures->setText(file);
}

void ProyectoPSM::onBrowseTemplates() {
    QString defaultDir = "Database/Templates";
    QString startPath = getSmartStartDir(ui.txtPathTemplates->text(), defaultDir);

    // Lógica dinámica: Cambiamos el titulo según el checkbox
    QString title;
    if (ui.chkSkipTemplates->isChecked()) {
        // Caso INPUT: El usuario busca plantillas ya existentes para cargar
        title = "Seleccionar Carpeta de Plantillas (Origen)";
    }
    else {
        // Caso OUTPUT: El usuario busca dónde guardar las nuevas plantillas
        title = "Seleccionar Carpeta de Plantillas (Destino)";
    }

    // Nota: En ambos casos usamos getExistingDirectory porque las templates 
    // son un conjunto de archivos dentro de una carpeta, no un archivo único.
    QString dir = QFileDialog::getExistingDirectory(this, title, startPath);

    if (!dir.isEmpty()) ui.txtPathTemplates->setText(dir);
}

void ProyectoPSM::onBrowseModel() {
    QString defaultFile = "Database/Models/modelM.yml";

    QString startFile = getSmartStartDir(ui.txtPathModel->text(), "Database");

    if (QFileInfo(startFile).isDir()) {
        if (ui.txtPathModel->text().isEmpty()) startFile = defaultFile;
    }

    QString file;
    if (ui.chkSkipTrain->isChecked()) {
        file = QFileDialog::getOpenFileName(this, "Cargar Modelo Existente",
            startFile,
            "YAML Files (*.yml *.yaml)");
    }
    else {
        file = QFileDialog::getSaveFileName(this, "Guardar Nuevo Modelo",
            startFile,
            "YAML Files (*.yml *.yaml)");
    }
    if (!file.isEmpty()) ui.txtPathModel->setText(file);
}

//---------------------------------------------------------------------------
// Funciones para actualizar la UI según los checkboxes
//---------------------------------------------------------------------------
void ProyectoPSM::onCheckSkipSeg(bool checked) {
    ui.txtPathRaw->setEnabled(!checked);
    ui.btnBrowseRaw->setEnabled(!checked);
    // Cambiar placeholder o etiqueta para indicar que ahora cargamos desde segmentadas
    if (checked) ui.label_2->setText("Cargar Segmentadas:");
    else ui.label_2->setText("Salida Segmentadas:");
}

void ProyectoPSM::onCheckSkipExtract(bool checked) {
    // Si saltamos extracción, necesitamos cargar features, pero no necesitamos la carpeta de segmentadas
    // Esto depende de cómo se quiera encadena.
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

//Funcion para iniciar el proceso de entrenamiento en un hilo separado
void ProyectoPSM::onStartTrainingClicked() {
    // 1. Configurar rutas desde la UI
    TrainingConfig config;
    config.rawFolder = ui.txtPathRaw->text();
    config.segFolder = ui.txtPathSeg->text();
    config.featuresFile = ui.txtPathFeatures->text();
    config.templatesFolder = ui.txtPathTemplates->text();
    config.modelFile = ui.txtPathModel->text();
	config.evaluationFolder = ui.txtPathTest->text();

    config.skipSegmentation= ui.chkSkipSeg->isChecked();
    config.skipExtraction = ui.chkSkipExtract->isChecked();
    config.skipTemplates = ui.chkSkipTemplates->isChecked();
	config.skipTraining = ui.chkSkipTrain->isChecked();
	config.skipEvaluation = ui.chkSkipEval->isChecked();

    // Resetear UI
    ui.txtLogTrain->clear();
    ui.progressBarSeg->setValue(0);
    ui.progressBarExtract->setValue(0);
    ui.progressBarTemplates->setValue(0);
    ui.progressBarTrain->setValue(0);
    ui.progressBarEval->setValue(0);
    ui.btnStartTraining->setEnabled(false); // Bloquear botón

    // 2. Crear Worker y Thread
    QThread* thread = new QThread;
    TrainingWorker* worker = new TrainingWorker(config);
    worker->moveToThread(thread);

    // 3. Conectar señales

    // Cuando el hilo arranca -> worker empieza a procesar
    connect(thread, &QThread::started, worker, &TrainingWorker::process);

    // Actualizar barra de progreso
    connect(worker, &TrainingWorker::progressSeg, ui.progressBarSeg, &QProgressBar::setValue);
    connect(worker, &TrainingWorker::progressExtract, ui.progressBarExtract, &QProgressBar::setValue);
    connect(worker, &TrainingWorker::progressTemplates, ui.progressBarTemplates, &QProgressBar::setValue);
    connect(worker, &TrainingWorker::progressTrain, ui.progressBarTrain, &QProgressBar::setValue);
	connect(worker, &TrainingWorker::progressEval, ui.progressBarEval, &QProgressBar::setValue);

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
        // Actualizar pestañas de ajustes automáticamente para comodidad
        if (!ui.txtPathModel->text().isEmpty()) {
            ui.txtSetModel->setText(ui.txtPathModel->text());

            // Inferir el scaler
            QFileInfo info(ui.txtPathModel->text());
            QString scalerPath = info.absolutePath() + "/" + info.baseName() + "_scaler.yml";
            ui.txtSetScaler->setText(scalerPath);
        }
        });

    // 4. Iniciar
    thread->start();
}

//---------------------------------------------------------------------------
// Logica de gestión de la cámara
//---------------------------------------------------------------------------
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


//----------------------------------------------------------------------------
// Lógica de captura y visualización de imágenes
//----------------------------------------------------------------------------
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

// Mostrar la imagen actuaL con las cajas dibujadas y el resultado de clasificación
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

    // Dibujar los bonding boxes (Segmentación en vivo)
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
	// Dibujar clasificaciones en vivo
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

            // Preparar texto y métricas
            QString text = lastClassLabels[i];
            QFontMetrics fm(font);
            int tw = fm.horizontalAdvance(text);
            int th = fm.height();
            int padding = 4;

            // Dimensiones totales del rectángulo de fondo
            int labelW = tw + padding;
            int labelH = th + padding;

            // Calcular posición inicial
            // Encima de la caja, alineado a la izquierda
            int labelX = x;
            int labelY = y - labelH;

            // Si la etiqueta se sale por la derecha
            if (labelX + labelW > scaled.width()) {
                labelX = scaled.width() - labelW;
            }
            if (labelX < 0) labelX = 0;

            // Si la etiqueta se sale por arriba
            if (labelY < 0) {
                labelY = y;
            }

            // Dibujar fondo negro
            p.fillRect(labelX, labelY, labelW, labelH, QColor(0, 0, 0, 150));

            // Dibujar texto
            p.drawText(labelX + 2, labelY + th, text);
        }
    }
    ui.lblVideoLive->setPixmap(scaled);
}

//----------------------------------------------------------------------------
// Lógica de Segmentación y Clasificación en Vivo
//----------------------------------------------------------------------------
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
        QString modelPath = ui.txtSetModel->text();
        QString scalerPath = ui.txtSetScaler->text();
        QString tplPath = ui.txtSetTemplates->text();

        // 1. Cargar SVM
        if (!svmClf_->IsLoaded()) {
            if (!QFile::exists(modelPath)) {
                QMessageBox::warning(this, "Error Configuración", "El archivo de modelo especificado en Ajustes no existe:\n" + modelPath);
                ui.chkLiveClass->setChecked(false);
                return;
            }
            // Convertir QString a std::string para tu clase Clasificador
            svmClf_->Load(modelPath.toStdString(), scalerPath.toStdString());
        }

        // 2. Cargar Templates (Si cambiaron la ruta, reinicializamos el objeto)
        if (orientTemplatesDir_ != tplPath) {
            orientTemplatesDir_ = tplPath; // Guardamos la nueva ruta
            // Re-creamos el clasificador con la nueva ruta
            orientClf_ = std::make_unique<ClasificadorOrientacion>(orientTemplatesDir_.toStdString(), 128);
            orientTemplatesLoaded_ = false;
        }
        if (!orientTemplatesLoaded_) {
            EnsureOrientTemplatesLoaded();
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

//----------------------------------------------------------------------------
// Logica de Captura y Análisis Offline
//----------------------------------------------------------------------------
void ProyectoPSM::CapturarYAnalizar()
{
    // 1. Verificar imagen
    if (LastImage.empty()) {
        QMessageBox::warning(this, "Error", "No hay imagen en vivo para capturar.");
        return;
    }

    // 2. Congelar imagen actual
    CapturedImage = LastImage.clone();
    lastResultados_.clear();

    // 3. Cambiar a la pestaña de Análisis
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // 4. Limpiar visualización anterior para evitar confusión
    ui.lblOfflineMain->clear();
    ui.lblOfflineThumb1->clear(); ui.lblOfflineThumb2->clear(); ui.lblOfflineThumb3->clear();

    // 5. Mostrar la imagen capturada tal cual
    DisplayMat(ui.lblOfflineMain, CapturedImage);
}

void ProyectoPSM::CargarImagenDisco()
{
    // 1. Abrir diálogo
    static QString startDir = "../../Database";
    QString fileName = QFileDialog::getOpenFileName(this, tr("Abrir Imagen"), startDir, tr("Images (*.png *.jpg *.bmp);;All (*)"));
    if (fileName.isEmpty()) return;

    // 2. Cargar con QFile
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

	// 4. Limpiar segmentación previa
    lastResultados_.clear();

    // 5. Asegurar que estamos en la pestaña correcta
    ui.tabWidget->setCurrentWidget(ui.tabAnalysis);

    // 6. Limpiar y mostrar imagen cruda
    ui.lblOfflineMain->clear();
    ui.lblOfflineThumb1->clear(); ui.lblOfflineThumb2->clear(); ui.lblOfflineThumb3->clear();
    DisplayMat(ui.lblOfflineMain, CapturedImage);
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

//Funcion principal de segmentación offline 
void ProyectoPSM::ProcesarImagenOffline(const cv::Mat& img)
{
    if (img.empty()) return;

    ui.lblOfflineMain->setText("Procesando...");

    // Actualizar tamaños de las labels sin que el usuario lo note
    this->setUpdatesEnabled(false);

    ui.tabWidgetAnalysis->setCurrentWidget(ui.subTabDebugSeg);
    QApplication::processEvents();

    // Recorrer sub-pestañas
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

	// Rellenar las pestañas de debug
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

// Funcion principal de clasificación offline
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

    // 2. Carga de SVM si no está cargado
    if (!svmClf_->IsLoaded()) {
        std::string pathModel = ui.txtSetModel->text().toStdString();
        std::string pathScaler = ui.txtSetScaler->text().toStdString();

        if (!QFile::exists(QString::fromStdString(pathModel))) {
            QMessageBox::warning(this, "Error", "Configura la ruta del modelo en la pestaña Ajustes.");
            return;
        }

        bool ok = svmClf_->Load(pathModel, pathScaler);
        if (!ok) {
            QMessageBox::warning(this, "Error Crítico", "No se pudo cargar el modelo SVM.\nVerifica las rutas en Ajustes.");
            return;
        }
    }

    // 3. Convertir Mat (BGR) a formato compatible con Qt (RGB)
    cv::Mat rgbMat;
    cv::cvtColor(CapturedImage, rgbMat, cv::COLOR_BGR2RGB);

    // 4. Crear una QImage sobre la que pintaremos
    // Hacemos .copy() para tener una copia profunda y poder modificarla sin tocar la original
    QImage displayImg = QImage(rgbMat.data, rgbMat.cols, rgbMat.rows,
        static_cast<int>(rgbMat.step), QImage::Format_RGB888).copy();

    // 5. Iniciar el pintor
    QPainter p(&displayImg);

    // 6. Configurar fuente dinámica según tamaño de imagen
    QFont font = p.font();
    int pixelSize = std::max<double>(12, displayImg.width() / 40); // Ajusta el divisor para cambiar tamaño
    font.setPixelSize(pixelSize);
    font.setBold(true);
    p.setFont(font);

    // 7. Limpiamos miniaturas
    QLabel* thumbs[] = { ui.lblOfflineThumb1, ui.lblOfflineThumb2, ui.lblOfflineThumb3 };
    for (int k = 0; k < 3; ++k) thumbs[k]->clear();

    for (size_t i = 0; i < lastResultados_.size(); ++i) {
        ResultadoPieza& res = lastResultados_[i];
        if (res.imagenRecortada.empty()) continue;

        std::string codigoPieza = "";
        int clasePredicha = -1;
        bool svmExito = false;

        // --- Paso A: SVM ---
        if (svmClf_->IsLoaded()) {
            clasePredicha = svmClf_->Predict(res.imagenRecortada);
            if (clasePredicha > 0) {
                codigoPieza = std::to_string(clasePredicha);
                if (clasePredicha < 10) codigoPieza = "0" + codigoPieza;
                res.id = clasePredicha;
                svmExito = true;
            }
        }

        // --- Paso B: ORIENTACIÓN ---
        QString labelInfo = "Desc.";

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

        // --- Parte de visualizacion ---

        // 1. Miniatura
        if (i < 3) {
            DisplayMat(thumbs[i], res.imagenRecortada);
        }

        // 2. Dibujar sobre la imagen principal usando QPainter

        // Convertir coordenadas
        int x = res.boundingBox.x;
        int y = res.boundingBox.y;
        int w = res.boundingBox.width;
        int h = res.boundingBox.height;

        // Dibujar rectángulo verde
        QPen pen(Qt::green);
        pen.setWidth(3);
        p.setPen(pen);
        p.drawRect(x, y, w, h);

        // Calcular métricas para el fondo y posición
        QFontMetrics fm(font);
        int tw = fm.horizontalAdvance(labelInfo);
        int th = fm.height();
        int padding = 4;

        int labelW = tw + padding;
        int labelH = th + padding;

        // Posición inicial
        int labelX = x;
        int labelY = y - labelH;


        // Si la etiqueta se sale por la derecha
        if (labelX + labelW > displayImg.width()) {
            labelX = displayImg.width() - labelW;
        }
        if (labelX < 0) labelX = 0;

        // Si la etiqueta se sale por arriba
        if (labelY < 0) {
            labelY = y;
        }

        // Dibujar fondo negro
        p.fillRect(labelX, labelY, labelW, labelH, QColor(0, 0, 0, 150));

        // Dibujar texto encima
        p.setPen(Qt::green);
        p.drawText(labelX + 2, labelY + th, labelInfo);
    }

    p.end();

    // Mostrar resultado final en el Label
    ui.lblOfflineMain->setPixmap(QPixmap::fromImage(displayImg)
        .scaled(ui.lblOfflineMain->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}


//----------------------------------------------------------------------------
// Funciones para actulizar imagenes y mostrar graficas en la interfaz grafica
//----------------------------------------------------------------------------
void ProyectoPSM::UpdateFileNameLabel()
{
    int idx = ui.boxImageNumber->value();
    QString fileName;

    if (ui.chkUseDbNames->isChecked()) {
        // MODO DATABASE: Usa la lista NameList
        if (idx > 0 && idx <= NameList.size()) {
            fileName = QString::fromStdString(NameList[idx - 1]);
        }
        else {
            fileName = "[Fuera de Rango]";
        }
    }
    else {
        // MODO GENÉRICO: Usa Imagen_XX
        fileName = QString("Imagen_%1").arg(idx, 2, 10, QChar('0'));
    }

    ui.lblImageName->setText("Nombre: " + fileName + ".jpg");
}

void ProyectoPSM::SaveImageAs()
{
    if (CapturedImage.empty()) return;

    // 1. Abrir diálogo obligando a extensión .jpg
    QString fileName = QFileDialog::getSaveFileName(this, tr("Guardar Imagen"), "", tr("JPEG Image (*.jpg);;All Files (*)"));
    if (fileName.isEmpty()) return;

    // 2. Auto-correcion: Si el usuario no escribió ".jpg", se lo ponemos nosotros
    if (!fileName.endsWith(".jpg", Qt::CaseInsensitive) && !fileName.endsWith(".jpeg", Qt::CaseInsensitive)) {
        fileName += ".jpg";
    }

    // 3. Guardado robusto (Buffer OpenCV -> QFile Qt)
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

            // Feedback Visual (Cambiar texto del botón)
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
        int idx = ui.boxImageNumber->value();
        std::string nameStr;

        // Decidir nombre según el checkbox
        if (ui.chkUseDbNames->isChecked()) {
            // Modo Database
            nameStr = (idx <= NameList.size() && idx > 0) ? NameList[idx - 1] : "captura_extra_" + std::to_string(idx);
        }
        else {
            // Modo Genérico
            QString genName = QString("Imagen_%1").arg(idx, 2, 10, QChar('0'));
            nameStr = genName.toStdString();
        }

        std::string Path = "Database/" + nameStr + ".jpg";
        cv::imwrite(Path, CapturedImage);

        ui.pbtnGuardar->setText("¡Guardado!");
        ui.pbtnGuardar->setEnabled(false);
        QTimer::singleShot(1000, [this]() {
            ui.pbtnGuardar->setText("Guardar");
            ui.pbtnGuardar->setEnabled(true);
            });

        if (idx < 9999) {
            ui.boxImageNumber->setValue(idx + 1);
        }
    }
}

void ProyectoPSM::LoadDefaultSettings()
{
    // Rutas por defecto (relativas)
    if (ui.txtSetTemplates->text().isEmpty())
        ui.txtSetTemplates->setText("Database/Templates");

    if (ui.txtSetModel->text().isEmpty())
        ui.txtSetModel->setText("Database/Models/modelM.yml");

    if (ui.txtSetScaler->text().isEmpty())
        ui.txtSetScaler->setText("Database/Models/modelM_scaler.yml");
}

void ProyectoPSM::onSetBrowseTemplates() {
    QString dir = QFileDialog::getExistingDirectory(this, "Carpeta de Templates",
        getSmartStartDir(ui.txtSetTemplates->text(), "Database/Templates"));
    if (!dir.isEmpty()) {
        ui.txtSetTemplates->setText(dir);
        // Forzamos recarga del clasificador de orientación la próxima vez que se use
        orientTemplatesLoaded_ = false;
    }
}

void ProyectoPSM::onSetBrowseModel() {
    QString file = QFileDialog::getOpenFileName(this, "Seleccionar Modelo SVM",
        getSmartStartDir(ui.txtSetModel->text(), "Database/Models"),
        "YAML Files (*.yml *.yaml)");
    if (!file.isEmpty()) {
        ui.txtSetModel->setText(file);

        // Auto-detectar scaler: Si seleccionan "model.yml", buscamos "model_scaler.yml"
        QFileInfo info(file);
        QString scalerName = info.absolutePath() + "/" + info.baseName() + "_scaler.yml";
        if (QFile::exists(scalerName)) {
            ui.txtSetScaler->setText(scalerName);
        }

		EnsureOrientTemplatesLoaded();
    }
}

void ProyectoPSM::onSetBrowseScaler() {
    QString file = QFileDialog::getOpenFileName(this, "Seleccionar Scaler",
        getSmartStartDir(ui.txtSetScaler->text(), "Database/Models"),
        "YAML Files (*.yml *.yaml)");
    if (!file.isEmpty()) ui.txtSetScaler->setText(file);
}

bool ProyectoPSM::EnsureOrientTemplatesLoaded()
{
    if (orientTemplatesLoaded_) return true;

    // Actualizar ruta desde UI antes de cargar
    QString currentUiPath = ui.txtSetTemplates->text();
    if (!currentUiPath.isEmpty() && orientTemplatesDir_ != currentUiPath) {
        orientTemplatesDir_ = currentUiPath;
        orientClf_ = std::make_unique<ClasificadorOrientacion>(orientTemplatesDir_.toStdString(), 128);
    }

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

void ProyectoPSM::onLoadEvaluationFile()
{
    // 1. Abrir archivo
    QString startDir = getSmartStartDir(ui.txtPathEval->text(), "Database/Models");
    QString fileName = QFileDialog::getOpenFileName(this, "Abrir Archivo de Evaluación",
        startDir, "Text Files (*.txt *.csv);;All Files (*)");
    if (fileName.isEmpty()) return;

    ui.txtPathEval->setText(fileName);

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Error", "No se pudo abrir el archivo.");
        return;
    }

    // 2. Parseae (Ajustando índices de 1-12 a 0-11)
    QVector<int> trueLabels;
    QVector<int> predLabels;
    QTextStream in(&file);

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#") || line.startsWith("filename")) continue;

        QStringList parts = line.split(',');
        if (parts.size() >= 3) {
            bool ok1, ok2;
            // Restamos 1 para pasar de rango 1..12 a 0..11
            int t = parts[1].toInt(&ok1) - 1;
            int p = parts[2].toInt(&ok2) - 1;

            // Ignoramos negativos si los hubiera
            if (ok1 && ok2 && t >= 0 && p >= 0) { 
                trueLabels.push_back(t);
                predLabels.push_back(p);
            }
        }
    }
    file.close();

    if (trueLabels.isEmpty()) {
        QMessageBox::warning(this, "Error", "No se encontraron datos válidos.");
        return;
    }

    // 3. Definir las 12 Clases Fijas ("01", "02"... "12")
    QStringList qtClassNames;
    for (int i = 1; i <= 12; ++i) {
        // arg(valor, ancho, base, relleno) -> Genera 01, 02, 03...
        qtClassNames << QString("%1").arg(i, 2, 10, QChar('0'));
    }

    // 4. Configurar Visualizador
    ClassificationVisualizer visualizer;

    // Calcular tamaño: Usamos el tamaño del Label o un mínimo de 800px para que se vea nítido
    int w = ui.lblConfusionMatrix->width();
    int h = ui.lblConfusionMatrix->height();
    int imgSize = std::max<double>(800, std::min<double>(w, h)); // Forzamos alta resolución

    QImage matrixImg = visualizer.generateConfusionMatrix(trueLabels, predLabels, qtClassNames, imgSize);

    if (matrixImg.isNull()) return;

    // 5. Mostrar (Escalado suave)
    ui.lblConfusionMatrix->setPixmap(QPixmap::fromImage(matrixImg).scaled(
        ui.lblConfusionMatrix->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ProyectoPSM::onSaveConfusionMatrix()
{
    // Obtener el objeto por valor, no por puntero
    QPixmap pix = ui.lblConfusionMatrix->pixmap();

    // Comprobación de nulidad
    if (pix.isNull()) {
        QMessageBox::warning(this, "Aviso", "No hay ninguna matriz generada para guardar.");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this, "Guardar Matriz",
        "ConfusionMatrix.png",
        "Images (*.png *.jpg)");

    if (fileName.isEmpty()) return;

    if (!pix.save(fileName)) {
        QMessageBox::warning(this, "Error", "No se pudo guardar la imagen.");
    }
}

void ProyectoPSM::onLoadFeaturesPlot()
{
    QString startDir = getSmartStartDir(ui.txtPathFeaturesPlot->text(), "Database/Models");
    QString fileName = QFileDialog::getOpenFileName(this, "Cargar Features para Gráfico",
        startDir, "YAML Files (*.yml *.yaml);;All Files (*)");
    if (fileName.isEmpty()) return;

    ui.txtPathFeaturesPlot->setText(fileName);

    // Cargar con OpenCV
    try {
        cv::FileStorage fs(fileName.toLocal8Bit().constData(), cv::FileStorage::READ);

        if (!fs.isOpened()) {
            QMessageBox::warning(this, "Error de Apertura",
                "No se pudo abrir el archivo YAML.\n"
                "Verifica que la ruta no tenga caracteres extraños o que el archivo exista:\n" + fileName);
            return;
        }

        // Leer matrices
        fs["samples"] >> m_featuresLoaded;
        cv::Mat labelsMat;
        fs["responses"] >> labelsMat;

        if (labelsMat.empty()) fs["labels"] >> labelsMat;

        fs.release();

        // Validaciones
        if (m_featuresLoaded.empty() || labelsMat.empty()) {
            QMessageBox::warning(this, "Error Datos", "El archivo se abrió pero no contiene matrices 'samples' o 'responses' válidas.");
            m_featuresReady = false;
            return;
        }

        // Convertir labels a vector<int> estándar
        m_labelsLoaded.clear();
        if (labelsMat.rows > 0) {
            // Asegurar que sean enteros
            if (labelsMat.type() != CV_32S) labelsMat.convertTo(labelsMat, CV_32S);

            for (int i = 0; i < labelsMat.rows; ++i) {
                int l = labelsMat.at<int>(i, 0);
                // Si detectamos rango 1..12, restamos 1 para que sea 0..11
                if (l >= 1 && l <= 12) l -= 1;
                m_labelsLoaded.push_back(l);
            }
        }

        // Validar dimensiones
        if (m_featuresLoaded.rows != (int)m_labelsLoaded.size()) {
            QMessageBox::warning(this, "Error", "El número de muestras y etiquetas no coincide.");
            m_featuresReady = false;
            return;
        }

        m_featuresReady = true;
        QMessageBox::information(this, "Éxito", QString("Cargadas %1 muestras correctamente.").arg(m_featuresLoaded.rows));

    }
    catch (const cv::Exception& e) {
        QMessageBox::critical(this, "Excepción OpenCV", QString::fromStdString(e.what()));
        m_featuresReady = false;
    }
}

void ProyectoPSM::onScatterModeChanged()
{
    // Bloquear señales para evitar recargas visuales
    ui.cboX->blockSignals(true);
    ui.cboY->blockSignals(true);
    ui.cboX->clear();
    ui.cboY->clear();

    if (ui.rbPCAGlobal->isChecked()) {
        // Global: Desactivar todo
        ui.cboX->setEnabled(false);
        ui.cboY->setEnabled(false);
        ui.lblX->setText("Clase A:");
        ui.lblY->setText("Clase B:");
        ui.lblVs->setText("VS");
    }
    else if (ui.rbPCACompare->isChecked()) {
        // Comparar Clases: Rellenar con Clases (01..12)
        ui.cboX->setEnabled(true);
        ui.cboY->setEnabled(true);
        ui.lblX->setText("Clase A:");
        ui.lblY->setText("Clase B:");
        ui.lblVs->setText("VS");

        for (int i = 1; i <= 12; ++i) {
            QString name = QString("%1").arg(i, 2, 10, QChar('0'));
            ui.cboX->addItem(name, i - 1); // Data = indice clase (0..11)
            ui.cboY->addItem(name, i - 1);
        }
        // Seleccionar distintos por defecto
        if (ui.cboY->count() > 1) ui.cboY->setCurrentIndex(1);
    }
    else if (ui.rbManualFeat->isChecked()) {
        // Manual: Rellenar con Características
        ui.cboX->setEnabled(true);
        ui.cboY->setEnabled(true);
        ui.lblX->setText("Eje X:");
        ui.lblY->setText("Eje Y:");
        ui.lblVs->setText("vs"); 

        for (int i = 0; i < m_featureNames.size(); ++i) {
            ui.cboX->addItem(m_featureNames[i], i); 
            ui.cboY->addItem(m_featureNames[i], i);
        }
        // Seleccionar 2 características típicas por defecto (ej: Matiz vs Circularidad)
        // H_mean_circ es index 6, Circularity es index 5
        if (ui.cboX->count() > 6) {
            ui.cboX->setCurrentIndex(6); // H
            ui.cboY->setCurrentIndex(5); // Circularity
        }
    }

    ui.cboX->blockSignals(false);
    ui.cboY->blockSignals(false);
}

void ProyectoPSM::onGenerateScatter()
{
    if (!m_featuresReady) {
        QMessageBox::warning(this, "Aviso", "Primero carga un archivo de features válido.");
        return;
    }

    cv::Mat dataToShow;
    std::vector<int> labelsToShow;
    QStringList classNames; // Para la leyenda
    for (int i = 1; i <= 12; ++i) classNames << QString("%1").arg(i, 2, 10, QChar('0'));

    bool usePCA = true;
    QStringList axisLabels;

    // --- Opción 1: GLOBAL 
    if (ui.rbPCAGlobal->isChecked()) {
        dataToShow = m_featuresLoaded;
        labelsToShow = m_labelsLoaded;
        usePCA = true;
        axisLabels << "Componente Principal 1" << "Componente Principal 2";
    }
    // --- Opción 2: COMPARAR CLASES 
    else if (ui.rbPCACompare->isChecked()) {
        int classA = ui.cboX->currentData().toInt();
        int classB = ui.cboY->currentData().toInt();

        if (classA == classB) {
            QMessageBox::warning(this, "Aviso", "Selecciona dos clases distintas.");
            return;
        }

        // Filtrar datos
        for (int i = 0; i < m_featuresLoaded.rows; ++i) {
            int l = m_labelsLoaded[i];
            if (l == classA || l == classB) {
                dataToShow.push_back(m_featuresLoaded.row(i));
                labelsToShow.push_back(l);
            }
        }

        if (dataToShow.empty()) {
            QMessageBox::warning(this, "Error", "No hay datos para esas clases."); return;
        }

        usePCA = true;
        axisLabels << "PC1 (Discriminante)" << "PC2";
    }
    // --- OPCIÓN 3: Manual (Caracteristicas) 
    else if (ui.rbManualFeat->isChecked()) {
        int colX = ui.cboX->currentData().toInt(); // Índice de columna 0..11
        int colY = ui.cboY->currentData().toInt(); // Índice de columna 0..11

        // Validar rango
        if (colX < 0 || colX >= m_featuresLoaded.cols || colY < 0 || colY >= m_featuresLoaded.cols) {
            QMessageBox::warning(this, "Error", "Índice de característica inválido.");
            return;
        }

        // Crear matriz de Nx2 copiando las columnas seleccionadas
        dataToShow.create(m_featuresLoaded.rows, 2, m_featuresLoaded.type());

        m_featuresLoaded.col(colX).copyTo(dataToShow.col(0));
        m_featuresLoaded.col(colY).copyTo(dataToShow.col(1));

        labelsToShow = m_labelsLoaded; // Usamos todos los puntos
        usePCA = false; // No hacemos PCA, pintamos directo

        // Nombres para los ejes
        axisLabels << m_featureNames.value(colX, "Eje X") << m_featureNames.value(colY, "Eje Y");
    }

    // Generar Gráfico
    ClassificationVisualizer visualizer;
    int w = ui.lblScatterPlot->width();
    int h = ui.lblScatterPlot->height();
    int size = std::max<double>(600, std::min<double>(w, h));

    QImage result = visualizer.generateScatterPlot(dataToShow, labelsToShow, classNames, size, usePCA, axisLabels);

    if (!result.isNull()) {
        ui.lblScatterPlot->setPixmap(QPixmap::fromImage(result).scaled(
            ui.lblScatterPlot->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    else {
        ui.lblScatterPlot->setText("Error al generar gráfico.");
    }
}

void ProyectoPSM::onSaveScatter()
{
    QPixmap pix = ui.lblScatterPlot->pixmap();
    if (pix.isNull()) return;

    QString fileName = QFileDialog::getSaveFileName(this, "Guardar Gráfico", "ScatterPlot.png", "Images (*.png *.jpg)");
    if (!fileName.isEmpty()) pix.save(fileName);
}
