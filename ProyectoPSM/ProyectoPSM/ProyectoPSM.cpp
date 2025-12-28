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

    // Registrar tipos para señales
    qRegisterMetaType<shared_ptr<Mat>>("std::shared_ptr<cv::Mat>");
    qRegisterMetaType<std::vector<QRectF>>("std::vector<QRectF>");
    qRegisterMetaType<std::vector<QImage>>("std::vector<QImage>");

    // Crear carpeta DB
    if (!filesystem::exists("Database")) {
        filesystem::create_directory("Database");
    }

    // Inicializar lógica
    NameList = NameHelper::GenerarNombres();
    LiveSegmentationEnabled = false;
    SegProcessing = false;
    segInFlight = 0;
    SegmentationIntervalMs = 40;
    LastSegmentationTime = chrono::steady_clock::now() - chrono::milliseconds(SegmentationIntervalMs);

    // Inicializar Cámara
    Camera = new CVideoAcquisition();

    // Configurar Worker Thread
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);

    // Conexiones Worker
    connect(segWorker, &SegmentationWorker::finishedResult, this, &ProyectoPSM::UpdateSegmentationResults, Qt::QueuedConnection);
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);
    segThread->start();

    // Timer segmentación en vivo
    segTimer = new QTimer(this);
    segTimer->setInterval(SegmentationIntervalMs);
    connect(segTimer, &QTimer::timeout, this, &ProyectoPSM::onSegmentationTimer);
    segTimer->start();

    // ---------------------------------------------------------
    // CONEXIONES DE INTERFAZ
    // ---------------------------------------------------------

    // Tab "En Vivo"
    connect(ui.pbtnEncender, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
    connect(ui.chkLiveSeg, SIGNAL(toggled(bool)), this, SLOT(EnableLiveSegmentation(bool)));
    connect(ui.btnCapturarAnalizar, SIGNAL(clicked()), this, SLOT(CapturarYAnalizar()));

    // Tab "Análisis"
    connect(ui.btnCargarDisco, SIGNAL(clicked()), this, SLOT(CargarImagenDisco()));
    connect(ui.btnRecalcSeg, SIGNAL(clicked()), this, SLOT(RecalcularSegmentacion()));
    connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));

    // Inicializar estados de botones de análisis
    ui.pbtnGuardar->setEnabled(false);

    // ESTADO DE LA CÁMARA
    if (Camera->CameraOK) {
        // Cámara detectada
        ui.lblStatusCamara->setText("Estado: Conectada (Lista)");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: green;");

        ui.pbtnEncender->setEnabled(true);
        ui.btnReconectar->setEnabled(false);
        ui.btnCapturarAnalizar->setEnabled(false); // Desactivado hasta que se encienda

        // Conectar señal de nueva imagen
        connect(ui.pbtnEncender, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
        connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));

        Camera->SetCameraAutoExposure();
    }
    else {
        // Cámara NO detectada
        ui.lblStatusCamara->setText("Estado: DESCONECTADA");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: red;");

        ui.pbtnEncender->setEnabled(false);      // Botón inusable
        ui.pbtnEncender->setText("No Disponible");
        ui.btnCapturarAnalizar->setEnabled(false);
        ui.btnReconectar->setEnabled(true);     // Permitir intentar reconectar (futuro)
    }

    ImageIndex = 0;
    SavedImageIndex = 1;
    ui.boxImageNumber->setValue(SavedImageIndex);
}

ProyectoPSM::~ProyectoPSM()
{
    if (segThread) {
        segThread->quit();
        segThread->wait();
    }
}


// GESTIÓN DE CÁMARA Y VIVO
void ProyectoPSM::EnableButtons(bool StartCapture)
{
    if (StartCapture) {
        // Se ha encendido
        ui.pbtnEncender->setText("Apagar Cámara");
        ui.btnCapturarAnalizar->setEnabled(true);
        ui.lblStatusCamara->setText("Estado: CAPTURANDO");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: blue;");
        ui.btnReconectar->setEnabled(false);
    }
    else {
        // Se ha apagado
        ui.pbtnEncender->setText("Encender Cámara");
        ui.btnCapturarAnalizar->setEnabled(false);
        ui.lblStatusCamara->setText("Estado: Conectada (Standby)");
        ui.lblStatusCamara->setStyleSheet("font-weight: bold; color: green;");
        ui.lblVideoLive->clear();
        ui.lblVideoLive->setText("Cámara Pausada");
        ui.btnReconectar->setEnabled(true);
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

    // DIBUJAR CAJAS VERDES + NÚMEROS (Segmentación en vivo)
    if (LiveSegmentationEnabled && !lastBoxesNormalized.empty()) {
        QPainter p(&scaled);

        // Configurar pincel para las cajas
        QPen penBox(Qt::green);
        penBox.setWidth(3);
        p.setPen(penBox);

        // Configurar fuente para los números
        QFont font = p.font();
        // Tamaño dinámico pero legible (min 14px)
        font.setPixelSize(std::max<double>(12, scaled.height() / 25));
        font.setBold(true);
        p.setFont(font);

        // Usamos un índice 'i' para saber qué número pintar
        for (size_t i = 0; i < lastBoxesNormalized.size(); ++i) {
            const auto& boxNorm = lastBoxesNormalized[i];

            // Desnormalizar coordenadas
            int x = static_cast<int>(boxNorm.x() * scaled.width());
            int y = static_cast<int>(boxNorm.y() * scaled.height());
            int w = static_cast<int>(boxNorm.width() * scaled.width());
            int h = static_cast<int>(boxNorm.height() * scaled.height());

            // 1. Dibujar Rectángulo
            p.drawRect(x, y, w, h);

            // 2. Dibujar Número (ID = i + 1)
            QString text = QString::number(i + 1);

            // Calcular posición del texto (encima de la esquina izquierda)
            int textY = y - 5;
            // Si la pieza está muy arriba y el texto se sale, lo ponemos dentro
            if (textY < font.pixelSize()) {
                textY = y + font.pixelSize() + 5;
            }

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

    // 3. Parar segmentación en vivo para ahorrar recursos y evitar conflictos
    if (ui.chkLiveSeg->isChecked()) {
        ui.chkLiveSeg->setChecked(false); // Esto dispara EnableLiveSegmentation(false)
    }

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
    // Clonamos para pintar encima sin estropear la imagen que guardaremos en disco
    cv::Mat displayImg = img.clone();

    // Dibujamos los recuadros sobre la imagen completa
    for (const auto& res : resultados) {
        // Rectángulo verde (BGR: 0, 255, 0), grosor 3
        cv::rectangle(displayImg, res.boundingBox, cv::Scalar(0, 255, 0), 3);

        // Opcional: Escribir el ID encima de la pieza
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
                // Dejamos un texto gris indicando hueco vacío
                thumbs[i]->setText("---");
            }
        }
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

        // Feedback visual (cambiar texto del botón momentáneamente)
        ui.pbtnGuardar->setText("¡Guardado!");
        ui.pbtnGuardar->setEnabled(false); // Evitar doble click rápido
        QTimer::singleShot(1000, [this]() {
            ui.pbtnGuardar->setText("Guardar Imagen");
            ui.pbtnGuardar->setEnabled(true);
            });

        // Avanzar índice
        if (idx < 9999) ui.boxImageNumber->setValue(idx + 1);
    }
}