#include "ProyectoPSM.h"
#include <filesystem>
#include <QFileDialog>
#include "Segmentacion.h"
#include <chrono>
#include <QMetaType>
#include <QDebug>
#include <QPainter>

Q_DECLARE_METATYPE(shared_ptr<Mat>)

//Recibe la imagen y obtiene el bounding box y el thumbnail segmentado
void SegmentationWorker::process(shared_ptr<Mat> snapshotPtr )
{
	// resultados por defecto
    QRectF normalizedBox(0,0,0,0); 
    QImage qthumb; 

    try {
		// comprobar entrada
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finishedBox(normalizedBox);
            emit finishedThumbnail(qthumb);
            return;
        }
        vector<ResultadoPieza> resultados = Segmentacion::Segmentar(*snapshotPtr);

        if (!resultados.empty()) {
            // Nos quedamos con la mejor
            const ResultadoPieza& pieza = resultados[0];

            // Calcular Bounding Box
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

            // Construir thumbnail pequeño a partir del CROP limpio
            Mat crop = pieza.imagenRecortada;

            if (!crop.empty()) {
                const int thumbW = 160; // Ancho fijo para la miniatura
                int srcW = crop.cols;
                int srcH = crop.rows;

                // Calcular altura proporcional
                int thumbH = max(1, (int)((double)thumbW * srcH / max(1, srcW)));

                Mat thumb;
                resize(crop, thumb, cv::Size(thumbW, thumbH), 0, 0, INTER_LINEAR);

                // Convertir a QImage RGB
                Mat thumb_rgb;
                if (thumb.channels() == 3)
                    cvtColor(thumb, thumb_rgb, COLOR_BGR2RGB);
                else
                    cvtColor(thumb, thumb_rgb, COLOR_GRAY2RGB);

                qthumb = QImage(reinterpret_cast<const uchar*>(thumb_rgb.data),
                    thumb_rgb.cols, thumb_rgb.rows,
                    static_cast<int>(thumb_rgb.step),
                    QImage::Format_RGB888).copy();
            }
        }
    }
    catch (const exception& e) {
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
    // cuando el worker calcule el bbox normalizado, actualizar UI 
    connect(segWorker, &SegmentationWorker::finishedBox, this, &ProyectoPSM::UpdateSegmentationBox, Qt::QueuedConnection);
    // cuando el worker produzca un thumbnail, encolarlo al UI
    connect(segWorker, &SegmentationWorker::finishedThumbnail, this, &ProyectoPSM::EnqueueSegThumbnail, Qt::QueuedConnection);
    // emitir trabajo al worker
    connect(this, &ProyectoPSM::requestSegmentation, segWorker, &SegmentationWorker::process, Qt::QueuedConnection);
    segThread->start();

    // crear y arrancar timer que pide frames periódicamente para segmentar
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

	}
	else {
		ui.lblImagen->setText("ERROR: No se ha podido establecer comunicación con la cámara.");
		ui.pbtnCapturar->setEnabled(false);

	}
}

ProyectoPSM::~ProyectoPSM()
{
    // Detener el hilo de segmentación de manera ordenada
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
	}
	else {
		ui.pbtnCapturar->setEnabled(StartCapture);
		ui.pbtnEncender->setText("Apagar");
	}
}

//pinta el 
void ProyectoPSM::ShowImage()
{
	if (!LastImage.empty() and (ui.pbtnCapturar->isEnabled())) {
        // construir pixmap a partir de la imagen actual
		QImage qimg(LastImage.data, LastImage.cols, LastImage.rows, LastImage.step, QImage::Format_BGR888);
		QPixmap pix = QPixmap::fromImage(qimg.copy());

        // si hay un bbox disponible se pinta encima
        if (!lastBoxNormalized.isNull() && lastBoxNormalized.width() > 0 && lastBoxNormalized.height() > 0) {
            QPainter p(&pix);
            QPen pen(Qt::green);
            pen.setWidthF(max(1.0, pix.width() * 0.005)); // grosor proporcional
            pen.setStyle(Qt::SolidLine);
            p.setPen(pen);
            p.setRenderHint(QPainter::Antialiasing, true);

            // convertir coords normalizadas [0..1] a pixmap coordinates
            int x = static_cast<int>(lastBoxNormalized.x() * pix.width());
            int y = static_cast<int>(lastBoxNormalized.y() * pix.height());
            int w = static_cast<int>(lastBoxNormalized.width() * pix.width());
            int h = static_cast<int>(lastBoxNormalized.height() * pix.height());

            // asegurar dentro de límites
            x = max(0, min(x, pix.width()-1));
            y = max(0, min(y, pix.height()-1));
            if (w <= 0) w = 1;
            if (h <= 0) h = 1;
            if (x + w > pix.width()) w = pix.width() - x;
            if (y + h > pix.height()) h = pix.height() - y;

            p.drawRect(x, y, w, h);
        }

		ui.lblImagen->setPixmap(pix);
	}
}

//muestra la imagen capturada (para la base de datos)
void ProyectoPSM::VisualizeImage()
{
	ui.tabWidget->setCurrentIndex(1);
	SavedImageIndex = ui.boxImageNumber->value();
	// Comprobar que no se pase del tamaño del vector
	string texto = "";
	if (SavedImageIndex < 1 || SavedImageIndex > static_cast<int>(NameList.size())) {
		texto = "Índice de imagen fuera de rango. Valores válidos: 1 - " + to_string(NameList.size());
		ui.pbtnGuardar->setEnabled(false);
	}
	else
	{
		texto = "Guardar siguiente imagen como: " + (NameList[SavedImageIndex - 1]);
	}
	ui.txtImageName->setText(QString::fromStdString(texto));
	if (!LastImage.empty()) {
		CapturedImage = LastImage.clone();
		ui.lblImagenCapturada->setPixmap(QPixmap::fromImage(QImage(CapturedImage.data, CapturedImage.cols, CapturedImage.rows, CapturedImage.step, QImage::Format_BGR888)));
	}
}
//guarda la imagen capturada (base de datos)
void ProyectoPSM::SaveImage()
{
	if (!CapturedImage.empty()) {
		//string Name = "prueba_" + to_string(SavedImageIndex);
		string Name = NameList[SavedImageIndex-1];
		string Path = "C:/Users/Lenovo/Desktop/Máster/1er Semestre/PSM/proyecto/ProyectoPSM/Database/test2//" + Name + ".jpg";
		imwrite(Path, CapturedImage);
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


void ProyectoPSM::NewImage(Mat Img)
{
    if (Img.empty()) {
        return;
    }
    LastImage = std::move(Img);
    ShowImage();
    ++ImageIndex;
}

// timer slot: tomar un frame cada intervalo y enviarlo al worker si está libre
void ProyectoPSM::onSegmentationTimer()
{
    if (!LiveSegmentationEnabled)
        return;
    if (SegProcessing.load())
        return; // worker ocupado, ignorar esta toma
    if (LastImage.empty())
        return;

    // intentar reservar un slot in-flight
    int prev = segInFlight.fetch_add(1, memory_order_relaxed);
    if (prev >= maxSegInFlight) {
        // estaba lleno: deshacer y salir
        segInFlight.fetch_sub(1, memory_order_relaxed);
        return;
    }

    // marcar como ocupado para evitar dobles reservas por el mismo frame
    SegProcessing = true;

	// Clonar imagen en alta resolución
    cv::Mat proc = LastImage.clone();

    // preparar target de visualización (tamaño del QLabel) - no usado por worker ahora
    const int targetW = ui.lblImagSegmentada->width();
    const int targetH = ui.lblImagSegmentada->height();

    // empaquetar y emitir trabajo (queued connection)
    auto snapshotPtr = make_shared<Mat>(std::move(proc));
    emit requestSegmentation(snapshotPtr, targetW, targetH);

    // El worker emitirá finishedBox y finishedThumbnail; EnqueueSegThumbnail liberará segInFlight.
}

// slot que activa/desactiva la segmentación en vivo
void ProyectoPSM::EnableLiveSegmentation(bool enabled)
{
	ui.tabWidget->setCurrentIndex(2);
	LiveSegmentationEnabled = enabled;
	if (enabled) {
		ui.lblImagNoSegmentada->setText(QString::fromStdString("Segmentacion en vivo ACTIVADA"));
	} else {
		ui.lblImagNoSegmentada->setText(QString::fromStdString("Segmentacion en vivo DESACTIVADA"));
		ui.lblImagSegmentada->clear();
        // reset flag por seguridad
        SegProcessing = false;
        lastBoxNormalized = QRectF();
        segInFlight.store(0);
	}
}

// slot llamado por el worker con el bbox normalizado (ejecuta en GUI)
void ProyectoPSM::UpdateSegmentationBox(const QRectF &box)
{
    // guardar bbox para dibujar encima del frame mostrado
    lastBoxNormalized = box;
    // refrescar la imagen mostrada (dibujará el rectángulo en ShowImage)
    ShowImage();

    // liberar flag de procesamiento para permitir la siguiente toma (bbox recibida)
    SegProcessing = false;
}

// recibe miniaturas desde el worker y los muestra en dos QLabel (circular)
void ProyectoPSM::EnqueueSegThumbnail(const QImage &thumb)
{
    // ejecuta en hilo GUI
    if (thumb.isNull()) {
        // liberar in-flight y salir
        segInFlight.fetch_sub(1,memory_order_relaxed);
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

    // liberar slot in-flight para permitir nuevas peticiones
    segInFlight.fetch_sub(1, memory_order_relaxed);
}