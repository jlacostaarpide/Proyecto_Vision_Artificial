#include "ProyectoPSM.h"
#include <filesystem>
#include <QFileDialog>
#include "Segmentacion.h"
#include <chrono>
#include <QMetaType>
#include <QDebug>

Q_DECLARE_METATYPE(std::shared_ptr<cv::Mat>)

void SegmentationWorker::process(std::shared_ptr<cv::Mat> snapshotPtr, int targetW, int targetH)
{
    QImage qseg; // por defecto imagen vacía (se interpreta como no resultado)
    try {
        if (!snapshotPtr || snapshotPtr->empty()) {
            emit finished(qseg);
            return;
        }

        // Segmentación sobre la imagen recibida (ya reducida a tamaño de procesamiento)
        cv::Mat seg = Segmentacion::Segment(*snapshotPtr);

        if (!seg.empty()) {
            // ajustar tamaño para mostrar en la UI si hace falta
            cv::Mat seg_for_q;
            if (targetW > 0 && targetH > 0 && (seg.cols != targetW || seg.rows != targetH)) {
                cv::resize(seg, seg_for_q, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);
            } else {
                seg_for_q = seg;
            }

            // convertir a RGB y construir QImage (hacer .copy() para asegurar que los datos sean independientes)
            cv::Mat seg_rgb;
            if (seg_for_q.channels() == 3) {
                cv::cvtColor(seg_for_q, seg_rgb, cv::COLOR_BGR2RGB);
                qseg = QImage(reinterpret_cast<const uchar*>(seg_rgb.data),
                              seg_rgb.cols, seg_rgb.rows,
                              static_cast<int>(seg_rgb.step),
                              QImage::Format_RGB888).copy();
            } else if (seg_for_q.channels() == 1) {
                cv::Mat tmp;
                cv::cvtColor(seg_for_q, tmp, cv::COLOR_GRAY2RGB);
                qseg = QImage(reinterpret_cast<const uchar*>(tmp.data),
                              tmp.cols, tmp.rows,
                              static_cast<int>(tmp.step),
                              QImage::Format_RGB888).copy();
            } else {
                // formatos exóticos: devolver vacío
            }
        }
    }
    catch (const std::exception &e) {
        qDebug() << "SegmentationWorker exception:" << e.what();
    }
    catch (...) {
        qDebug() << "SegmentationWorker unknown exception";
    }

    emit finished(qseg);
}

ProyectoPSM::ProyectoPSM(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    // registrar tipo para queued connections
    qRegisterMetaType<std::shared_ptr<cv::Mat>>("std::shared_ptr<cv::Mat>");

	if (!std::filesystem::exists("Database")) {
		std::filesystem::create_directory("Database");
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

    // intervalo recomendado: por ejemplo 2000 ms (2 s). Ajusta según tus necesidades.
    SegmentationIntervalMs = 2000;
    LastSegmentationTime = std::chrono::steady_clock::now() - std::chrono::milliseconds(SegmentationIntervalMs);

    // crear worker y thread para segmentación (reutilizable)
    segWorker = new SegmentationWorker();
    segThread = new QThread(this);
    segWorker->moveToThread(segThread);
    connect(segThread, &QThread::finished, segWorker, &QObject::deleteLater);
    // cuando el worker termine, actualizar UI (queued)
    connect(segWorker, &SegmentationWorker::finished, this, &ProyectoPSM::UpdateSegmentationUI, Qt::QueuedConnection);
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
        segWorker = nullptr; // será borrado por finished->deleteLater()
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

void ProyectoPSM::ShowImage()
{
	if (!LastImage.empty() and (ui.pbtnCapturar->isEnabled())) {
		ui.lblImagen->setPixmap(QPixmap::fromImage(QImage(LastImage.data, LastImage.cols, LastImage.rows, LastImage.step, QImage::Format_BGR888)));
	}
}

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

void ProyectoPSM::SaveImage()
{
	if (!CapturedImage.empty()) {
		//string Name = "prueba_" + to_string(SavedImageIndex);
		string Name = NameList[SavedImageIndex-1];
		string Path = "Database//" + Name + ".jpg";
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
    // Actualizar sólo la imagen en pantalla (sin lanzar segmentación aquí)
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

    // marcar como ocupado
    SegProcessing = true;

    // crear snapshot reducido para acelerar la segmentación
    cv::Mat proc;
    const int srcW = LastImage.cols;
    const int srcH = LastImage.rows;
    int outW = min(SegmentationProcWidth, srcW);
    int outH = static_cast<int>((double)outW * srcH / srcW);
    if (outW <= 0 || outH <= 0) {
        // fallback a copia completa si algo raro
        proc = LastImage.clone();
    } else {
        cv::resize(LastImage, proc, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);
    }

    // preparar target de visualización (tamaño del QLabel)
    const int targetW = ui.lblImagSegmentada->width();
    const int targetH = ui.lblImagSegmentada->height();

    // empaquetar y emitir trabajo (queued connection)
    auto snapshotPtr = std::make_shared<cv::Mat>(std::move(proc));
    emit requestSegmentation(snapshotPtr, targetW, targetH);
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
	}
}

// slot llamado en hilo GUI para actualizar la imagen segmentada
void ProyectoPSM::UpdateSegmentationUI(const QImage& segImage)
{
	if (segImage.isNull()) {
		// no hay resultado de segmentación
		ui.lblImagSegmentada->clear();
	}
	else {
		QPixmap pix = QPixmap::fromImage(segImage);
		ui.lblImagSegmentada->setPixmap(pix.scaled(ui.lblImagSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		ui.lblImagSegmentada->setAlignment(Qt::AlignCenter);
	}

    // liberar flag de procesamiento para permitir la siguiente toma
    SegProcessing = false;
}