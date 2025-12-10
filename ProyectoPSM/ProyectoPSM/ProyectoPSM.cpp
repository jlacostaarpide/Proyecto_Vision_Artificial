#include "ProyectoPSM.h"
#include <filesystem>
#include <QFileDialog>
#include "Segmentacion.h"
#include <thread>

ProyectoPSM::ProyectoPSM(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
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
{}

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

    // Mover cabecera de Img a LastImage para evitar una copia extra del header.
    // (La data sigue compartida hasta que se haga un clone explícito.)
    LastImage = std::move(Img);
    ShowImage();
    ++ImageIndex;

    // Si no está activada la segmentación en vivo, salir pronto.
    if (!LiveSegmentationEnabled) {
        return;
    }

    // Intentar poner el flag; si ya hay procesamiento en curso, ignorar este frame.
    bool expected = false;
    if (!SegProcessing.compare_exchange_strong(expected, true)) {
        return;
    }

    // Guardar tamaño destino de la UI ahora (no acceder a ui desde el hilo worker).
    const int targetW = ui.lblImagSegmentada->width();
    const int targetH = ui.lblImagSegmentada->height();

    // Tomar snapshot profundo (independiente) para procesar fuera del hilo GUI.
    cv::Mat snapshot = LastImage.clone();

    // Lanzar trabajo en hilo aparte (detach). Se usa try/catch para garantizar que se libere el flag.
    std::thread([this, snapshot, targetW, targetH]() mutable {
        QImage qseg; // por defecto imagen vacía (se interpreta como no resultado)
        try {
            cv::Mat seg = Segmentacion::Segment(snapshot);

            if (!seg.empty()) {
                // reducir resolución antes de convertir a QImage para ahorrar CPU/memoria si hace falta
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
                    // si la segmentación devolviera máscara en 1 canal, convertir a formato RGB simple
                    cv::Mat tmp;
                    cv::cvtColor(seg_for_q, tmp, cv::COLOR_GRAY2RGB);
                    qseg = QImage(reinterpret_cast<const uchar*>(tmp.data),
                                  tmp.cols, tmp.rows,
                                  static_cast<int>(tmp.step),
                                  QImage::Format_RGB888).copy();
                } else {
                    // soporte por si hay otro número de canales: devolver vacío
                }
            }
        }
        catch (const std::exception &e) {
            qDebug() << "Segmentation thread exception:" << e.what();
            // dejar qseg vacío para indicar fallo
        }
        catch (...) {
            qDebug() << "Segmentation thread unknown exception";
        }

        // actualizar UI en hilo GUI
        QMetaObject::invokeMethod(this, "UpdateSegmentationUI", Qt::QueuedConnection, Q_ARG(QImage, qseg));

        // liberar flag de procesamiento siempre al final
        SegProcessing = false;
    }).detach();
}

// slot que activa/desactiva la segmentación en vivo
void ProyectoPSM::EnableLiveSegmentation(bool enabled)
{
	ui.tabWidget->setCurrentIndex(2);
	LiveSegmentationEnabled = enabled;
	if (enabled) {
		ui.lblImagNoSegmentada->setText(QString::fromStdString("Segmentacion en vivo ACTIVADA"));
	}
	else {
		ui.lblImagNoSegmentada->setText(QString::fromStdString("Segmentacion en vivo DESACTIVADA"));
		// limpiar el resultado en pantalla si quieres
		ui.lblImagSegmentada->clear();
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
	
}