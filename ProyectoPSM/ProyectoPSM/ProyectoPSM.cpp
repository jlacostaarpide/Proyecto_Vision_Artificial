#include "ProyectoPSM.h"
#include <filesystem>
#include <QFileDialog>
#include "Segmentacion.h"

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
		connect(ui.pbtnAbrirImag, SIGNAL(clicked()), this, SLOT(SelectImage()));

	}
	else {
		ui.lblImagen->setText("ERROR: No se ha podido establecer comunicación con la cámara.");
		ui.pbtnCapturar->setEnabled(false);

		//Este connect es para probar que funciona qt sin tener la cámara conectada, eliminar después
		connect(ui.pbtnSegmentar, SIGNAL(clicked()), this, SLOT(SelectImage()));
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

void ProyectoPSM::NewImage(Mat Img)
{
	if (!Img.empty()) {
		LastImage = Img;
		ShowImage();
		ImageIndex++;
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


void ProyectoPSM::SelectImage()
{
	ui.tabWidget->setCurrentIndex(2);

	// Abrir diálogo de selección de fichero
	OpenPicture();
	//Hacer segmentación
	cv::Mat seg = Segmentacion::Segment(LastImage);
	if (!seg.empty()) {
		cv::Mat seg_rgb;
		cv::cvtColor(seg, seg_rgb, cv::COLOR_BGR2RGB);
		QImage qseg(reinterpret_cast<const uchar*>(seg_rgb.data), seg_rgb.cols, seg_rgb.rows, static_cast<int>(seg_rgb.step), QImage::Format_RGB888);
		ui.lblImagSegmentada->setPixmap(QPixmap::fromImage(qseg.copy()).scaled(ui.lblImagSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
		ui.lblImagSegmentada->setAlignment(Qt::AlignCenter);
	}

}

void ProyectoPSM::OpenPicture()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image Files (*.png *.jpg *.bmp);;All Files (*)"));
	if (fileName.isEmpty()) return;

	qDebug() << "SelectImage: ruta seleccionada:" << fileName;

	// Comprobar existencia con QFile (maneja unicode correctamente)
	if (!QFile::exists(fileName)) {
		qDebug() << "SelectImage: fichero no existe según QFile()";
		return;
	}

	// Leer binario con QFile y decodificar con OpenCV (evita problemas de codificación de ruta)
	QFile f(fileName);
	if (!f.open(QIODevice::ReadOnly)) {
		qDebug() << "SelectImage: no se puede abrir el fichero con QFile()";
		return;
	}
	QByteArray fileData = f.readAll();
	f.close();

	std::vector<uchar> vec(fileData.begin(), fileData.end());
	cv::Mat image = cv::imdecode(vec, cv::IMREAD_COLOR); // devuelve BGR
	if (image.empty()) {
		qDebug() << "SelectImage: imdecode falló. Intentando QImage como fallback.";
	}

	qDebug() << "SelectImage: imagen decodificada:" << image.cols << "x" << image.rows << " channels:" << image.channels();

	// Convertir BGR -> RGB para QImage
	cv::Mat rgb;
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

	QImage qimg(reinterpret_cast<const uchar*>(rgb.data), rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
	QPixmap pix = QPixmap::fromImage(qimg.copy()); // copy() asegura memoria propia

	if (pix.isNull()) {
		qDebug() << "SelectImage: QPixmap nulo tras la conversion.";
		return;
	}

	QPixmap scaled = pix.scaled(ui.lblImagNoSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	ui.lblImagNoSegmentada->setPixmap(scaled);
	ui.lblImagNoSegmentada->setAlignment(Qt::AlignCenter);

	// Guardar para uso posterior
	LastImage = image.clone();
}