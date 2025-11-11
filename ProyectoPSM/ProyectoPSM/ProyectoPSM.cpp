#include "ProyectoPSM.h"
#include <filesystem>

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
