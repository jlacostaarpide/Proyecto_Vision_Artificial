#include "ProyectoPSM.h"

ProyectoPSM::ProyectoPSM(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    Camera = new CVideoAcquisition();
    if (Camera->CameraOK) {
		ui.pbtnCapturar->setEnabled(true);

		ImageIndex = 0;
		Camera->SetCameraAutoExposure();

		connect(ui.pbtnCapturar, SIGNAL(toggled(bool)), this, SLOT(EnableButtons(bool)));
		connect(ui.pbtnCapturar, SIGNAL(toggled(bool)), Camera, SLOT(StartStopCapture(bool)));
		connect(ui.pbtnGuardar, SIGNAL(clicked()), this, SLOT(SaveImage()));
		connect(ui.pbtnUlt, SIGNAL(clicked()), this, SLOT(GetImage()));
		connect(ui.pbtnLimpiar, SIGNAL(clicked()), this, SLOT(ClearImage()));
		connect(Camera, SIGNAL(NewImageSignal(Mat)), this, SLOT(NewImage(Mat)));
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
	ui.pbtnCapturar->setEnabled(!StartCapture);
	ui.pbtnGuardar->setEnabled(StartCapture);
	ui.pbtnUlt->setEnabled(StartCapture);
	ui.pbtnLimpiar->setEnabled(!StartCapture);
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
	if (!LastImage.empty()) {
		ui.lblImagen->setPixmap(QPixmap::fromImage(QImage(LastImage.data, LastImage.cols, LastImage.rows, LastImage.step, QImage::Format_BGR888)));
	}
}

void ProyectoPSM::SaveImage()
{
	if (!LastImage.empty()) {
		string Filename = "Image_" + to_string(ImageIndex) + ".png";
		imwrite(Filename, LastImage);
	}
}

void ProyectoPSM::ClearImage()
{
	ui.lblImagen->clear();
}

void ProyectoPSM::GetImage()
{
	ShowImage();
}

