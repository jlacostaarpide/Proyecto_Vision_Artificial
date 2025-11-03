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
		//ui.pbtnCapturar->setEnabled(false);
	}
}

ProyectoPSM::~ProyectoPSM()
{}

