#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_ProyectoPSM.h"
#include "VideoAcquisition.h"

class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget *parent = nullptr);
    ~ProyectoPSM();

private:
    Ui::ProyectoPSMClass ui;
	CVideoAcquisition* Camera;
    Mat LastImage;
    int ImageIndex;
};

