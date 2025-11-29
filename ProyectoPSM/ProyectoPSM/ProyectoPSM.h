#pragma once

#include <vector>
#include <string>

#include <QtWidgets/QMainWindow>
#include "ui_ProyectoPSM.h"
#include "VideoAcquisition.h"
#include "NameHelper.h"

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
	Mat CapturedImage;
    int ImageIndex;
	int SavedImageIndex;
    std::vector<std::string> NameList;

private slots:
    void EnableButtons(bool StartCapture);
    void NewImage(Mat Img);
    void SaveImage();
	void ShowImage();
	void VisualizeImage();
	void ReturnTab();
	void SelectImage();
	void OpenPicture();
};

