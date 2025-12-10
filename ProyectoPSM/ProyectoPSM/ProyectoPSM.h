#pragma once

#include <vector>
#include <string>
#include <atomic>

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

    // para segmentación en vivo
    bool LiveSegmentationEnabled;
    std::atomic<bool> SegProcessing;

private slots:
    void EnableButtons(bool StartCapture);
    void NewImage(Mat Img);
    void SaveImage();
	void ShowImage();
	void VisualizeImage();
	void ReturnTab(); 

    // live segmentation control + UI update
    void EnableLiveSegmentation(bool enabled);
    void UpdateSegmentationUI(const QImage &segImage);
};

