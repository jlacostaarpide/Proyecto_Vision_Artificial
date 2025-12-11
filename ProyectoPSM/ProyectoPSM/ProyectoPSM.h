#pragma once

#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <memory>

#include <QtWidgets/QMainWindow>
#include <QThread>
#include <QTimer>

#include "ui_ProyectoPSM.h"
#include "VideoAcquisition.h"
#include "NameHelper.h"

class SegmentationWorker : public QObject
{
    Q_OBJECT
public:
    explicit SegmentationWorker(QObject *parent = nullptr) : QObject(parent) {}
public slots:
    void process(std::shared_ptr<cv::Mat> snapshot, int targetW, int targetH);
signals:
    void finished(const QImage &segImage);
};

class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget *parent = nullptr);
    ~ProyectoPSM();

signals:
    void requestSegmentation(std::shared_ptr<cv::Mat> snapshot, int targetW, int targetH);

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

    // control de frecuencia de segmentación en vivo (milisegundos)
    std::chrono::steady_clock::time_point LastSegmentationTime;
    int SegmentationIntervalMs; // intervalo entre tomas (ms)

    // worker/thread para segmentación
    SegmentationWorker *segWorker = nullptr;
    QThread *segThread = nullptr;

    // timer que pide frames periódicamente para segmentar
    QTimer *segTimer = nullptr;

    // tamaño de procesamiento (ancho máximo) para acelerar la segmentación
    int SegmentationProcWidth = 320;

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

    // timer slot que pide un frame para segmentar (no bloqueante)
    void onSegmentationTimer();
};

