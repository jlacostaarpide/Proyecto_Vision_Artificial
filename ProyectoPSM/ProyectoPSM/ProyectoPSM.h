
#include <vector>
#include <string>
#include <atomic>
#include <chrono>
#include <memory>

#include <QtWidgets/QMainWindow>
#include <QThread>
#include <QTimer>
#include <QRectF>
#include <QString>

#include "ui_ProyectoPSM.h"
#include "VideoAcquisition.h"
#include "NameHelper.h"

class SegmentationWorker : public QObject
{
    Q_OBJECT
public:
    explicit SegmentationWorker(QObject *parent = nullptr) : QObject(parent) {}
public slots:
    void process(std::shared_ptr<cv::Mat> snapshot);
signals:
    // bounding box normalizado [0..1]
    void finishedBox(const QRectF &box);
    // thumbnail pequeño de la región segmentada (RGB)
    void finishedThumbnail(const QImage &thumb);
};

class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget *parent = nullptr);
    ~ProyectoPSM();

signals:
    void requestSegmentation(std::shared_ptr<cv::Mat> snapshot);

private:
    Ui::ProyectoPSMClass ui;
	CVideoAcquisition* Camera = nullptr;
    Mat LastImage;
	Mat CapturedImage;
    int ImageIndex = 0;
	int SavedImageIndex = 1;
    std::vector<std::string> NameList;

    // para segmentación en vivo
    bool LiveSegmentationEnabled = false;
    std::atomic<bool> SegProcessing{false};

    std::chrono::steady_clock::time_point LastSegmentationTime;
    int SegmentationIntervalMs = 2000; // intervalo entre tomas (ms)

    // worker/thread para segmentación
    SegmentationWorker *segWorker = nullptr;
    QThread *segThread = nullptr;

	// timer para segmentar frames periodicamente
    QTimer *segTimer = nullptr;

    // tamaño de procesamiento (ancho máximo) para acelerar la segmentación
    int SegmentationProcWidth = 320;

    // último bbox normalizado calculado por el worker
    QRectF lastBoxNormalized;

    // control de thumbnails / resultados en vuelo
    std::atomic<int> segInFlight{0};
    const int maxSegInFlight = 3; // tamaño del buffer
    int segThumbNext = 0; // para saber en que label poner la miniatura

    // ruta del archivo para segmentación offline
    QString fileName;

private slots:
    void EnableButtons(bool StartCapture);
    void NewImage(Mat Img);
    void SaveImage();
	void ShowImage();
	void VisualizeImage();
	void ReturnTab();

	// control de segmentación en vivo
    void EnableLiveSegmentation(bool enabled);
    void UpdateSegmentationBox(const QRectF &box);

    // slot para recibir thumbnails desde el worker y mostrar en UI
    void EnqueueSegThumbnail(const QImage &thumb);

    // timer slot que pide un frame para segmentar (no bloqueante)
    void onSegmentationTimer();

    // selección/procesado de imagen desde fichero (offline)
    void SegmentarImagDisco();
	void SegmentarImagCapturada();
    void OpenPicture();
	void SegmentationMode(int index);
};

