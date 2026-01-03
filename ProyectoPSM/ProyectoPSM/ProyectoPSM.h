
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
#include <memory>

#include "ui_ProyectoPSM.h"
#include "VideoAcquisition.h"
#include "NameHelper.h"

#include "Segmentacion.h"
#include "Clasificador.h"
#include "ClasificadorOrientacion.h"
#include "TrainingWorker.h"

class SegmentationWorker : public QObject
{
    Q_OBJECT
public:
    explicit SegmentationWorker(QObject *parent = nullptr) : QObject(parent) {}
public slots:
    void process(std::shared_ptr<cv::Mat> snapshot);
signals:
    void finishedResult(const std::vector<QRectF>& boxes,
        const std::vector<QImage>& thumbnails,
        const std::vector<cv::Mat>& crops);
};

class ClasificationWorker : public QObject {
    Q_OBJECT
public:
    ClasificationWorker(Clasificador* svm, ClasificadorOrientacion* orient)
        : svmClf(svm), orientClf(orient) {
    }

public slots:
    void process(std::vector<cv::Mat> crops, std::vector<QRectF> boxes);
signals:
    void finishedResult(std::vector<QRectF> boxes, std::vector<QString> labels);
private:
    Clasificador* svmClf;
    ClasificadorOrientacion* orientClf;
};

class ProyectoPSM : public QMainWindow
{
    Q_OBJECT

public:
    ProyectoPSM(QWidget* parent = nullptr);
    ~ProyectoPSM();

signals:
    void requestSegmentation(std::shared_ptr<cv::Mat> snapshot);
    void requestClassification(std::vector<cv::Mat> crops, std::vector<QRectF> boxes);

private:
    Ui::ProyectoPSMClass ui;
    CVideoAcquisition* Camera = nullptr;
    Mat LastImage;
    Mat CapturedImage;
    int ImageIndex = 0;
    int SavedImageIndex = 1;
    std::vector<std::string> NameList;

    QTimer* statusTimer = nullptr;
    void SetCameraStatusUI(bool isConnected);

    // Variables de control
    bool LiveSegmentationEnabled = false;
    std::atomic<bool> SegProcessing{ false };
    std::atomic<int> segInFlight{ 0 }; // Control de saturación

    ClasificationWorker* classWorker;
    QThread* classThread;
    std::atomic<bool> LiveClassificationEnabled;
    std::atomic<bool> ClassProcessing; // Para evitar saturación

    // Resultados para pintar en vivo
    std::vector<QRectF> lastClassBoxes;
    std::vector<QString> lastClassLabels;

    QTimer* segTimer = nullptr;
    int SegmentationIntervalMs = 40;
    int ClasificationIntervalMs = 150;
    std::chrono::steady_clock::time_point LastSegmentationTime;

    // Worker threads
    SegmentationWorker* segWorker = nullptr;
    QThread* segThread = nullptr;

    // AHORA guardamos una LISTA de cajas para dibujar
    std::vector<QRectF> lastBoxesNormalized;

    QString fileName;

    std::unique_ptr<ClasificadorOrientacion> orientClf_;
    std::unique_ptr<Clasificador> svmClf_;
    bool orientTemplatesLoaded_ = false;
    QString orientTemplatesDir_;

    // Guarda el último resultado de la segmentación offline para poder clasificarlo después
    std::vector<ResultadoPieza> lastResultados_;

    void LoadDefaultSettings(); // Para poner rutas por defecto al iniciar

	// función de entrenamiento si no hay modelo
    void maybeTrain();
    void runEvalGlobal();
	void runEvalAmarillas();

private slots:
    void EnableButtons(bool StartCapture);
    void NewImage(Mat Img);
    void SaveImage();
    void ShowImage();
    void ReconectarCamara();
    void CheckCameraStatus();

    // control de segmentación
    void EnableLiveSegmentation(bool enabled);
    void UpdateSegmentationResults(const std::vector<QRectF>& boxes,
        const std::vector<QImage>& thumbnails,
        const std::vector<cv::Mat>& crops);

    // timer slot que pide un frame para segmentar (no bloqueante)
    void onSegmentationTimer();

    // Clasificación
    void onCheckLiveClass(bool checked);
    void UpdateClassificationResults(std::vector<QRectF> boxes, std::vector<QString> labels);

    // selección/procesado de imagen desde fichero (offline)
    void CapturarYAnalizar();
    void CargarImagenDisco();
    void RecalcularSegmentacion();
    void ProcesarImagenOffline(const cv::Mat& img);
    void UpdateFileNameLabel();
    void SaveImageAs();
    void ProcesarClasificacionOffline();

    // Entrenamiento
    void onBrowseRaw();
    void onBrowseSeg();
    void onBrowseFeatures();
    void onBrowseModel();
    void onBrowseTest();

    void onCheckSkipSeg(bool checked);
    void onCheckSkipExtract(bool checked);
    void onCheckSkipTrain(bool checked);
    void onCheckSkipEval(bool checked);

    void onStartTrainingClicked();

    // Ajustes
    void onSetBrowseTemplates();
    void onSetBrowseModel();
    void onSetBrowseScaler();



    void AbrirYClasificarOrientacion();

    bool EnsureOrientTemplatesLoaded();
    void OnBatchSegmentar();

};