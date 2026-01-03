#pragma once
#include <QObject>
#include <QString>
#include <QDir>
#include <atomic>
#include <vector>

// Estructura simple para pasar configuración
struct TrainingConfig {
    QString rawFolder;       // Entrada: Imágenes Raw
    QString segFolder;       // Salida: Imágenes Recortadas
    QString featuresFile;
    QString modelFile;
    QString evaluationFolder;      // Carpeta de Test (si aplica)
	bool skipSegmentation;
    bool skipExtraction;
    bool skipTraining;
    bool skipEvaluation;
};

class TrainingWorker : public QObject {
    Q_OBJECT

public:
    explicit TrainingWorker(TrainingConfig config, QObject* parent = nullptr)
        : QObject(parent), cfg(config) {
        stopRequested.store(false);
    }

public slots:
    void process();  // El método principal que arranca todo
    void stop() { stopRequested.store(true); }

signals:
    // Actualizar barras de progreso (0 a 100)
    void progressSeg(int percent);
    void progressExtract(int percent);
    void progressTrain(int percent);

    // Mensajes para el log (texto negro)
    void logMessage(QString msg);

    // Señal final
    void finished();

private:
    TrainingConfig cfg;
    std::atomic<bool> stopRequested;

    // Pasos internos
    void runStepSegmentation();
    void runStepExtraction();
    void runStepTraining();
};