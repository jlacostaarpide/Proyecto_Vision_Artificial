#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>

using namespace cv;
using namespace std;
using namespace Pylon;
using namespace GenApi;
using namespace Basler_UniversalCameraParams;

//clase que hereda de la clase "hilos"
class CVideoAcquisition : public QThread
{
	Q_OBJECT 

private:
	//variables de la clase de acceso privado
	CBaslerUniversalInstantCamera* Camera; //variable para gestionar la captura de imagenes 	
	CImageFormatConverter FormatConverter; //variable para conversión de formatos
	CGrabResultPtr PtrGrabResult; //variable para guardar el resultado de la captura
	CPylonImage PylonImage; //variable para guardar la imagen de la cámara
	Mat OpenCvImage; //variable para guardar la última imagen en formato OpenCv	
	bool Recording; //variable para indicar si se está grabando
	
	//método que se ejecutará cuando se llame a la función start de esta clase
	void run();

public:
	//constructor por defecto
	CVideoAcquisition();

	//destructor
	~CVideoAcquisition();

	bool CameraOK; //variable para indicar si se ha realizado la comunicación con la cámara

	Mat GetImage(); //función que devuelve la última imagen
	void SetCameraAutoExposure(); //función para poner en automatico el tiempo de exposición
	void SetCameraExposure(double exposure); //función para cambiar el tiempo de exposición
	
signals:	
	//señal que se produce cada vez que hay una nueva imagen disponible
	void NewImageSignal(Mat Img);

public slots:	
	//slot para ejecutar/parar la adquisición de imágenes
	void StartStopCapture(bool StartCapture);
};