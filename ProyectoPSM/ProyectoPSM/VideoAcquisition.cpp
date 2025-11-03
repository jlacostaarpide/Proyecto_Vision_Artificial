#include "VideoAcquisition.h"

//constructor
CVideoAcquisition::CVideoAcquisition()
{
	try
	{
		PylonInitialize();

		//comenzar sin conexión con la cámara y sin grabar
		CameraOK = false;
		Recording = false;

		//crear el objeto que se utiliza para capturar imagenes
		Camera = new CBaslerUniversalInstantCamera(CTlFactory::GetInstance().CreateFirstDevice());
		//definir el formato de la imagen
		FormatConverter.OutputPixelFormat = PixelType_BGR8packed;

		//abrir la cámara			
		Camera->Open();
		CameraOK = Camera->IsOpen();
	}
	catch (RuntimeException ex)
	{	
		cout << ex.GetDescription() << endl;
	}
}

//destructor
CVideoAcquisition::~CVideoAcquisition(void)
{
	//parar captura
	StartStopCapture(false);
	wait(100);

	//liberar la memoria del objeto
	Camera->Close();
	PylonTerminate();
}

//funcion para empezar a capturar imagenes o parar
void CVideoAcquisition::StartStopCapture(bool startCapture)
{	
	//si hay que capturar
	if (startCapture)
	{
		//si se ha podido abrir la cámara
		if (CameraOK)
		{
			//habilitar la captura
			Camera->StartGrabbing();
			Recording = true;
			//lanzar el hilo de captura
			this->start(); 
		}
		else
			//si la cámara no se ha podido abrir, se lanza un mensaje de error
			qDebug() << QString("ERROR: La cámara no ha podido ser abierta");
	}		
	else
		//si no hay que capturar, deshabilitar la captura	
		Recording = false;
}

//función que captura imágenes
void CVideoAcquisition::run(void)
{	
	//mientras la cámara esté grabando
	while (Recording)
	{		
		//cuando la imagen esté disponible		
		Camera->RetrieveResult(5000, PtrGrabResult, TimeoutHandling_ThrowException);
		if (PtrGrabResult->GrabSucceeded())
		{		
			//capturar la imagen			
			FormatConverter.Convert(PylonImage, PtrGrabResult);			
			//convertirla a opencv
			OpenCvImage = Mat(PylonImage.GetHeight(), PylonImage.GetWidth(), CV_8UC3, PylonImage.GetBuffer());			
			//lanzar la señal de que ya hay disponible una nueva imagen
			emit NewImageSignal(OpenCvImage);
		}	
	}
	//parar la grabación de la cámara
	Camera->StopGrabbing();
}

//función que devuelve la ultima imagen obtenida
Mat CVideoAcquisition::GetImage()
{	
	//devolver la última imagen capturada
	return OpenCvImage.clone();
}

//función para poner en automatico el tiempo de exposición
void CVideoAcquisition::SetCameraAutoExposure()
{	
	Camera->ExposureAuto.SetValue(ExposureAuto_Continuous);
}

//función para cambiar el tiempo de exposición
void CVideoAcquisition::SetCameraExposure(double exposure)
{
	//valor de fábrica: 350000,0
	Camera->ExposureAuto.SetValue(ExposureAuto_Off);
	Camera->ExposureTimeAbs.SetValue(exposure);
}

