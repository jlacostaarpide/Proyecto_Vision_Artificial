#include "VideoAcquisition.h"

// Constructor
CVideoAcquisition::CVideoAcquisition()
{
    try
    {
        PylonInitialize();

        CameraOK = false;
        Recording = false;

        // Intentar conectar con la primera cámara disponible
        CTlFactory& tlFactory = CTlFactory::GetInstance();
        DeviceInfoList_t devices;
        if (tlFactory.EnumerateDevices(devices) == 0) {
            return;
        }

        Camera = new CBaslerUniversalInstantCamera(tlFactory.CreateFirstDevice());
        FormatConverter.OutputPixelFormat = PixelType_BGR8packed;

        Camera->Open();
        CameraOK = Camera->IsOpen();
    }
    catch (const GenericException& ex)
    {
        // Capturar excepciones de Pylon al iniciar
        cout << "Error iniciando camara: " << ex.GetDescription() << endl;
        CameraOK = false;
    }
    catch (...)
    {
        cout << "Error desconocido iniciando camara." << endl;
        CameraOK = false;
    }
}

// Destructor
CVideoAcquisition::~CVideoAcquisition(void)
{
    // Parar captura de forma segura
    Recording = false;
    wait(1000); // Esperar a que el hilo termine (1 seg max)

    if (Camera) {
        try {
            if (Camera->IsOpen()) {
                Camera->Close();
            }
            delete Camera;
            Camera = nullptr;
        }
        catch (...) {}
    }
    PylonTerminate();
}

// Iniciar/Parar captura
void CVideoAcquisition::StartStopCapture(bool startCapture)
{
    if (startCapture)
    {
        if (CameraOK && Camera && !Camera->IsGrabbing())
        {
            try {
                Camera->StartGrabbing();
                Recording = true;
                this->start();
            }
            catch (const GenericException& e) {
                qDebug() << "Error al iniciar captura:" << e.GetDescription();
                CameraOK = false;
            }
        }
        else if (!CameraOK) {
            qDebug() << "ERROR: Camara no lista.";
        }
    }
    else
    {
        Recording = false;
        // Esperar a que el hilo termine
        wait();
    }
}

// Bucle principal de captura
void CVideoAcquisition::run(void)
{
    while (Recording && CameraOK)
    {
        try {
            // Esperar resultado (Timeout 1000ms para revisar 'Recording' freq)
            if (Camera->RetrieveResult(1000, PtrGrabResult, TimeoutHandling_Return)) {
                if (PtrGrabResult->GrabSucceeded())
                {
                    FormatConverter.Convert(PylonImage, PtrGrabResult);
                    OpenCvImage = Mat(PylonImage.GetHeight(), PylonImage.GetWidth(), CV_8UC3, PylonImage.GetBuffer());
                    emit NewImageSignal(OpenCvImage);
                }
                else {
                    qDebug() << "Error GrabResult: " << PtrGrabResult->GetErrorCode() << " " << PtrGrabResult->GetErrorDescription();
                }
            }
        }
        catch (const GenericException& e) {
            // ¡CRÍTICO! Si se desconecta el cable, entra aquí.
            if (Camera->IsCameraDeviceRemoved()) {
                qDebug() << "CAMARA DESCONECTADA FISICAMENTE.";
                CameraOK = false;
                Recording = false; // Salir del bucle
            }
            else {
                qDebug() << "Error en run(): " << e.GetDescription();
            }
        }
    }

    // Parar grabación de forma segura si sigue abierta
    if (Camera && Camera->IsGrabbing()) {
        try {
            Camera->StopGrabbing();
        }
        catch (...) {}
    }
}

Mat CVideoAcquisition::GetImage()
{
    if (!OpenCvImage.empty()) return OpenCvImage.clone();
    return Mat();
}

void CVideoAcquisition::SetCameraAutoExposure()
{
    if (CameraOK && Camera) {
        try {
            Camera->ExposureAuto.SetValue(ExposureAuto_Continuous);
        }
        catch (...) {}
    }
}

void CVideoAcquisition::SetCameraExposure(double exposure)
{
    if (CameraOK && Camera) {
        try {
            Camera->ExposureAuto.SetValue(ExposureAuto_Off);
            Camera->ExposureTimeAbs.SetValue(exposure);
        }
        catch (...) {}
    }
}