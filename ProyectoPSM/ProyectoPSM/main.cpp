#include "ProyectoPSM.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    ProyectoPSM window;
    
    window.showMaximized();
    return app.exec();
}
