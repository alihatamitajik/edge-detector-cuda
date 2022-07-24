#include <QtWidgets/QApplication>

#include "panel.h"
#include <stdio.h>
#include <iostream>
#include <Windows.h>

int main(int argc, char *argv[])
{
 //Uncomment For Debug Console
//#ifdef _WIN32
//    if (AttachConsole(ATTACH_PARENT_PROCESS) || AllocConsole()) {
//        freopen("CONOUT$", "w", stdout);
//        freopen("CONOUT$", "w", stderr);
//    }
//#endif
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon("./Resources/fav.ico"));
    QCoreApplication::setApplicationName("Sobel");
    QCoreApplication::setApplicationVersion("0.1.0");
    Panel p;
    p.show();
    return app.exec();
}
