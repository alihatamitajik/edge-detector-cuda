#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include <stdio.h>
#include <stdlib.h>

#include "edge.cuh"
#include "cuda_runtime.h"




int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
