#include <QtWidgets/QApplication>

#include "panel.h"


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    Panel p;
    p.show();
    return app.exec();
}
