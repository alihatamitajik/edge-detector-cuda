#include <QtWidgets/QApplication>

#include <stdio.h>
#include <stdlib.h>

#include "edge.cuh"
#include "cuda_runtime.h"

#include "panel.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


using namespace cv;

int main(int argc, char *argv[])
{

    QApplication app(argc, argv);
    Panel p;
    p.show();
    return app.exec();
}
