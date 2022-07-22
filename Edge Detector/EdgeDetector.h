#pragma once

#include <stdlib.h>
#include <iostream>

#include <QPixmap>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class EdgeDetector
{
public:
	void openFile(String);
	QPixmap getQPix();
	void clear();

private:
	Mat img;
	uint8_t* input;
	uint8_t* bright;
	uint8_t* edges;
	QPixmap input_pix;
	QPixmap bright_pix;
	QPixmap edge_pix;
};

