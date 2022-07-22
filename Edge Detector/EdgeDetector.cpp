#include "EdgeDetector.h"

#include "edge.cuh"
#include "cuda_runtime.h"

#include <QImage>

void EdgeDetector::openFile(String filename) {
	img = imread(filename);
	cvtColor(img, img, COLOR_BGR2GRAY);
	input = img.isContinuous() ? img.data : img.clone().data;
}

QPixmap EdgeDetector::getQPix() {
	
	return QPixmap::fromImage(QImage(img.data, 
		img.cols, 
		img.rows, 
		img.step, 
		QImage::Format_Grayscale8
	));
}

void EdgeDetector::clear() {
	/* TODO */
}