#include "EdgeDetector.h"

#include "edge.cuh"
#include "cuda_runtime.h"

#include <string> 
#include <QImage>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

void EdgeDetector::openFile(String filename) {
	clear();
	img = imread(filename);
	cvtColor(img, img, COLOR_BGR2GRAY);
	input = img.isContinuous() ? img.data : img.clone().data;
}

QPixmap EdgeDetector::getQPix() {
	return QPixmap::fromImage(QImage(input, 
		img.cols, 
		img.rows, 
		img.step, 
		QImage::Format_Grayscale8
	));
}

void EdgeDetector::clear() {
	/* TODO */
}

String EdgeDetector::getDimensionStr() {
	Size s = img.size();
	return fmt::format("Dim: {}x{}", s.width, s.height);
}

String EdgeDetector::getSizeStr() {
	Size s = img.size();
	unsigned long long size = s.height * s.width;
	if (size < 1000) {
		return std::to_string(size) + "B";
	}
	else if (size < 1000000) {
		return fmt::format("Size: {:.2f}KB", size/1000.0);
	} 
	else  if (size < 1000000000) {
		return fmt::format("Size: {:.2f}MB", size / 1000000.0);
	}
	else  if (size < 1000000000000) {
		return fmt::format("Size: {:.2f}GB", size / 1000000000.0);
	}
	else {
		return "Size: Error";
	}
}