#include "EdgeDetector.h"

#include "edge.cuh"
#include "cuda_runtime.h"

#include <QImage>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

void EdgeDetector::openFile(std::string filename) {
	clear();
	img = cv::imread(filename);
	cvtColor(img, img, cv::COLOR_BGR2GRAY);
	input = img.isContinuous() ? img.data : img.clone().data;
	size = img.size();
	bright = (uint8_t*)malloc(size.height*size.width*sizeof(uint8_t));
	edges = (uint8_t*)malloc(size.height * size.width * sizeof(uint8_t));
}

QPixmap EdgeDetector::getQPix() {
	return QPixmap::fromImage(QImage(input, 
		img.cols, 
		img.rows, 
		img.step, 
		QImage::Format_Grayscale8
	));
}

QPixmap EdgeDetector::getBrightnessPix() {
	return QPixmap::fromImage(QImage(bright,
		img.cols,
		img.rows,
		img.step,
		QImage::Format_Grayscale8
	));
}

QPixmap EdgeDetector::getEdgePix() {
	return QPixmap::fromImage(QImage(edges,
		img.cols,
		img.rows,
		img.step,
		QImage::Format_Grayscale8
	));
}

void EdgeDetector::showMaximized() {
	cv::Mat result = cv::Mat(size.height, size.width, CV_MAKETYPE(CV_8U, 1), edges);
	cv::namedWindow("Edges", cv::WINDOW_NORMAL);
	cv::imshow("Edges" , result);
}

void EdgeDetector::clear() {
	if (img.data != input) {
		free(input);
	}
	img.release();
	free(bright);
	free(edges);
}

bool EdgeDetector::save(std::string path) {
	cv::Mat result = cv::Mat(size.height, size.width, CV_MAKETYPE(CV_8U, 1), edges);
	bool res = imwrite(path, result);
	result.release();
	return res;
}

std::string EdgeDetector::getDimensionStr() {
	cv::Size s = this->size;
	return fmt::format("Dim: {}x{}", s.width, s.height);
}

std::string EdgeDetector::getSizeStr() {
	unsigned long long size = this->size.height * this->size.width;
	if (size < 1000) {
		return std::to_string(size) + " B";
	}
	else if (size < 1000000) {
		return fmt::format("Size: {:.2f} KB", size/1000.0);
	} 
	else  if (size < 1000000000) {
		return fmt::format("Size: {:.2f} MB", size / 1000000.0);
	}
	else  if (size < 1000000000000) {
		return fmt::format("Size: {:.2f} GB", size / 1000000000.0);
	}
	else {
		return "Size: Error";
	}
}

detectStat_t EdgeDetector::detectEdges(int threshold, int brightness) {
	cudaError_t cudaStatus = cudaErrorIllegalState;
	if (threshold < 0 || threshold > 255) {
		return detectStat_t{0, 0, 0, DetectorErrors::BAD_THRESHOLD, cudaStatus};
	}
	else if (brightness < -255 || brightness > 255) {
		return detectStat_t{ 0, 0, 0, DetectorErrors::BAD_BRIGHTNESS, cudaStatus};
	}
	else {
		detectStat_t stat;
		cudaStatus = launchDetectEdge(input, bright, edges, size.width, size.height,
			brightness, threshold, &stat.mallocTime, &stat.edgeTime, &stat.brightnessTime);

		if (cudaStatus != cudaSuccess) {
			stat.e = DetectorErrors::CUDA_ERROR;
			stat.cudaE = cudaStatus;

			return stat;
		}
		else {
			stat.e = DetectorErrors::SUCCESS;
			stat.cudaE = cudaStatus;
			return stat;
		}
	}
}