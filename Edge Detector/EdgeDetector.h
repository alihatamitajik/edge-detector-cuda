#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>

#include <QPixmap>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <exception>
#include "cuda_runtime.h"

enum class DetectorErrors
{
	SUCCESS, // Operation finished successfully
	BAD_THRESHOLD, // Bad threshold was provided
	BAD_BRIGHTNESS, // Bas brightness was provided
	CUDA_ERROR // An cuda error occured during kernel execution
};

/*
 * Statistics
 * This struct contains the statistics and results of the kernel executions. */
typedef struct statistisc {
	/* Memorry allocation time */
	double mallocTime;
	/* Setting Brightness Time (Execution time of changeBrightnessCUDA kernel) */
	double brightnessTime;
	/* Detevting Edge Time (Execution time of detectEdgeCUDA (some varient of it) kernel)*/
	double edgeTime;
	/* If an error occured above fields will be 0 and e != SUCCESS */
	DetectorErrors e;
	/* If the error was caused by CUDA the following will contain error struct */
	cudaError_t cudaE;
} detectStat_t;

class EdgeDetector
{
public:
	void openFile(std::string);
	QPixmap getQPix();
	QPixmap getBrightnessPix();
	QPixmap getEdgePix();
	void showMaximized();
	void clear();
	bool save(std::string);
	std::string getSizeStr();
	std::string getDimensionStr();
	detectStat_t detectEdges(int threshold, int brightness);

private:
	cv::Mat img;
	cv::Size size;
	uint8_t* input;
	uint8_t* bright;
	uint8_t* edges;
};

