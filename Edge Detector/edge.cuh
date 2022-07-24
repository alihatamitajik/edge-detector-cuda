#ifndef _EDGE_CUH
#define _EDGE_CUH

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>

cudaError_t launchDetectEdge(uint8_t* input, uint8_t* bright, uint8_t* edge,
	int width, int height, int brightness, int threshold,
	float* mem_ms, float* bright_ms, float* detecct_ms);

#endif // !_EDGE_CUH
