#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include <stdlib.h>
#include <assert.h> 
#include <stdio.h>
#include <time.h>

#include "edge.cuh"


#define ASSERT(exp, ...) cudaStatus = exp; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, __VA_ARGS__); \
        goto Error; \
    } 
