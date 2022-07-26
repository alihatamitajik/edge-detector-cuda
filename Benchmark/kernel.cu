#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

constexpr auto BLOCK_DIM = 32;

#define checkGpuError(exp) cudaStatus = exp; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        goto Error; \
    } 

/* Edge detection with Sobel filter using
 * naive image convolution operator (Implemented from psudocode of [1])
 *
 * This function will not be used as the sobel filter we use and is just for
 * performance comparison purposes.
 *
 *
 * [1] https://en.wikipedia.org/wiki/Sobel_operator
 */
__global__ void sobelCUDA(const uint8_t* image,
    const int8_t* xKernel,
    const int8_t* yKernel,
    uint8_t* output,
    int width,
    int height,
    int kernelDim,
    int threshold)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * width + j;

    int center = (kernelDim - 1) / 2;
    float S1 = 0, S2 = 0;
    int jshift, ishift;
    int out;

    if (i >= center && j >= center &&
        i < height - center && j < width - center)
    {
        for (int ii = 0; ii < kernelDim; ii++) {
            for (int jj = 0; jj < kernelDim; jj++) {
                jshift = jj + j - center;
                ishift = ii + i - center;
                S1 += image[ishift * width + jshift] * xKernel[ii * kernelDim + jj];
                S2 += image[ishift * width + jshift] * yKernel[ii * kernelDim + jj];
            }
        }

        out = sqrtf(S1 * S1 + S2 * S2);
        output[idx] = out > threshold ? out : 0;
    }
}

/*
 * Hard-Coded Sobel Filter in the kernel.
 *
 * In this implementation, we use the properties of the algorithm (make use of
 * zeros in the filters, Loop unrowling, using subtraction instead of "*-1" and
 * reducing kernel launches.
 *
 * The above strategies boosts the performance of the code!
 *
 */

__global__ void sobelOptimizedCUDA(const uint8_t* image, uint8_t* output,
    int width, int height, int threshold)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = i * width + j;

    float S1, S2;
    int out;
    if (j > 0 && i > 0 && j < width - 1 && i < height - 1) {
        S1 = image[(i - 1) * width + (j + 1)] - image[(i - 1) * width + (j - 1)] - image[(i + 1) * width + (j - 1)] +
            2 * (image[i * width + (j + 1)] - image[i * width + (j - 1)]) + (image[(i + 1) * width + (j + 1)]);

        S2 = (image[(i - 1) * width + (j - 1)]) + (2 * image[(i - 1) * width + j]) + (image[(i - 1) * width + (j + 1)]) +
            (-1 * image[(i + 1) * width + (j - 1)]) + (-2 * image[(i + 1) * width + j]) + (-1 * image[(i + 1) * width + (j + 1)]);


        out = sqrtf(S1 * S1 + S2 * S2);
        output[idx] = out > threshold ? out : 0;
    }
}

/*
 * Hard-Coded + Shared Memory
 *
 * Also we will access the memory way less with shared data and also helps the
 * bandwidth.
 */
__global__ void sobelOptimizedShCUDA(const uint8_t* image, uint8_t* output,
    int width, int height, int threshold)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * width + j;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int y34 = y * 34;
    int out;
    float S1, S2;

    __shared__ uint8_t sdata[34 * 34];
     if (i < height && j < width) {
        // Main data
        sdata[(y + 1) * 34 + x + 1] = image[idx];

        // Fix Boudnries
        if (y == 0 && blockIdx.y != 0) {
            sdata[x + 1] = image[(i - 1) * width + j];
            if (x == 0 && blockIdx.x != 0) {
                sdata[0] = image[(i - 1) * width + (j - 1)];
            }
        }
        if (x == 0 && blockIdx.x != 0) {
            sdata[y34 + 34] = image[(i)*width + j - 1];
            if (y == 31 && i != height - 1) {
                sdata[33 * 34] = image[(i + 1) * width + j - 1];
            }
        }
        if (y == 31 && i != height - 1) {
            sdata[33 * 34 + x + 1] = image[((i + 1) * width + j)];
            if (x == 31 && j != width - 1) {
                sdata[33 * 34 + 33] = image[(i + 1) * width + j + 1];
            }
        }
        if (x == 31 && j != width - 1) {
            sdata[y34 + 34 + 33] = image[i * width + j + 1];
            if (y == 0 && blockIdx.y != 0) {
                sdata[33] = image[(i - 1) * width + j + 1];
            }
        }
    }
    // waits untill shared data is completed
    __syncthreads();

    if (i >= 1 && j >= 1 &&
        i < height - 1 && j < width - 1)
    {
        S1 = sdata[y34 + x + 2] - sdata[y * 34 + x]
            + 2 * (sdata[y34 + 34 + x + 2] - sdata[(y + 1) * 34 + x])
            + sdata[y34 + 70 + x] - sdata[y34 + 68 + x];

        S2 = sdata[y34 + x + 70] + sdata[y34 + 68 + x]
            + 2 * (sdata[y34 + 69 + x] - sdata[y34 + x + 1])
            - sdata[y34 + x + 2] - sdata[y34 + x];

        out = sqrtf(S1 * S1 + S2 * S2);
        output[idx] = out > threshold ? out : 0;
    }
}




__host__ cudaError_t launchDetectEdgeOprimized(uint8_t* input, int width, int height, int threshold, float* detecct_ms, bool isShared)
{
    cudaError_t cudaStatus;
    uint8_t* dev_input;
    uint8_t* dev_edge;
    size_t imageSize = width * height * sizeof(uint8_t);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(width / BLOCK_DIM + (width % BLOCK_DIM != 0),
        height / BLOCK_DIM + (width % BLOCK_DIM != 0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkGpuError(cudaMalloc((void**)&dev_input, imageSize));
    checkGpuError(cudaMalloc((void**)&dev_edge, imageSize));
    checkGpuError(cudaMemcpy(dev_input, input, imageSize, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    if (isShared)
        sobelOptimizedShCUDA <<<grid, block>>> (dev_input, dev_edge, width, height, threshold);
    else
        sobelOptimizedCUDA   <<<grid, block>>> (dev_input, dev_edge, width, height, threshold);

    cudaEventRecord(stop);
    checkGpuError(cudaDeviceSynchronize());

    cudaEventElapsedTime(detecct_ms, start, stop);

Error:
    cudaFree(dev_input);
    cudaFree(dev_edge);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cudaStatus;
}


cudaError_t launchNaiveSobel(uint8_t* input, int width, int height, int threshold, float* detecct_ms)
{
    cudaError_t cudaStatus;
    uint8_t* dev_input;
    uint8_t* dev_edge;
    size_t imageSize = width * height * sizeof(uint8_t);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(width / BLOCK_DIM + (width % BLOCK_DIM != 0),
        height / BLOCK_DIM + (width % BLOCK_DIM != 0));

    int8_t* dev_xK;
    int8_t* dev_yK;
    size_t kernelSize = 3 * 3 * sizeof(int8_t);

    // Kernels are aligned properly so we only should multiply values
    int8_t xKernel[][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int8_t yKernel[][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkGpuError(cudaMalloc((void**)&dev_input, imageSize));
    checkGpuError(cudaMalloc((void**)&dev_edge, imageSize));
    checkGpuError(cudaMalloc((void**)&dev_xK, kernelSize));
    checkGpuError(cudaMalloc((void**)&dev_yK, kernelSize));
    checkGpuError(cudaMemcpy(dev_input, input, imageSize, cudaMemcpyHostToDevice));
    checkGpuError(cudaMemcpy(dev_xK, xKernel, kernelSize, cudaMemcpyHostToDevice));
    checkGpuError(cudaMemcpy(dev_yK, yKernel, kernelSize, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    sobelCUDA <<<grid, block>>> (dev_input, dev_xK, dev_yK, dev_edge,
        width, height, 3, threshold);
    cudaEventRecord(stop);

    checkGpuError(cudaDeviceSynchronize());
    cudaEventElapsedTime(detecct_ms, start, stop);

Error:
    cudaFree(dev_xK);
    cudaFree(dev_yK);
    cudaFree(dev_input);
    cudaFree(dev_edge);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return cudaStatus;
}


void naiveSobeCPU(uint8_t* input, uint8_t* edge,
    int width, int height, int threshold)
{
    // Kernels are aligned properly so we only should multiply values
    int8_t xKernel[][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int8_t yKernel[][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            int index = j + i * width;
            float magX = 0;
            float magY = 0;
            for (int a = -1; a < 1; a++)
            {
                for (int b = -1; b < 1; b++)
                {
                    magX += input[index] * xKernel[a + 1][b + 1];
                    magY += input[index] * yKernel[a + 1][b + 1];
                }
            }
            edge[index] = max(sqrt((magX * magX) + (magY * magY)), 255.0);
        }
    }
}



void sobelOptimizedCPU(uint8_t* input, uint8_t* edge,
    int width, int height, int threshold)
{
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float magX = input[(i - 1) * width + (j + 1)] - input[(i - 1) * width + (j - 1)] - input[(i + 1) * width + (j - 1)] +
                2 * (input[i * width + (j + 1)] - input[i * width + (j - 1)]) + (input[(i + 1) * width + (j + 1)]);

            float magY = (input[(i - 1) * width + (j - 1)]) + (2 * input[(i - 1) * width + j]) + (input[(i - 1) * width + (j + 1)]) +
                (-1 * input[(i + 1) * width + (j - 1)]) + (-2 * input[(i + 1) * width + j]) + (-1 * input[(i + 1) * width + (j + 1)]);
            edge[i * width + j] = sqrt((magX * magX) + (magY * magY));
        }
    }
}


int main() {
    char* testImages[] = { "./image/small.jpg", 
                           "./image/meduim.jpg", 
                           "./image/large.jpg", 
                           "./image/extra.jpg", 
                           "./image/verylarge.jpg" };
    

    float total_time;
    float ms;


    uint8_t* input;
    uint8_t* output;
    for (int i = 0; i < 5; i++)
    {
        cv::Mat img = cv::imread(testImages[i]);
        cvtColor(img, img, cv::COLOR_BGR2GRAY);

        input = img.isContinuous() ? img.data : img.clone().data;
        output = (uint8_t*) malloc(img.size().width * img.size().height * sizeof(uint8_t));
        printf("File name: %s\n", testImages[i]);
        printf("Size: %d x %d\n", img.size().width, img.size().height);


        printf("Naive Sobel: ");
        launchNaiveSobel(input, img.size().width, img.size().height, 0, &ms);
        printf("%.3f\n", ms);

        printf("Optimized Sobel: ");
        launchDetectEdgeOprimized(input, img.size().width, img.size().height, 0, &ms, false);
        printf("%.3f\n", ms);

        printf("Optimized Sobel with Shared Memory: ");
        launchDetectEdgeOprimized(input, img.size().width, img.size().height, 0, &ms, true);
        printf("%.3f\n", ms);

        printf("Naive Sobel CPU: ");
        auto startT = std::chrono::high_resolution_clock::now();
        naiveSobeCPU(input, output, img.size().width, img.size().height, 0);
        auto endT = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> float_ms = endT - startT;
        printf("%.3f\n", float_ms.count());

        printf("Optimized Sobel CPU: ");
        startT = std::chrono::high_resolution_clock::now();
        sobelOptimizedCPU(input, output, img.size().width, img.size().height, 0);
        endT = std::chrono::high_resolution_clock::now();
        float_ms = endT - startT;
        printf("%.3f\n", float_ms.count());

        if (input != img.data) {
            free(input);
        }
        img.release();
        free(output);
        printf("\n\n============================================\n\n");
    }
}