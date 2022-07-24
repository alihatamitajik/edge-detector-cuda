#include "edge.cuh"

constexpr auto BLOCK_SIZE = 1024;
constexpr auto BLOCK_DIM = 32;

#define checkGpuError(exp) cudaStatus = exp; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        goto Error; \
    } 

__global__ void changeBrightnessCUDA(uint8_t* input, const int width, 
    const int height, const int brightness)
{
    int val;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * width + j;

    if (i < height && j < width) {
        val = input[idx] + brightness;
        // Truncate the result (0..255)
        if (val > 255) {
            input[idx] = 255;
        }
        else if (val < 0) {
            input[idx] = 0;
        }
        else {
            input[idx] = val;
        }
    }
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

cudaError_t naiveSobel(uint8_t* dev_input, uint8_t* dev_edge, 
    int width, int height, int threshold) 
{
    cudaError_t cudaStatus;

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

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(width / BLOCK_DIM + 1, height / BLOCK_DIM + 1);

    int8_t* dev_xK;
    int8_t* dev_yK;
    size_t kernelSize = 3 * 3 * sizeof(int8_t);

    checkGpuError(cudaMalloc((void**)&dev_xK, kernelSize));

    checkGpuError(cudaMalloc((void**)&dev_yK, kernelSize));

    checkGpuError(cudaMemcpy(dev_xK, xKernel, kernelSize, cudaMemcpyHostToDevice));

    checkGpuError(cudaMemcpy(dev_yK, yKernel, kernelSize, cudaMemcpyHostToDevice));

    sobelCUDA<<<grid, block>>>(dev_input, dev_xK, dev_yK, dev_edge,
        width, height, 3, threshold);

    checkGpuError(cudaGetLastError());

    // Wait untill Data is ready
    checkGpuError(cudaDeviceSynchronize());

Error:
    cudaFree(xKernel);
    cudaFree(yKernel);
    return cudaStatus;
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
    int out;
    float S1, S2;

    __shared__ uint8_t sdata[34][34];
    if (i < height && j < width) {
        // Main data
        sdata[threadIdx.y + 1][threadIdx.x + 1] = image[idx];
        
        // Boudnries
        if (threadIdx.y == 0 && blockIdx.y != 0) {
            sdata[0][threadIdx.x + 1] = image[(i - 1) * width + j];
            if (threadIdx.x == 0 && blockIdx.x != 0) {
                sdata[0][0] = image[(i - 1) * width + (j-1)];
            }
        }
        if (threadIdx.x == 0 && blockIdx.x != 0) {
            sdata[threadIdx.y + 1][0] = image[(i) * width + j - 1];
            if (threadIdx.y == 31 && i != height - 1) {
                sdata[33][0] = image[(i + 1) * width + j - 1];
            }
        }
        if (threadIdx.y == 31 && i != height - 1) {
            sdata[33][threadIdx.x + 1] = image[((i + 1) * width + j)];
            if (threadIdx.x == 31 && j != width - 1) {
                sdata[33][33] = image[(i + 1) * width + j + 1];
            }
        }
        if (threadIdx.x == 31 && j != width - 1) {
            sdata[threadIdx.y + 1][33] = image[i * width + j + 1];
            if (threadIdx.y == 0 && blockIdx.y != 0) {
                sdata[0][33] = image[(i - 1) * width + j + 1];
            }
        }
    }
    // waits untill shared data is completed
    __syncthreads();

    if (idx == 0) {
        for (int ii = 0; ii < 33; ii++) {
            for (int jj = 0; jj < 33; jj++) {
                if (sdata[ii + 1][jj + 1] != image[ii * width + jj]) {
                    printf("%d, %d::%d, %d\n", blockIdx.x, blockIdx.y, ii, jj);
                }
            }
        }
    }

    if (i >= 1 && j >= 1 &&
        i < height - 1 && j < width - 1)
    { 
        S1 = sdata[threadIdx.y][threadIdx.x + 2] - sdata[threadIdx.y][threadIdx.x]
            + 2 * (sdata[threadIdx.y + 1][threadIdx.x + 2] - sdata[threadIdx.y + 1][threadIdx.x])
            + sdata[threadIdx.y + 2][threadIdx.x + 2] - sdata[threadIdx.y + 2][threadIdx.x];

        S2 = sdata[threadIdx.y + 2][threadIdx.x + 2] + sdata[threadIdx.y + 2][threadIdx.x] 
            + 2 * (sdata[threadIdx.y + 2][threadIdx.x + 1] - sdata[threadIdx.y][threadIdx.x + 1])
            - sdata[threadIdx.y][threadIdx.x + 2] - sdata[threadIdx.y][threadIdx.x];


        out = sqrtf(S1 * S1 + S2 * S2);
        output[idx] = out > threshold ? out : 0;
    }
}

__host__ cudaError_t launchDetectEdge(uint8_t * input, uint8_t * bright, uint8_t * edge,
    int width, int height, int brightness, int threshold)
{
    cudaError_t cudaStatus;
    uint8_t* dev_input;
    uint8_t* dev_edge;
    size_t imageSize = width * height * sizeof(uint8_t);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(width/BLOCK_DIM + (width%BLOCK_DIM!=0), 
        height/BLOCK_DIM + (width % BLOCK_DIM != 0));

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkGpuError(cudaSetDevice(0));

    checkGpuError(cudaMalloc((void**)&dev_input, imageSize));

    checkGpuError(cudaMalloc((void**)&dev_edge, imageSize));

    checkGpuError(cudaMemcpy(dev_input, input, imageSize, cudaMemcpyHostToDevice));


    changeBrightnessCUDA <<<grid, block>>> (dev_input, width, height, brightness);
    checkGpuError(cudaGetLastError());
    checkGpuError(cudaDeviceSynchronize());


    checkGpuError(cudaMemcpyAsync(bright, dev_input, imageSize, cudaMemcpyDeviceToHost));


    //checkGpuError(naiveSobel(dev_input, dev_edge, width, height, threshold));

    sobelOptimizedShCUDA <<<grid, block>>> (dev_input, dev_edge, width, height, threshold);
    checkGpuError(cudaGetLastError());
    checkGpuError(cudaDeviceSynchronize());

    checkGpuError(cudaMemcpy(edge, dev_edge, imageSize, cudaMemcpyDeviceToHost));

Error:
    cudaFree(dev_input);
    cudaFree(dev_edge);
    return cudaStatus;
}