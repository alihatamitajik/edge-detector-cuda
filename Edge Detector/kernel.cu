#include "edge.cuh"

constexpr auto BLOCK_SIZE = 1024;
constexpr auto BLOCK_DIM = 32;

#define ASSERT(exp, ...) cudaStatus = exp; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, __VA_ARGS__); \
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

    ASSERT(cudaMalloc((void**)&dev_xK, kernelSize),
        "CudaMalloc Error: Can't allocate dev_input!");

    ASSERT(cudaMalloc((void**)&dev_yK, kernelSize),
        "CudaMalloc Error: Can't allocate dev_input!");

    ASSERT(cudaMemcpy(dev_xK, xKernel, kernelSize, cudaMemcpyHostToDevice),
        "Memcpy Error for xKernel");

    ASSERT(cudaMemcpy(dev_yK, yKernel, kernelSize, cudaMemcpyHostToDevice),
        "Memcpy Error for yKernel");

    sobelCUDA<<<grid, block>>>(dev_input, dev_xK, dev_yK, dev_edge,
        width, height, 3, threshold);

    ASSERT(cudaGetLastError(),
        "sobelCUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

    // Wait untill Data is ready
    ASSERT(cudaDeviceSynchronize(),
        "cudaDeviceSynchronize returned error code %d after launching sobelCUDA!\n"
        , cudaStatus);

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


__host__ cudaError_t launchDetectEdge(uint8_t * input, uint8_t * bright, uint8_t * edge,
    int width, int height, int brightness, int threshold)
{
    cudaError_t cudaStatus;
    uint8_t* dev_input;
    uint8_t* dev_edge;
    size_t imageSize = width * height * sizeof(uint8_t);
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(width/BLOCK_DIM + 1, height/BLOCK_DIM + 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0),
        "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    ASSERT(cudaMalloc((void**)&dev_input, imageSize),
        "CudaMalloc Error: Can't allocate dev_input!");

    ASSERT(cudaMalloc((void**)&dev_edge, imageSize),
        "CudaMalloc Error: Can't allocate dev_edge!");

    ASSERT(cudaMemcpy(dev_input, input, imageSize, cudaMemcpyHostToDevice),
        "CudaMemcpy Error: Can't copy input to dev_input!");


    changeBrightnessCUDA <<<grid, block>>> (dev_input, width, height, brightness);
    ASSERT(cudaGetLastError(),
        "changeBrightnessCUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

    // Wait untill Data is ready
    ASSERT(cudaDeviceSynchronize(),
        "cudaDeviceSynchronize returned error code %d after launching changeBrightnessCUDA!\n"
        , cudaStatus);

    ASSERT(cudaMemcpyAsync(bright, dev_input, imageSize, cudaMemcpyDeviceToHost),
        "CudaMemcpy Async Error: Can't copy brightness-changes input to host!");


    ASSERT(naiveSobel(dev_input, dev_edge, width, height, threshold),
        "Naive Sobel Failed");


    ASSERT(cudaMemcpy(edge, dev_edge, imageSize, cudaMemcpyDeviceToHost),
        "CudaMemcpy Async Error: Can't copy brightness-changes input to host!");


Error:
    cudaFree(dev_input);
    cudaFree(dev_edge);
    return cudaStatus;
}