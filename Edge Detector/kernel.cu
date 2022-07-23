#include "edge.cuh"

constexpr auto BLOCK_SIZE = 1024;
constexpr auto BLOCK_DIM = 32;

#define ASSERT(exp, ...) cudaStatus = exp; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, __VA_ARGS__); \
        goto Error; \
    } 


__global__ void changeBrightnessCUDA(uint8_t* input, const int width, const int height, const int brightness)
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


// One implementation of naive covolution


// One convoloution with shared data


// One convolution with handwriten sobel


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


    // Wait until brightness copy and edge detection launch is done
    ASSERT(cudaDeviceSynchronize(),
        "cudaDeviceSynchronize returned error code %d after launching cudaMemcpyAsync!\n"
        , cudaStatus);


Error:
    cudaFree(dev_input);
    cudaFree(dev_edge);
    return cudaStatus;
}