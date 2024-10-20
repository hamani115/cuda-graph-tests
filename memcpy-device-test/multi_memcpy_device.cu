// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>
#include <iomanip>  // For std::setprecision

// CUDA headers
#include <cuda_runtime.h>

// local headers
#include "cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

#define NSTEP 10000

// Define the array size
const int N = 4096;
const int dimA = N;

// CUDA kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, float *a3, float *a4, float *a5,
                           float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i]
                  + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

///////////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////////
int main()
{
    // Choose one CUDA device
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    // Create a CUDA stream to execute asynchronous operations on this device
    cudaStream_t queue;
    CUDA_CHECK(cudaStreamCreate(&queue));

    size_t memSize = dimA * sizeof(float);
    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));

    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize h_a
    for (int i = 0; i < dimA; ++i) {
        h_a[i] = i;
    }

    // Pointers for device memory
    float *d_result;
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5, *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;

    // Part 1 of 5: allocate the device memory
    CUDA_CHECK(cudaMallocAsync(&d_a1, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a2, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a3, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a4, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a5, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a6, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a7, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a8, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a9, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_a10, memSize, queue));
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, queue)); // Allocate device memory for result

    // Initialize d_result to zero
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, queue));

    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timings
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    float firstTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Part 2 of 5: host to device memory copy
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a, memSize, cudaMemcpyHostToDevice, queue));

    // Start Timer
    CUDA_CHECK(cudaEventRecord(start, queue));

    // Reset d_result to zero
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, queue));

    // Part 3 of 5: device to device memory copies
    CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, queue));

    // Part 4 of 5: single kernel launch after all memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, queue>>>(d_a1, d_a2, d_a3, d_a4, d_a5,
                                                            d_a6, d_a7, d_a8, d_a9, d_a10,
                                                            d_result, N);

    // Part 5 of 5: device to host memory copy
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, queue));

    // Wait for the graph execution to finish
    CUDA_CHECK(cudaStreamSynchronize(queue));

    // End Timer
    CUDA_CHECK(cudaEventRecord(stop, queue));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    for (int istep = 0; istep < NSTEP-1; istep++) {
        // Start Timer
        CUDA_CHECK(cudaEventRecord(start, queue));

        // Reset d_result to zero
        CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, queue));

        // Part 3 of 5: device to device memory copies
        CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, queue));
        CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, queue));

        // Part 4 of 5: single kernel launch after all memcpys
        add_arrays<<<blocksPerGrid, threadsPerBlock, 0, queue>>>(d_a1, d_a2, d_a3, d_a4, d_a5,
                                                                d_a6, d_a7, d_a8, d_a9, d_a10,
                                                                d_result, N);

        // Part 5 of 5: device to host memory copy
        CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, queue));

        // Wait for the graph execution to finish
        CUDA_CHECK(cudaStreamSynchronize(queue));

        // End Timer
        CUDA_CHECK(cudaEventRecord(stop, queue));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        if (istep >= skipBy) {
            totalTime += elapsedTime;
            if (elapsedTime > upperTime) {
                upperTime = elapsedTime;
            }
            if (elapsedTime < lowerTime) {
                lowerTime = elapsedTime;
            }
            if (istep == skipBy) {
                lowerTime = elapsedTime;
            }
        }
    }

    float AverageTime = (totalTime+firstTime) / (NSTEP - skipBy);
    std::cout << "Average Time: " << AverageTime << "ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << "ms" << std::endl;
    std::cout << "Total Time without first run: " << totalTime << "ms" << std::endl;
    std::cout << "Total Time with first run: " << (totalTime+firstTime) << "ms" << std::endl;

    // **Print h_result before testing**
    std::cout << "h_result contents before verification:\n";
    std::cout << std::fixed << std::setprecision(2); // Set precision for floating-point output
    std::cout << "h_result[" << dimA - 1 << "] = " << h_result[dimA - 1] << "\n";

    // Verify the data on the host is correct
    for (int i = 0; i < dimA; ++i)
    {
        float expected = i * 10.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Destroy the CUDA stream
    CUDA_CHECK(cudaStreamDestroy(queue));

    // Destroy the events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_result));

    // Free the device memory
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_a2));
    CUDA_CHECK(cudaFree(d_a3));
    CUDA_CHECK(cudaFree(d_a4));
    CUDA_CHECK(cudaFree(d_a5));
    CUDA_CHECK(cudaFree(d_a6));
    CUDA_CHECK(cudaFree(d_a7));
    CUDA_CHECK(cudaFree(d_a8));
    CUDA_CHECK(cudaFree(d_a9));
    CUDA_CHECK(cudaFree(d_a10));
    CUDA_CHECK(cudaFree(d_result));

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors. Good work!
    std::cout << "Correct!" << std::endl;

    return 0;
}
