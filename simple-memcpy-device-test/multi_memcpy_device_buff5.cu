// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>
#include <iomanip>  // For std::setprecision

// CUDA headers
#include <cuda_runtime.h>

// Local headers
#include "../cuda_check.h"

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

#define N (1 << 12)  // 4096 elements
#define NSTEP 10000

// CUDA kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, float *a3, float *a4, float *a5,
                        //    float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i];
                //   + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

int main()
{
    // Choose one CUDA device
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    // Create a CUDA stream to execute asynchronous operations on this device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t memSize = N * sizeof(float);
    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize h_a
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    // Pointers for device memory
    float *d_result;
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5;//, *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;

    // Allocate the device memory
    CUDA_CHECK(cudaMallocAsync(&d_a1, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a2, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a3, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a4, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a5, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a6, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a7, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a8, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a9, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a10, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream)); // Allocate device memory for result

    // Initialize d_result to zero
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, stream));

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

    // Host to device memory copy
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a, memSize, cudaMemcpyHostToDevice, stream));

    // Start Timer for first run
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Reset d_result to zero
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, stream));

    // Device to device memory copies
    CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, stream));

    // Single kernel launch after all memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_a1, d_a2, d_a3, d_a4, d_a5,
        // d_a6, d_a7, d_a8, d_a9, d_a10,
        d_result, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Device to host memory copy
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // Wait for the execution to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // End Timer for first run
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    for (int istep = 0; istep < NSTEP - 1; istep++) {
        // Start Timer for each run
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Reset d_result to zero
        CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, stream));

        // Device to device memory copies
        CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, stream));

        // Single kernel launch after all memcpys
        add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_a1, d_a2, d_a3, d_a4, d_a5,
            // d_a6, d_a7, d_a8, d_a9, d_a10,
            d_result, N);
        CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

        // Device to host memory copy
        CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

        // Wait for the execution to finish
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // End Timer for each run
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Time Calculations
        if (istep >= skipBy) {
            totalTime += elapsedTime;

            // Welford's algorithm for calculating mean and variance
            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

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

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstTime) / (NSTEP - skipBy);
    double varianceTime = 0.0;
    if (count > 1) {
        varianceTime = M2 / (count - 1);
    }
    // Ensure variance is not negative due to floating-point errors
    if (varianceTime < 0.0) {
        varianceTime = 0.0;
    }
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << skipBy << std::endl;
    std::cout << "Kernels: " << 1 << std::endl;
    std::cout << "Block Size: " << threadsPerBlock << std::endl;
    std::cout << "Grid Size: " << blocksPerGrid << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "First Run: " << firstTime << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - skipBy)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // **Print h_result before testing**
    std::cout << "h_result contents before verification:\n";
    std::cout << std::fixed << std::setprecision(2); // Set precision for floating-point output
    std::cout << "h_result[" << N - 1 << "] = " << h_result[N - 1] << "\n";

    // Verify the data on the host is correct
    for (int i = 0; i < N; ++i)
    {
        float expected = i * 5.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Destroy the CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));

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
    // CUDA_CHECK(cudaFree(d_a6));
    // CUDA_CHECK(cudaFree(d_a7));
    // CUDA_CHECK(cudaFree(d_a8));
    // CUDA_CHECK(cudaFree(d_a9));
    // CUDA_CHECK(cudaFree(d_a10));
    CUDA_CHECK(cudaFree(d_result));

    // No run-time errors.
    std::cout << "Correct!" << std::endl;

    return 0;
}
