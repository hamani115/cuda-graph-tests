// C++ standard headers
#include <cassert>
#include <iostream>
#include <vector>
#include <iomanip>  // For std::setprecision
#include <cmath>    // For sqrt in standard deviation calculation

// CUDA headers
#include <cuda_runtime.h>

// Local headers
#include "../cuda_check.h"

// Here you can set the device ID
#define MYDEVICE 0

#define N (1U << 20) //(1 << 12) // 4096 elements
#define NSTEP 100000
#define SKIPBY 0

// CUDA kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, //float *a3, float *a4, float *a5,
                        //    float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i];// + a3[i] + a4[i] + a5[i];
                //   + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

// Function for non-graph implementation
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout) {
    // Choose one CUDA device
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    size_t memSize = N * sizeof(float);

    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize h_a using index 'i'
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    // Pointers for device memory
    float *d_result;
    float *d_a1, *d_a2;//, *d_a3, *d_a4, *d_a5;//, *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate the device memory
    CUDA_CHECK(cudaMallocAsync(&d_a1, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a2, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a3, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a4, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a5, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a6, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a7, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a8, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a9, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a10, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream)); // Allocate device memory for result

    // Initialize d_result to zero
    CUDA_CHECK(cudaMemsetAsync(d_result, 0, memSize, stream));

    // Set up grid and block dimensions
    // int threadsPerBlock = 256;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timings
    cudaEvent_t start, stop;
    float firstTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Host to device memory copy
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a, memSize, cudaMemcpyHostToDevice, stream));

    // Start Timer for first run
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Device to device memory copies
    CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, stream));

    // Single kernel launch after all memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_a1, d_a2,// d_a3, d_a4, d_a5,
        // d_a6, d_a7, d_a8, d_a9, d_a10,
        d_result);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Device to host memory copy
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // Wait for the execution to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // End Timer for first run
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    // Variables for timing statistics
    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // int skipBy = 100;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    for (int istep = 0; istep < NSTEP - 1; istep++) {
        // Start Timer for each run
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Device to device memory copies
        CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, stream));

        // Single kernel launch after all memcpys
        add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_a1, d_a2, //d_a3, d_a4, d_a5,
            // d_a6, d_a7, d_a8, d_a9, d_a10,
            d_result);
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
        if (istep >= SKIPBY) {
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
            if (elapsedTime < lowerTime || lowerTime == 0.0f) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstTime) / NSTEP;
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup (No Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: " << 1 << std::endl;
    std::cout << "Block Size: " << threadsPerBlock << std::endl;
    std::cout << "Grid Size: " << blocksPerGrid << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // Verify the data on the host is correct
    for (int i = 0; i < N; ++i)
    {
        float expected = i * 2.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_a2));
    // CUDA_CHECK(cudaFree(d_a3));
    // CUDA_CHECK(cudaFree(d_a4));
    // CUDA_CHECK(cudaFree(d_a5));
    // CUDA_CHECK(cudaFree(d_a6));
    // CUDA_CHECK(cudaFree(d_a7));
    // CUDA_CHECK(cudaFree(d_a8));
    // CUDA_CHECK(cudaFree(d_a9));
    // CUDA_CHECK(cudaFree(d_a10));
    CUDA_CHECK(cudaFree(d_result));

    // Return total time including first run
    // return totalTime + firstTime;
    *totalTimeWith = totalTime + firstTime;
    *totalTimeWithout = totalTime;
}

// Function for graph implementation
void runWithGraph(float* totalTimeWith, float* totalTimeWithout) {
    // Choose one CUDA device
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    size_t memSize = N * sizeof(float);

    // Allocate pinned host memory for h_a and h_result
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize h_a using index 'i'
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    // Allocate the device memory
    float *d_a1, *d_a2;//, *d_a3, *d_a4, *d_a5;
    // float *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;
    float *d_result;

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMallocAsync(&d_a1, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_a2, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a3, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a4, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a5, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a6, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a7, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a8, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a9, memSize, stream));
    // CUDA_CHECK(cudaMallocAsync(&d_a10, memSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream));

    // Set up grid and block dimensions
    // int threadsPerBlock = 256;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timings
    cudaEvent_t start, stop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Host to device memory copy asynchronously
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a, memSize, cudaMemcpyHostToDevice, stream));

    // Start Timer for graph creation
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin capturing the stream to create the CUDA graph
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Device to device memory copies asynchronously
    CUDA_CHECK(cudaMemcpyAsync(d_a2, d_a1, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a3, d_a2, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a4, d_a3, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a5, d_a4, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a6, d_a5, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a7, d_a6, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a8, d_a7, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a9, d_a8, memSize, cudaMemcpyDeviceToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_a10, d_a9, memSize, cudaMemcpyDeviceToDevice, stream));

    // Single kernel launch after all memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a1, d_a2,// d_a3, d_a4, d_a5,
                                                            //   d_a6, d_a7, d_a8, d_a9, d_a10,
                                                              d_result);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors

    // Device to host memory copy asynchronously
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // End capturing the stream to create the CUDA graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // End Timer for graph creation
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, start, stop));

    // Variables for timing statistics
    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // int skipBy = 100;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    for (int istep = 0; istep < NSTEP - 1; istep++) {
        // Start Timer for each run
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        // Wait for the graph execution to finish
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // End Timer for each run
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Time Calculations
        if (istep >= SKIPBY) {
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
            if (elapsedTime < lowerTime || lowerTime == 0.0f) {
                lowerTime = elapsedTime;
            }
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + graphCreateTime) / NSTEP;
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup (With Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: " << 1 << std::endl;
    std::cout << "Block Size: " << threadsPerBlock << std::endl;
    std::cout << "Grid Size: " << blocksPerGrid << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Verify the data on the host is correct
    for (int i = 0; i < N; ++i)
    {
        float expected = i * 2.0f; // Since each a_i contains i, and we sum over 10 arrays
        if (h_result[i] != expected) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_result[i] << std::endl;
            assert(false);
        }
    }

    // Clean up
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_a2));
    // CUDA_CHECK(cudaFree(d_a3));
    // CUDA_CHECK(cudaFree(d_a4));
    // CUDA_CHECK(cudaFree(d_a5));
    // CUDA_CHECK(cudaFree(d_a6));
    // CUDA_CHECK(cudaFree(d_a7));
    // CUDA_CHECK(cudaFree(d_a8));
    // CUDA_CHECK(cudaFree(d_a9));
    // CUDA_CHECK(cudaFree(d_a10));
    CUDA_CHECK(cudaFree(d_result));

    // Return total time including graph creation
    // return totalTime + graphCreateTime;
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main() {
    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    // float nonGraphTotalTime = runWithoutGraph(N);
    runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    // float graphTotalTime = runWithGraph(N);
    runWithGraph(&graphTotalTime, &graphTotalTimeWithout);

     // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerKernel = difference / (NSTEP);
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Compute the difference for without including Graph
    float difference2 = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    float diffPerKernel2 = difference2 / (NSTEP-1);
    float diffPercentage2 = (difference2 / nonGraphTotalTimeWithout) * 100;

    // Print the differences
    std::cout << "=======Comparison without Graph Creation=======" << std::endl;
    std::cout << "Difference: " << difference2 << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

    // Print the differences
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
