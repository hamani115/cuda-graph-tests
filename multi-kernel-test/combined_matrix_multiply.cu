#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation

// Local headers
#include "../cuda_check.h"

#define N 1024        // Matrix dimensions (N x N)
#define NSTEP 500
#define NKERNEL 100 // Number of kernels to launch per iteration
#define SKIPBY 0

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < width && col < width) {
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Function to perform matrix multiplication without using CUDA Graphs
void matrixMultiplyNoGraph(int width, float* totalTimeWith, float* totalTimeWithout) {
    // Allocate host memory
    float* h_A = (float*)malloc(width * width * sizeof(float));
    float* h_B = (float*)malloc(width * width * sizeof(float));
    float* h_C = (float*)malloc(width * width * sizeof(float));

    // Check host memory allocation
    // if (!h_A || !h_B || !h_C) {
    //     fprintf(stderr, "Failed to allocate host memory\n");
    //     return EXIT_FAILURE;
    // }

    // Initialize matrices using index i
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, width * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, width * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, width * width * sizeof(float)));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 block(32, 32);  // 1024 threads
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create CUDA events
    cudaEvent_t start, stop;
    float firstTime = 0.0f;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;

    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Start recording time for first run
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin first run
    for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
        matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, width);
    }
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
    CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure all kernels finish

    // Stop recording time for first run
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    // Execute the kernel multiple times and measure performance
    for (int j = 0; j < NSTEP - 1; j++) {
        // Start the timer for each iteration
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch the kernel multiple times
        for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
            matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, width);
        }
        CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure all kernels finish

        // Stop the timer for each iteration
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Time calculations
        if (j >= SKIPBY) {
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
    std::cout << "Kernels per Iteration: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // Copy result back to host (optional, not used in this context)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Destroy the CUDA stream and events
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free device and host memory
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Return total time including first run
    // return totalTime + firstTime;
    *totalTimeWith = totalTime + firstTime;
    *totalTimeWithout = totalTime;
}

// Function to perform matrix multiplication using CUDA Graphs
void matrixMultiplyWithGraph(int width, float* totalTimeWith, float* totalTimeWithout) {
    // Allocate host memory
    float* h_A = (float*)malloc(width * width * sizeof(float));
    float* h_B = (float*)malloc(width * width * sizeof(float));
    float* h_C = (float*)malloc(width * width * sizeof(float));

    // Check host memory allocation
    // if (!h_A || !h_B || !h_C) {
    //     fprintf(stderr, "Failed to allocate host memory\n");
    //     return EXIT_FAILURE;
    // }

    // Initialize matrices using index i
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, width * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, width * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, width * width * sizeof(float)));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 block(32, 32); // 1024 threads
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create CUDA graph and events
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    float graphCreateTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;

    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start recording time for graph creation
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin graph capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Enqueue kernel launches into the graph
    for (int i = 0; i < NKERNEL; i++) {
        matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, width);
    }
    // Check for any kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // End graph capture
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Stop recording time for graph creation
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, start, stop));

    // Execute the graph multiple times and measure performance
    for (int i = 0; i < NSTEP - 1; i++) {
        // Start the timer for each iteration
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

        // Synchronize to ensure all operations are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Stop the timer for each iteration
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Time calculations
        if (i >= SKIPBY) {
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
    std::cout << "Kernels per Iteration: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Copy result back to host (optional, not used in this context)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free device and host memory
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Return total time including graph creation
    // return totalTime + graphCreateTime;
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main() {
    // Measure time for non-graph implementation
    // float nonGraphTotalTime = matrixMultiplyNoGraph(N);

    // // Measure time for graph implementation
    // float graphTotalTime = matrixMultiplyWithGraph(N);

    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    // float nonGraphTotalTime = matrixMultiplyNoGraph(N);
    matrixMultiplyNoGraph(N, &nonGraphTotalTime, &nonGraphTotalTimeWithout);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    // float graphTotalTime = matrixMultiplyWithGraph(N);
    matrixMultiplyWithGraph(N, &graphTotalTime, &graphTotalTimeWithout);

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
