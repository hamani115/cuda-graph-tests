#include <cuda_runtime.h>
#include <stdio.h>
// #include <chrono>
#include <iostream>

// Local headers
#include "../cuda_check.h"

#define N 64  // Matrix dimensions (64x64)

#define NSTEP 100000
#define NKERNEL 10  // INDEPENDENT VARIABLE: CHANGE THE NUMBER OF KERNELS (10 OR 100)

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
void matrixMultiplyNoGraph(float* A, float* B, float* C, int width) {
    // Define block and grid sizes
    dim3 block(32, 32);  // 1024 threads
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    // dim3 grid(6, 6);  // 36 Blocks

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create CUDA events
    cudaEvent_t start, stop;
    float firstTime = 0.0f;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start recording time for first run
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin first run
    for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
        matMulKernel<<<grid, block, 0, stream>>>(A, B, C, width);
    }
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
    CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure all kernels finish

    // Stop recording time for first run
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Execute the kernel multiple times and measure performance
    for (int j = 0; j < NSTEP - 1; j++) {
        // Start the timer for each iteration
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch the kernel multiple times
        for (int i = 0; i < NKERNEL; i++) {  // Run NKERNEL iterations
            matMulKernel<<<grid, block, 0, stream>>>(A, B, C, width);
        }
        CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
        CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure all kernels finish

        // Stop the timer for each iteration
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Time calculations
        if (j >= skipBy) {
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
            if (i == skipBy) {
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
    std::cout << "Kernels: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "First Run: " << firstTime << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - skipBy)) << " ms" << std::endl;
    std::cout << "Variance: " <<  varianceTime << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstTime << " ms" << std::endl;

    // Destroy the CUDA stream and events
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    // Allocate host memory
    float* h_A = (float*)malloc(N * N * sizeof(float));
    float* h_B = (float*)malloc(N * N * sizeof(float));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    // Check host memory allocation
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand() % 100);
        h_B[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Measure time using std::chrono
    // auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyNoGraph(d_A, d_B, d_C, N);
    // auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate elapsed time
    // std::chrono::duration<double> elapsed = end - start;
    // printf("Elapsed time without CUDA Graphs: %f seconds\n", elapsed.count());

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
