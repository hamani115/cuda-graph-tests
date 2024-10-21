#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#include "cuda_check.h"

#define N 64//(1<<6) // Matrix dimensions (4096x4096)

#define NSTEP 100000//10
#define NKERNEL 100

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

void matrixMultiplyNoGraph2(float* A, float* B, float* C, int width) {
    dim3 block(32, 32); // 1024 threads
    // dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    dim3 grid(6,6); // 36 Blocks

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    float firstTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f; 
    float lowerTime = 0.0f; 
    int skipBy = 0; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop); 

    CUDA_CHECK(cudaEventRecord(start, stream)); 
    for (int i = 0; i < NKERNEL; i++) {  // Run 100 iterations
        matMulKernel<<<grid, block, 0, stream>>>(A, B, C, width);
    }
    // Synchronize after all kernels have been launched
    cudaStreamSynchronize(stream); // Ensure all kernels finish
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop)); 
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop)); 

    for (int j = 0; j < NSTEP-1; j++) {

        CUDA_CHECK(cudaEventRecord(start, stream));  
        for (int i = 0; i < NKERNEL; i++) {  // Run 100 iterations
            matMulKernel<<<grid, block, 0, stream>>>(A, B, C, width);
        }
        // Synchronize after all kernels have been launched
        cudaStreamSynchronize(stream); // Ensure all kernels finish
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop)); 
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop)); 
        if(j >= skipBy){ 
            totalTime += elapsedTime; 
            if(elapsedTime > upperTime) { 
                upperTime = elapsedTime;
            }   
            if(elapsedTime < lowerTime) {
                lowerTime = elapsedTime; 
            } 
            if(j == skipBy){
                lowerTime = elapsedTime; 
            }   
        }  
    }
    float AverageTime = (totalTime+firstTime) / (NSTEP - skipBy);
    std::cout << "Average Time: " << AverageTime << "ms" << std::endl;
    std::cout << "Time Spread: " << upperTime <<  " - " << lowerTime << "ms" << std::endl;
    std::cout << "Total Time without first run: " << totalTime << "ms" << std::endl;
    std::cout << "Total Time with first run: " << (totalTime+firstTime) << "ms" << std::endl;

    cudaStreamDestroy(stream);
}


int main() {
    // Allocate host memory
    float* h_A = (float*)malloc(N * N * sizeof(float));
    float* h_B = (float*)malloc(N * N * sizeof(float));
    float* h_C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyNoGraph2(d_A, d_B, d_C, N);
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    printf("Elapsed time without CUDA Graphs: %f seconds\n", elapsed.count());

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}

