#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include "cuda_check.h"

#define N (1<<12)//1024  // Size of the arrays

#define NSTEP 10000//2000
//#define NKERNEL 1

// CUDA kernel to add 10 arrays element-wise
__global__ void add_arrays(float *a1, float *a2, float *a3, float *a4, float *a5,
                           float *a6, float *a7, float *a8, float *a9, float *a10,
                           float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        result[i] = a1[i] + a2[i] + a3[i] + a4[i] + a5[i]
                  + a6[i] + a7[i] + a8[i] + a9[i] + a10[i];
    }
}

int main() {
    const int size = N * sizeof(float);

    // Allocate host memory
    float *h_a1 = (float*)malloc(size);
    float *h_a2 = (float*)malloc(size);
    float *h_a3 = (float*)malloc(size);
    float *h_a4 = (float*)malloc(size);
    float *h_a5 = (float*)malloc(size);
    float *h_a6 = (float*)malloc(size);
    float *h_a7 = (float*)malloc(size);
    float *h_a8 = (float*)malloc(size);
    float *h_a9 = (float*)malloc(size);
    float *h_a10 = (float*)malloc(size);
    float *h_result = (float*)malloc(size);

    // Initialize host arrays with values
    for (int i = 0; i < N; i++) {
        h_a1[i] = 1.0f;
        h_a2[i] = 2.0f;
        h_a3[i] = 3.0f;
        h_a4[i] = 4.0f;
        h_a5[i] = 5.0f;
        h_a6[i] = 6.0f;
        h_a7[i] = 7.0f;
        h_a8[i] = 8.0f;
        h_a9[i] = 9.0f;
        h_a10[i] = 10.0f;
        h_result[i] = 0.0f;  // Initialize result array to zero
    }

    // Allocate device memory
    float *d_a1, *d_a2, *d_a3, *d_a4, *d_a5;
    float *d_a6, *d_a7, *d_a8, *d_a9, *d_a10;
    float *d_result;

    cudaMalloc((void**)&d_a1, size);
    cudaMalloc((void**)&d_a2, size);
    cudaMalloc((void**)&d_a3, size);
    cudaMalloc((void**)&d_a4, size);
    cudaMalloc((void**)&d_a5, size);
    cudaMalloc((void**)&d_a6, size);
    cudaMalloc((void**)&d_a7, size);
    cudaMalloc((void**)&d_a8, size);
    cudaMalloc((void**)&d_a9, size);
    cudaMalloc((void**)&d_a10, size);
    cudaMalloc((void**)&d_result, size);

    // Set Timer
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    float graphCreateTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 100;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // cudaMallocAsync((void**)&d_a1, size, stream);
    // cudaMallocAsync((void**)&d_a2, size, stream);
    // cudaMallocAsync((void**)&d_a3, size, stream);
    // cudaMallocAsync((void**)&d_a4, size, stream);
    // cudaMallocAsync((void**)&d_a5, size, stream);
    // cudaMallocAsync((void**)&d_a6, size, stream);
    // cudaMallocAsync((void**)&d_a7, size, stream);
    // cudaMallocAsync((void**)&d_a8, size, stream);
    // cudaMallocAsync((void**)&d_a9, size, stream);
    // cudaMallocAsync((void**)&d_a10, size, stream);
    // cudaMallocAsync((void**)&d_result, size, stream);
    CUDA_CHECK(cudaEventRecord(start, stream));
    //Graph
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Copy host arrays to device arrays asynchronously
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a1, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a2, h_a2, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a3, h_a3, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a4, h_a4, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a5, h_a5, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a6, h_a6, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a7, h_a7, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a8, h_a8, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a9, h_a9, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a10, h_a10, size, cudaMemcpyHostToDevice, stream));

    // Launch kernel to add arrays on the created stream
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_a1, d_a2, d_a3, d_a4, d_a5,
        d_a6, d_a7, d_a8, d_a9, d_a10,
        d_result
    );

    // Copy result back to host asynchronously
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream));
    
    // End Capture
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    
    CUDA_CHECK(cudaEventRecord(stop, stream)); 
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, start, stop));
    
    for(int istep = 0; istep < NSTEP-1; istep++) {
        // Modifying buffers
        for (int i = 0; i < N; i++) {
            h_a1[i] += 1.0f; // or any other modification
            h_a2[i] += 1.0f;
            h_a3[i] += 1.0f;
            h_a4[i] += 1.0f;
            h_a5[i] += 1.0f;
            h_a6[i] += 1.0f;
            h_a7[i] += 1.0f;
            h_a8[i] += 1.0f;
            h_a9[i] += 1.0f;
            h_a10[i] += 1.0f;
        }

        // Start Timer
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch Graph
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        // Synchronize the stream to ensure all operations are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // for(int j = 0; j < N; j++) {
        //  printf("RESULT %d: %f\n", j, h_result[j]);
        // }
        // End Timer
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        if(istep >= skipBy){
            totalTime += elapsedTime;
            if(elapsedTime > upperTime) {
                upperTime = elapsedTime;
            }
            if(elapsedTime < lowerTime) {
                lowerTime = elapsedTime;
            }
            if(istep == skipBy){
                lowerTime = elapsedTime;
            }
        }
        //std::cout << "Elapsed time " << istep << ": " << elapsedTime << "ms" << std::endl;
    }
    // Time Calculations
    float AverageTime = (totalTime + graphCreateTime) / (NSTEP - skipBy);
    std::cout << "Average Time: " << AverageTime << "ms" << std::endl;
    std::cout << "Time Spread: " << upperTime <<  " - " << lowerTime << "ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << "ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << (totalTime + graphCreateTime) << "ms" << std::endl;
    // Verify the result on the host
    int correct = 1;
    for (int i = 0; i < N; i++) {
        float expected = h_a1[i] + h_a2[i] + h_a3[i] + h_a4[i] + h_a5[i]
                       + h_a6[i] + h_a7[i] + h_a8[i] + h_a9[i] + h_a10[i];
        //printf("Test expected: %f\n", expected);
        //printf("Test result: %f\n", h_result[i]);
        if (h_result[i] != expected) {
            correct = 0;
            printf("Error at index %d: Expected %f, got %f\n", i, expected, h_result[i]);
            break;
        }
    }

    if (correct) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
    }

    // Destroy the graph and exec object
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(instance));

    // Destroy the stream
    cudaStreamDestroy(stream);

    // Free device memory
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_a3);
    cudaFree(d_a4);
    cudaFree(d_a5);
    cudaFree(d_a6);
    cudaFree(d_a7);
    cudaFree(d_a8);
    cudaFree(d_a9);
    cudaFree(d_a10);
    cudaFree(d_result);

    // Free host memory
    free(h_a1);
    free(h_a2);
    free(h_a3);
    free(h_a4);
    free(h_a5);
    free(h_a6);
    free(h_a7);
    free(h_a8);
    free(h_a9);
    free(h_a10);
    free(h_result);
    
    return 0;
}

