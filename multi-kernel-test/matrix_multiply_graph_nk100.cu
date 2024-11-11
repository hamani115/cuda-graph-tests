#include <stdio.h>
// #include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Local headers
#include "../cuda_check.h"

#define N 64  // Matrix dimensions (64x64)

#define NSTEP 100000
#define NKERNEL 100  // INDEPENDENT VARIABLE: CHANGE THE NUMBER OF KERNELS (10 OR 100)

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

// Function to perform matrix multiplication using CUDA Graphs
void matrixMultiplyWithGraph(float* A, float* B, float* C, int width) {
    // Define block and grid sizes
    dim3 block(32, 32); // 1024 threads
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    // dim3 grid(6,6); // 36 Blocks

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
    int skipBy = 0;
    // std::vector<float> times;
    // float sumTime = 0.0f;          // For calculating mean
    // float sumTimeSquared = 0.0f;   // For calculating variance
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Start recording time for graph creation
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin graph capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Enqueue kernel launches into the graph
    for (int i = 0; i < NKERNEL; i++) {
        matMulKernel<<<grid, block, 0, stream>>>(A, B, C, width);
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
    // times.push_back(graphCreateTime);
    // sumTimeSquared += graphCreateTime * graphCreateTime;

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
        if (i >= skipBy) {
            totalTime += elapsedTime;
            // times.push_back(elapsedTime);
            // sumTime += elapsedTime;
            // sumTimeSquared += elapsedTime * elapsedTime;

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
            // Uncomment to see elapsed time per iteration
            // std::cout << "Elapsed time " << i << ": " << elapsedTime << " ms" << std::endl;
        }
    }

    // Calculate mean and standard deviation
    float meanTime = (totalTime + graphCreateTime) / (NSTEP - skipBy);
    // float sumSq = 0.0f;
    // for (int i = 0; i < times.size(); i++) {
    //     sumSq += (times[i] - meanTime) * (times[i] - meanTime);
    // }
    // float varianceTime = sumTimeSquared / (NSTEP - skipBy);
    // float stdDevTime = sqrt(varianceTime);
    
    // float varianceTime2 = sumSq / (NSTEP - skipBy);
    // float stdDevTime2 = sqrt(varianceTime2);
    
    // Naive Algo
    // float varianceTime = (sumTimeSquared / (NSTEP - skipBy)) - (meanTime * meanTime);
    // float stdDevTime = sqrt(varianceTime);

    //  Welford's algorithm for calculating variance
    double varianceTime3 = 0.0;
    if (count > 1) {
        varianceTime3 = M2 / (count - 1);
    }

    // Ensure variance is not negative due to floating-point errors
    if (varianceTime3 < 0.0) {
        varianceTime3 = 0.0;
    }
    double stdDevTime3 = sqrt(varianceTime3);

    // Print out the time statistics
    std::cout << "=======Setup=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << skipBy << std::endl;
    std::cout << "Kernels: " << NKERNEL << std::endl;
    std::cout << "Block Size: " << block.x << " x " << block.y << std::endl;
    std::cout << "Grid Size: " << grid.x << " x " << grid.y << std::endl;
    std::cout << "Matrix Size: " << width << " x " << width << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "Graph Creation: " << graphCreateTime << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - skipBy)) << " ms" << std::endl;
    // std::cout << "Variance: " << varianceTime << " ms" << std::endl;
    // std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    // std::cout << "Variance2: " << varianceTime2 << " ms" << std::endl;
    // std::cout << "Standard Deviation2: " << stdDevTime2 << " ms" << std::endl;
    std::cout << "Variance3: " <<  varianceTime3 << meanTime << " ms" << std::endl;
    std::cout << "Standard Deviation3: " << stdDevTime3 << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(stream));
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

    // Initialize matrices with random values
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
    matrixMultiplyWithGraph(d_A, d_B, d_C, N);
    // auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate elapsed time
    // std::chrono::duration<double> elapsed = end - start;
    // printf("CHRONO: Elapsed time with CUDA Graphs: %f seconds\n", elapsed.count());

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
