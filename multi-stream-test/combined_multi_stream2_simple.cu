// Standard headers
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation

// Local headers
#include "../cuda_check.h"

#define NSTEP 10000
#define SKIPBY 0

// Kernel functions
__global__ void kernelA(double* arrayA, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] *= 2.0; }
}

__global__ void kernelB(int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayB[x] = 3; }
}

__global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] += arrayB[x]; }
}

__global__ void kernelD(double* arrayA, size_t size) {
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayA[x] += 2.0; }
}

__global__ void kernelE(int* arrayB, size_t size) {
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayB[x] += 2; }
}

struct set_vector_args {
    double* h_array;
    double value;
    size_t size;
};

void CUDART_CB set_vector(void* args) {
    set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
    double* array = h_args->h_array;
    size_t size = h_args->size;
    double value = h_args->value;

    // Initialize h_array with the specified value
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
    }

    // Do NOT delete h_args here
}

// Function for non-graph implementation with multiple streams
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout) {
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_array, arraySize * sizeof(double)));

    // Initialize host array
    std::fill_n(h_array, arraySize, initValue);
    // h_array.assign(h_array.size(), initValue);

    // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_arrayA, arraySize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_arrayB, arraySize * sizeof(int)));

    // Set Timer for first run
    cudaEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&firstCreateStart));
    CUDA_CHECK(cudaEventCreate(&firstCreateStop));

    // START measuring first run time
    CUDA_CHECK(cudaEventRecord(firstCreateStart, stream1));

    // Copy h_array to device on stream1
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, stream1));

    // Use events to synchronize between streams
    cudaEvent_t event1, event2;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));

    // Launch kernelA on stream1
    kernelA<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

    // Record event1 after kernelA in stream1
    CUDA_CHECK(cudaEventRecord(event1, stream1));

    kernelD<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

    // Make stream2 wait for event1
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event1, 0));

    // Launch kernelB on stream2
    kernelB<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);

    kernelE<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);

    // Record event2 after kernelB in stream2
    CUDA_CHECK(cudaEventRecord(event2, stream2));

    // Make stream1 wait for event2
    CUDA_CHECK(cudaStreamWaitEvent(stream1, event2, 0));

    // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
    kernelC<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host on stream1
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream1));

    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Wait for all operations to complete
    CUDA_CHECK(cudaEventRecord(firstCreateStop, stream1));
    CUDA_CHECK(cudaEventSynchronize(firstCreateStop));
    CUDA_CHECK(cudaEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Execute the sequence multiple times
    for(int i = 0; i < NSTEP; ++i){

        // Reinitialize host array
        std::fill_n(h_array, arraySize, initValue);

        CUDA_CHECK(cudaEventRecord(execStart, stream1));

        // Copy h_array to device on stream1
        CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, stream1));

        // Launch kernelA on stream1
        kernelA<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

        // Record event1 after kernelA in stream1
        CUDA_CHECK(cudaEventRecord(event1, stream1));

        kernelD<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

        // Make stream2 wait for event1
        CUDA_CHECK(cudaStreamWaitEvent(stream2, event1, 0));

        // Launch kernelB on stream2
        kernelB<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);

        kernelE<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);

        // Record event2 after kernelB in stream2
        CUDA_CHECK(cudaEventRecord(event2, stream2));

        // Make stream1 wait for event2
        CUDA_CHECK(cudaStreamWaitEvent(stream1, event2, 0));

        // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
        kernelC<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, d_arrayB, arraySize);

        // Copy data back to host on stream1
        CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream1));

        CUDA_CHECK(cudaStreamSynchronize(stream1));

        // Wait for all operations to complete
        CUDA_CHECK(cudaEventRecord(execStop, stream1));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

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
    float meanTime = (totalTime + firstCreateTime) / NSTEP;
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = sqrt(varianceTime);

    // Print out the time statistics
    std::cout << "=======Setup (No Graph)=======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Skip By: " << SKIPBY << std::endl;
    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    // Verify results
    // constexpr double expected = initValue * 2.0 + 3;
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    std::cout << "Validation passed!" << " Expected " << expected << std::endl;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Index " << i << ": Expected " << expected << " got " << h_array[i] << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(firstCreateStart));
    CUDA_CHECK(cudaEventDestroy(firstCreateStop));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_arrayA));
    CUDA_CHECK(cudaFree(d_arrayB));
    CUDA_CHECK(cudaFreeHost(h_array));

    // Return total time including first run
    // return totalTime + firstCreateTime;
    *totalTimeWith = totalTime + firstCreateTime;
    *totalTimeWithout = totalTime;
}


// Function for graph implementation with multiple streams
void runWithGraph(float* totalTimeWith, float* totalTimeWithout) {
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_array, arraySize * sizeof(double)));

    // Initialize host array
    std::fill_n(h_array, arraySize, initValue);

   // Create streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_arrayA, arraySize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_arrayB, arraySize * sizeof(int)));

    // Set Timer for graph creation
    cudaEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&graphCreateStart));
    CUDA_CHECK(cudaEventCreate(&graphCreateStop));

    // Start measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStart, stream1));

    // Begin graph capture on stream1 only
    CUDA_CHECK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Copy h_array to device on stream1
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, stream1));

    // Launch kernelA on stream1
    kernelA<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

    // Use events to synchronize between streams
    // Record event1 to be used by stream2
    cudaEvent_t event1;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventRecord(event1, stream1));

    // Launch kernelD on stream1 
    kernelD<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, arraySize);

    // All operations done before going to stream2
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event1, 0));

    // Launch kernelB on stream2 (stream2 is not capturing)
    kernelB<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);

    kernelE<<<numOfBlocks, threadsPerBlock, 0, stream2>>>(d_arrayB, arraySize);
    
    // Record event2 to be used by stream1
    cudaEvent_t event2;
    CUDA_CHECK(cudaEventCreate(&event2));
    CUDA_CHECK(cudaEventRecord(event2, stream2));

    // Waiting for event2 in stream 1
    // CUDA_CHECK(cudaStreamWaitEvent(stream1, event1, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream1, event2, 0));


    // Launch kernelC on stream1 (depends on d_arrayA and d_arrayB)
    kernelC<<<numOfBlocks, threadsPerBlock, 0, stream1>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host on stream1
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream1));

    // End graph capture
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream1, &graph));

    // Create an executable graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Destroy the graph template if not needed
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Stop measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStop, stream1));
    CUDA_CHECK(cudaEventSynchronize(graphCreateStop));
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Launch the graph multiple times
    for(int i = 0; i < NSTEP; ++i){
        //Reinitialize host array
        std::fill_n(h_array, arraySize, initValue);

        CUDA_CHECK(cudaEventRecord(execStart, stream1));

        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream1));

        // Wait for all operations to complete
        CUDA_CHECK(cudaEventRecord(execStop, stream1));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

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
    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = (initValue * 2.0 + 2.0) + (3 + 2);
    std::cout << "Validation passed!" << " Expected " << expected << std::endl;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Index " << i << ": Expected " << expected << " got " << h_array[i] << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_arrayA));
    CUDA_CHECK(cudaFree(d_arrayB));
    CUDA_CHECK(cudaFreeHost(h_array));

    // Return total time including graph creation
    // return totalTime + graphCreateTime;
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main() {
    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    // float nonGraphTotalTime = runWithoutGraph();
    runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    // float graphTotalTime = runWithGraph();
    runWithGraph(&graphTotalTime, &graphTotalTimeWithout);
    
    // std::cout << "Tests: " << std::endl;
    // std::cout << "NonGraph with: " << graphTotalTime << " ms" << std::endl;
    // std::cout << "NonGraph without: " << graphTotalTimeWithout << " ms" << std::endl;

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
