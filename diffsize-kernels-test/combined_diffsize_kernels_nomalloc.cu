// Standard headers
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation

// Local headers
#include "../cuda_check.h"

#define DEFAULT_NSTEP 100000
#define DEFAULT_SKIPBY 0

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

__global__ void kernelD(float* arrayD, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayD[x] = sinf(arrayD[x]); }
}

__global__ void kernelE(int* arrayE, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayE[x] += 5; }
}

// struct set_vector_args {
//     double* h_array;
//     double value;
//     size_t size;
// };

// void CUDART_CB set_vector(void* args) {
//     set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
//     double* array = h_args->h_array;
//     size_t size = h_args->size;
//     double value = h_args->value;

//     // Initialize h_array with the specified value
//     for (size_t i = 0; i < size; ++i) {
//         array[i] = value;
//     }

//     // Do NOT delete h_args here
// }`

// Function for non-graph implementation
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    // Define different array sizes
    constexpr size_t arraySizeA = 1U << 20; // 1,048,576 elements
    constexpr size_t arraySizeB = 1U << 18; // 262,144 elements
    constexpr size_t arraySizeC = 1U << 16; // 65,536 elements
    constexpr size_t arraySizeD = 1U << 17; // 131,072 elements
    constexpr size_t arraySizeE = 1U << 19; // 524,288 elements

    constexpr int threadsPerBlock = 256;

    // Compute the number of blocks for each kernel
    const int numBlocksA = (arraySizeA + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksB = (arraySizeB + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksC = (arraySizeC + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksD = (arraySizeD + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksE = (arraySizeE + threadsPerBlock - 1) / threadsPerBlock;

    constexpr double initValue = 2.0;

    // Host and device memory
    double* d_arrayA;
    int* d_arrayB;
    double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA = nullptr;
    int* h_arrayB = nullptr;
    float* h_arrayD = nullptr;
    int* h_arrayE = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayA, arraySizeA * sizeof(double)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayB, arraySizeB * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayD, arraySizeD * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayE, arraySizeE * sizeof(int)));

    // Initialize host arrays
    for (size_t i = 0; i < arraySizeA; i++) {
        h_arrayA[i] = initValue;
    }
    for (size_t i = 0; i < arraySizeB; i++) {
        h_arrayB[i] = 1;
    }
    for (size_t i = 0; i < arraySizeD; i++) {
        h_arrayD[i] = static_cast<float>(i) * 0.01f;
    }
    for (size_t i = 0; i < arraySizeE; i++) {
        h_arrayE[i] = 1;
    }

    // Set Timer for first run
    cudaEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&firstCreateStart));
    CUDA_CHECK(cudaEventCreate(&firstCreateStop));

    // Create a stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate device memory
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayC, arraySizeC * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE * sizeof(int), stream));

    // Start measuring first run time
    CUDA_CHECK(cudaEventRecord(firstCreateStart, stream));

    // Copy h_arrayA to device
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch kernels
    kernelA<<<numBlocksA, threadsPerBlock, 0, stream>>>(d_arrayA, arraySizeA);
    kernelB<<<numBlocksB, threadsPerBlock, 0, stream>>>(d_arrayB, arraySizeB);
    kernelC<<<numBlocksC, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySizeC);
    kernelD<<<numBlocksD, threadsPerBlock, 0, stream>>>(d_arrayD, arraySizeD);
    kernelE<<<numBlocksE, threadsPerBlock, 0, stream>>>(d_arrayE, arraySizeE);

    // Copy data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), cudaMemcpyDeviceToHost, stream));
    // Wait for all operations to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Stop measuring first run time
    CUDA_CHECK(cudaEventRecord(firstCreateStop, stream));
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

        // Initialize host arrays
        for (size_t j = 0; j < arraySizeA; j++) {
            h_arrayA[j] = initValue;
        }
        for (size_t j = 0; j < arraySizeB; j++) {
            h_arrayB[j] = 1;
        }
        for (size_t j = 0; j < arraySizeD; j++) {
            h_arrayD[j] = static_cast<float>(j) * 0.01f;
        }
        for (size_t j = 0; j < arraySizeE; j++) {
            h_arrayE[j] = 1;
        }

        CUDA_CHECK(cudaEventRecord(execStart, stream));

        // Copy h_arrayA to device
        CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Launch kernels
        kernelA<<<numBlocksA, threadsPerBlock, 0, stream>>>(d_arrayA, arraySizeA);
        kernelB<<<numBlocksB, threadsPerBlock, 0, stream>>>(d_arrayB, arraySizeB);
        kernelC<<<numBlocksC, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySizeC);
        kernelD<<<numBlocksD, threadsPerBlock, 0, stream>>>(d_arrayD, arraySizeD);
        kernelE<<<numBlocksE, threadsPerBlock, 0, stream>>>(d_arrayE, arraySizeE);

        // Copy data back to host
        CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), cudaMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventRecord(execStop, stream));
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
    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
    std::cout << "=======Results (No Graph)=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    std::cout << "h_arrayA: " << h_arrayA[5] << std::endl;
    std::cout << "h_arrayD: " << h_arrayD[5] << std::endl;
    std::cout << "h_arrayE: " << h_arrayE[5] << std::endl;

    // Verify results (simple check for demonstration purposes)
    constexpr double expectedA = initValue * 2.0 + 3; // For kernelA and kernelC
    bool passed = true;
    for(size_t i = 0; i < arraySizeA; i++){
        if(h_arrayA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << h_arrayA[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFreeAsync(d_arrayA, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayC, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayD, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayE, stream));
    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(firstCreateStart));
    CUDA_CHECK(cudaEventDestroy(firstCreateStop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_arrayA));
    CUDA_CHECK(cudaFreeHost(h_arrayB));
    CUDA_CHECK(cudaFreeHost(h_arrayD));

    // Return total time including first run
    *totalTimeWith = totalTime + firstCreateTime;
    *totalTimeWithout = totalTime;
}

// Function for graph implementation
void runWithGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    // Define different array sizes
    constexpr size_t arraySizeA = 1U << 20; // 1,048,576 elements
    constexpr size_t arraySizeB = 1U << 18; // 262,144 elements
    constexpr size_t arraySizeC = 1U << 16; // 65,536 elements
    constexpr size_t arraySizeD = 1U << 17; // 131,072 elements
    constexpr size_t arraySizeE = 1U << 19; // 524,288 elements

    constexpr int threadsPerBlock = 256;

    // Compute the number of blocks for each kernel
    const int numBlocksA = (arraySizeA + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksB = (arraySizeB + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksC = (arraySizeC + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksD = (arraySizeD + threadsPerBlock - 1) / threadsPerBlock;
    const int numBlocksE = (arraySizeE + threadsPerBlock - 1) / threadsPerBlock;

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA = nullptr;
    int* h_arrayB = nullptr;
    float* h_arrayD = nullptr;
    int* h_arrayE = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayA, arraySizeA * sizeof(double)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayB, arraySizeB * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayD, arraySizeD * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayE, arraySizeE * sizeof(int)));

    cudaStream_t captureStream;
    CUDA_CHECK(cudaStreamCreate(&captureStream));

    // set_vector_args* argsA = new set_vector_args{h_arrayA, initValue, arraySizeA};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, argsA));
    // Initialize host arrays
    for (size_t i = 0; i < arraySizeA; i++) {
        h_arrayA[i] = initValue;
    }
    for (size_t i = 0; i < arraySizeB; i++) {
        h_arrayB[i] = 1;
    }
    for (size_t i = 0; i < arraySizeD; i++) {
        h_arrayD[i] = static_cast<float>(i) * 0.01f;
    }
    for (size_t i = 0; i < arraySizeE; i++) {
        h_arrayE[i] = 1;
    }


    // Set Timer for graph creation
    cudaEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&graphCreateStart));
    CUDA_CHECK(cudaEventCreate(&graphCreateStop));

    // Start measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStart, captureStream));

    // Allocate device memory asynchronously
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA * sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB * sizeof(int), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayC, arraySizeC * sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD * sizeof(float), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE * sizeof(int), captureStream));

    // Start capturing operations
    CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

    // set_vector_args* argsA = new set_vector_args{h_arrayA, initValue, arraySizeA};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, argsA));

    // Copy arrays to device
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA * sizeof(double), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB * sizeof(int), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD * sizeof(float), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE * sizeof(int), cudaMemcpyHostToDevice, captureStream));

    // Launch kernels
    kernelA<<<numBlocksA, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySizeA);
    kernelB<<<numBlocksB, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySizeB);
    kernelC<<<numBlocksC, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySizeC);
    kernelD<<<numBlocksD, threadsPerBlock, 0, captureStream>>>(d_arrayD, arraySizeD);
    kernelE<<<numBlocksE, threadsPerBlock, 0, captureStream>>>(d_arrayE, arraySizeE);

    // Copy data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA * sizeof(double), cudaMemcpyDeviceToHost, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD * sizeof(float), cudaMemcpyDeviceToHost, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE * sizeof(int), cudaMemcpyDeviceToHost, captureStream));

    // Stop capturing
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));

    // Create an executable graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Destroy the graph template if not needed
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Stop measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStop, captureStream));
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
        // Initailize host arrays
        for (size_t j = 0; j < arraySizeA; j++) {
            h_arrayA[j] = initValue;
        }
        for (size_t j = 0; j < arraySizeB; j++) {
            h_arrayB[j] = 1;
        }
        for (size_t j = 0; j < arraySizeD; j++) {
            h_arrayD[j] = static_cast<float>(j) * 0.01f;
        }
        for (size_t j = 0; j < arraySizeE; j++) {
            h_arrayE[j] = 1;
        }

        CUDA_CHECK(cudaEventRecord(execStart, captureStream));

        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
        CUDA_CHECK(cudaStreamSynchronize(captureStream));

        CUDA_CHECK(cudaEventRecord(execStop, captureStream));
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
    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
    std::cout << "=======Results (With Graph)=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Verify results (simple check for demonstration purposes)
    constexpr double expectedA = initValue * 2.0 + 3; // For kernelA and kernelC
    bool passed = true;
    for(size_t i = 0; i < arraySizeA; i++){
        if(h_arrayA[i] != expectedA){
            passed = false;
            std::cerr << "Validation failed! Expected " << expectedA << " got " << h_arrayA[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Free device memory asynchronously
    CUDA_CHECK(cudaFreeAsync(d_arrayA, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayC, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayD, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayE, captureStream));
    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(captureStream));
    CUDA_CHECK(cudaFreeHost(h_arrayA));
    CUDA_CHECK(cudaFreeHost(h_arrayB));
    CUDA_CHECK(cudaFreeHost(h_arrayD));

    // Return total time including graph creation
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;

    // Measure time for non-graph implementation
    float nonGraphTotalTime, nonGraphTotalTimeWithout;
    runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout, NSTEP, SKIPBY);

    // Measure time for graph implementation
    float graphTotalTime, graphTotalTimeWithout;
    runWithGraph(&graphTotalTime, &graphTotalTimeWithout, NSTEP, SKIPBY);

    // Compute the difference
    float difference = nonGraphTotalTime - graphTotalTime;
    float diffPerKernel = difference / NSTEP;
    float diffPercentage = (difference / nonGraphTotalTime) * 100;

    // Compute the difference without including graph creation time
    float difference2 = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    float diffPerKernel2 = difference2 / (NSTEP-1);
    float diffPercentage2 = (difference2 / nonGraphTotalTimeWithout) * 100;

    // Print the differences
    std::cout << "=======Comparison without Graph Creation=======" << std::endl;
    std::cout << "Difference: " << difference2 << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

    // Print the differences including graph creation time
    std::cout << "=======Comparison=======" << std::endl;
    std::cout << "Difference: " << difference << " ms" << std::endl;
    std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
    std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

    return 0;
}
