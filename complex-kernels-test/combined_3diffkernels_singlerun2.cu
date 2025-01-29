#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
#include <algorithm> // For std::find
#include <chrono>

// Local headers
#include "../cuda_check.h"
#include "../util/csv_util.h"

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

std::vector<int> generateSequence(int N) {
    std::vector<int> sequence;
    int current = 5; // Starting point
    bool multiplyByTwo = true; // Flag to alternate between multiplying by 2 and 5
    
    while (current <= N) {
        sequence.push_back(current);
        
        if (multiplyByTwo) {
            current *= 2;
        } else {
            current *= 5;
        }
        
        multiplyByTwo = !multiplyByTwo; // Toggle the multiplier for next iteration
    }
    
    return sequence;
}

// Function for non-graph implementation
// void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
void runWithoutGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr, std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr, std::vector<float>& chronoTotalLaunchTimeWithArr,std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                int nstep, int skipby) {
    // Declare constants inside the function
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    // constexpr int iterations = 1000;
    constexpr double initValue = 2.0;

    // Host and device memory
    double* d_arrayA;
    int* d_arrayB;
    // std::vector<double> h_array(arraySize);
    double* h_array = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_array, arraySize * sizeof(double)));

    // Initialize host array using index i
    for (size_t i = 0; i < arraySize; i++) {
        h_array[i] = initValue;
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
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySize * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySize * sizeof(int), stream));

    // Start measuring first run time
    CUDA_CHECK(cudaEventRecord(firstCreateStart, stream));
    const auto graphStart = std::chrono::steady_clock::now();

    // Copy h_array to device
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, stream));

    // Launch kernels
    kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Wait for all operations to complete
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto graphEnd = std::chrono::steady_clock::now();
    // Stop measuring first run time
    CUDA_CHECK(cudaEventRecord(firstCreateStop, stream));
    CUDA_CHECK(cudaEventSynchronize(firstCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));
    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    std::chrono::duration<double> totalTimeChrono = std::chrono::duration<double>(0.0);
    std::chrono::duration<double> totalLunchTimeChrono = std::chrono::duration<double>(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // int skipBy = 0;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    // Execute the sequence multiple times
    for(int i = 1; i <= NSTEP; i++){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        CUDA_CHECK(cudaEventRecord(execStart, stream));
        const auto start = std::chrono::steady_clock::now();

        // Initialize host array using index i
        // for (size_t j = 0; j < arraySize; ++j) {
        //     h_array[j] = static_cast<double>(j);
        // }
        // Initialize host array
        // h_array.assign(h_array.size(), initValue);

        // Copy h_array to device
        CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Launch kernels
        kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
        kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
        kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

        // Copy data back to host
        CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        // CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(execStop, stream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
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

            for (const auto& num : nsteps) {
                if (num == i) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + firstCreateTime) / i;
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    // Print out the time statistics
                    std::cout << "=======Setup (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Size: " << arraySize << std::endl;
                    std::cout << "=======Results (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with firstRun: " << (totalTime + firstCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without firstRun: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;
                    // CHRONO
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;
                    
                    totalTimeWithArr.push_back(totalTime + firstCreateTime);
                    totalTimeWithoutArr.push_back(totalTime);
                    chronoTotalTimeWithArr.push_back(totalTimeWithChrono.count() * 1000.0);
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count() * 1000.0);
                    chronoTotalLaunchTimeWithArr.push_back(totalLunchTimeWithChrono.count() * 1000.0);
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count() * 1000.0);
                }
            }
        }
    }

    // // Calculate mean and standard deviation
    // float meanTime = (totalTime + firstCreateTime) / NSTEP;
    // double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    // double stdDevTime = sqrt(varianceTime);

    // // Print out the time statistics
    // std::cout << "=======Setup (No Graph)=======" << std::endl;
    // std::cout << "Iterations: " << NSTEP << std::endl;
    // std::cout << "Skip By: " << SKIPBY << std::endl;
    // std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    // std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    // std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    // std::cout << "Array Size: " << arraySize << std::endl;
    // std::cout << "=======Results (No Graph)=======" << std::endl;
    // std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
    // std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    // std::cout << "Average Time without firstRun: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    // std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    // std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    // std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    // std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    // std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; i++){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Expected " << expected << " got " << h_array[i] << " at index " << i << std::endl;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFreeAsync(d_arrayA, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, stream));
    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(firstCreateStart));
    CUDA_CHECK(cudaEventDestroy(firstCreateStop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Return total time including first run
    // return totalTime + firstCreateTime;
    // *totalTimeWith = totalTime + firstCreateTime;
    // *totalTimeWithout = totalTime;
}

// Function for graph implementation
// void runWithGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
void runWithGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr, std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr, std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                int nstep, int skipby) {
    // Declare constants inside the function
    const int NSTEP = nstep;
    const int SKIPBY = skipby;
    
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;
    // constexpr int iterations = 1000;
    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    double* h_array = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_array, arraySize * sizeof(double)));

    // Initialize host array using index i
    for (size_t i = 0; i < arraySize; i++) {
        h_array[i] = initValue;
    }

    // Set Timer for graph creation
    cudaEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&graphCreateStart));
    CUDA_CHECK(cudaEventCreate(&graphCreateStop));

    cudaStream_t captureStream;
    CUDA_CHECK(cudaStreamCreate(&captureStream));

    // Allocate device memory asynchronously
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySize * sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySize * sizeof(int), captureStream));

    // Start measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStart, captureStream));
    const auto graphStart = std::chrono::steady_clock::now();

    // Start capturing operations
    CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

    // set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, args));

    // Copy h_array to device
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, captureStream));

    // Launch kernels
    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, captureStream));

    // Stop capturing
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));

    // Create an executable graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Destroy the graph template if not needed
    CUDA_CHECK(cudaGraphDestroy(graph));

    // First Graph Launch
    CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));

    const auto graphEnd = std::chrono::steady_clock::now();
    // Stop measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStop, captureStream));
    CUDA_CHECK(cudaEventSynchronize(graphCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));
    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    std::chrono::duration<double> totalTimeChrono = std::chrono::duration<double>(0.0);
    std::chrono::duration<double> totalLunchTimeChrono = std::chrono::duration<double>(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // int skipBy = 0;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // std::vector<int> nsteps_base = {10000, 5000, 1000, 500, 100, 50, 10, 5};
    std::vector<int> nsteps = generateSequence(NSTEP);
    
    // Filter NSTEP values based on input
    // for (int ns : nsteps_base) {
    //     if (ns <= NSTEP) {
    //         nsteps.push_back(ns);
    //     }
    // }

    // Launch the graph multiple times
    for(int i = 1; i <= NSTEP; i++){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }
        
        CUDA_CHECK(cudaEventRecord(execStart, captureStream));
        const auto start = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
        // CUDA_CHECK(cudaStreamSynchronize(captureStream));

        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(execStop, captureStream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
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

            for (const auto& num : nsteps) {
                if (num == i) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + graphCreateTime) / i;
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    // Print out the time statistics
                    std::cout << "=======Setup (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Size: " << arraySize << std::endl;
                    std::cout << "=======Results (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with Graph: " << (totalTime + graphCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without Graph: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;
                    // CHRONO
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;
                    
                    totalTimeWithArr.push_back(totalTime + graphCreateTime);
                    totalTimeWithoutArr.push_back(totalTime);
                    chronoTotalTimeWithArr.push_back(totalTimeWithChrono.count() * 1000);
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count() * 1000);
                    chronoTotalLaunchTimeWithArr.push_back(totalLunchTimeWithChrono.count() * 1000);
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count() * 1000);

                }
            }
            
        }
    }

    // Calculate mean and standard deviation
    // float meanTime = (totalTime + graphCreateTime) / NSTEP;
    // double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    // double stdDevTime = sqrt(varianceTime);

    // // Print out the time statistics
    // std::cout << "=======Setup (With Graph)=======" << std::endl;
    // std::cout << "Iterations: " << NSTEP << std::endl;
    // std::cout << "Skip By: " << SKIPBY << std::endl;
    // std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
    // std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    // std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    // std::cout << "Array Size: " << arraySize << std::endl;
    // std::cout << "=======Results (With Graph)=======" << std::endl;
    // std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    // std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    // std::cout << "Average Time without Graph: " << (totalTime / (NSTEP - 1 - SKIPBY)) << " ms" << std::endl;
    // std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    // std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
    // std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
    // std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    // std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
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

    // Free device memory asynchronously
    CUDA_CHECK(cudaFreeAsync(d_arrayA, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, captureStream));
    // Clean up
    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(captureStream));
    CUDA_CHECK(cudaFreeHost(h_array));

    // Return total time including graph creation
    // return totalTime + graphCreateTime;
    // *totalTimeWith = totalTime + graphCreateTime;
    // *totalTimeWithout = totalTime;
}

int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;
    const int NUM_RUNS = (argc > 3) ? std::atoi(argv[3]) : 4;

    std::cout << "==============COMPLEX 3 DIFFERENT KERNELS TEST==============" << std::endl;
    std::vector<int> nsteps = generateSequence(NSTEP);
    // const int NUM_RUNS = 4;
    std::vector<CSVData> newDatas(nsteps.size());
    for (auto &newData : newDatas) {
        // Resize each vector to hold 'NUM_RUNS' elements
        newData.noneGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.GraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.noneGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.GraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.DiffTotalWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffPerStepWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.DiffTotalWith.resize(NUM_RUNS, 0.0f);
        newData.DiffPerStepWith.resize(NUM_RUNS, 0.0f);
        newData.DiffPercentWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoNoneGraphTotalLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoGraphTotalLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffTotalTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPerStepWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffTotalTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPerStepWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffPercentWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchTimeWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchPercentWithout.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchTimeWith.resize(NUM_RUNS, 0.0f);
        newData.ChronoDiffLaunchPercentWith.resize(NUM_RUNS, 0.0f);
    }

    for (int r = 0; r < NUM_RUNS; r++) {
        std::cout << "==============FOR RUN"<< r+1 << "==============" << std::endl;

        std::vector<float> noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr;
        // Measure time for non-graph implementation
        runWithoutGraph(noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr,
                        NSTEP, SKIPBY);

        std::vector<float> graphTotalTimeWithArr, graphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr;
        // Measure time for graph implementation
        runWithGraph(graphTotalTimeWithoutArr, graphTotalTimeWithArr, 
                    chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr,
                    chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr,
                    NSTEP, SKIPBY);


        for (int i = 0; i < nsteps.size(); i++) {
            // Compute the difference
            float difference = noneGraphTotalTimeWithArr[i] - graphTotalTimeWithArr[i];
            float diffPerKernel = difference / (nsteps[i] + 1);
            float diffPercentage = (difference / noneGraphTotalTimeWithArr[i]) * 100;

            // Compute the difference for without including Graph
            float difference2 = noneGraphTotalTimeWithoutArr[i] - graphTotalTimeWithoutArr[i];
            float diffPerKernel2 = difference2 / (nsteps[i]);
            float diffPercentage2 = (difference2 / noneGraphTotalTimeWithoutArr[i]) * 100;

            // Chrono Launch + Exec Time 
            float chronoDiffTotalTimeWith = chronoNoneGraphTotalTimeWithArr[i] - chronoGraphTotalTimeWithArr[i];
            float chronoDiffTotalTimeWithout = chronoNoneGraphTotalTimeWithoutArr[i] - chronoGraphTotalTimeWithoutArr[i];
            
            float chronoDiffPerStepWith = chronoDiffTotalTimeWith / (nsteps[i] + 1); 
            float chronoDiffPercentWith = (chronoDiffTotalTimeWith / chronoNoneGraphTotalTimeWithArr[i]) * 100;

            float chronoDiffPerStepWithout = chronoDiffTotalTimeWithout / (nsteps[i]); 
            float chronoDiffPercentWithout = (chronoDiffTotalTimeWithout / chronoNoneGraphTotalTimeWithoutArr[i]) * 100;

            // Chrono Launch Time
            float chronoDiffLaunchTimeWith = chronoNoneGraphTotalLaunchTimeWithArr[i] - chronoGraphTotalLaunchTimeWithArr[i];
            float chronoDiffLaunchTimeWithout = chronoNoneGraphTotalLaunchTimeWithoutArr[i] - chronoGraphTotalLaunchTimeWithoutArr[i];

            float chronoDiffLaunchPercentWithout = (chronoDiffLaunchTimeWithout / chronoNoneGraphTotalLaunchTimeWithoutArr[i]) * 100;
            float chronoDiffLaunchPercentWith = (chronoDiffLaunchTimeWith / chronoNoneGraphTotalLaunchTimeWithArr[i]) * 100;

            std::cout << "==============For NSTEP "<< nsteps[i] << "==============" << std::endl;
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
            
            //----------------- Constructing CSV File------------------------------------
            // Make a new data entry with updated values
            newDatas[i].NSTEP = nsteps[i];
            newDatas[i].SKIPBY = SKIPBY;
            newDatas[i].noneGraphTotalTimeWithout[r] = noneGraphTotalTimeWithoutArr[i];
            newDatas[i].GraphTotalTimeWithout[r] = graphTotalTimeWithoutArr[i];
            newDatas[i].noneGraphTotalTimeWith[r] = noneGraphTotalTimeWithArr[i];
            newDatas[i].GraphTotalTimeWith[r] = graphTotalTimeWithArr[i];
            newDatas[i].DiffTotalWithout[r] = difference2;
            newDatas[i].DiffPerStepWithout[r] = diffPerKernel2;
            newDatas[i].DiffPercentWithout[r] = diffPercentage2;
            newDatas[i].DiffTotalWith[r] = difference;
            newDatas[i].DiffPerStepWith[r] = diffPerKernel;
            newDatas[i].DiffPercentWith[r] = diffPercentage;
            newDatas[i].ChronoNoneGraphTotalTimeWithout[r] = chronoNoneGraphTotalTimeWithoutArr[i];
            newDatas[i].ChronoGraphTotalTimeWithout[r] = chronoGraphTotalTimeWithoutArr[i];
            newDatas[i].ChronoNoneGraphTotalLaunchTimeWithout[r] = chronoNoneGraphTotalLaunchTimeWithoutArr[i];
            newDatas[i].ChronoGraphTotalLaunchTimeWithout[r] = chronoGraphTotalLaunchTimeWithoutArr[i];
            newDatas[i].ChronoNoneGraphTotalTimeWith[r] = chronoNoneGraphTotalTimeWithArr[i];
            newDatas[i].ChronoGraphTotalTimeWith[r] = chronoGraphTotalTimeWithArr[i];
            newDatas[i].ChronoNoneGraphTotalLaunchTimeWith[r] = chronoNoneGraphTotalLaunchTimeWithArr[i];
            newDatas[i].ChronoGraphTotalLaunchTimeWith[r] = chronoGraphTotalLaunchTimeWithArr[i];
            newDatas[i].ChronoDiffTotalTimeWithout[r] = chronoDiffTotalTimeWithout;
            newDatas[i].ChronoDiffPerStepWithout[r] = chronoDiffPerStepWithout;
            newDatas[i].ChronoDiffPercentWithout[r] = chronoDiffPercentWithout;
            newDatas[i].ChronoDiffTotalTimeWith[r] = chronoDiffTotalTimeWith;
            newDatas[i].ChronoDiffPerStepWith[r] = chronoDiffPerStepWith;
            newDatas[i].ChronoDiffPercentWith[r] = chronoDiffPercentWith;
            newDatas[i].ChronoDiffLaunchTimeWithout[r] = chronoDiffLaunchTimeWithout;
            newDatas[i].ChronoDiffLaunchPercentWithout[r] = chronoDiffLaunchPercentWithout;
            newDatas[i].ChronoDiffLaunchTimeWith[r] = chronoDiffLaunchTimeWith;
            newDatas[i].ChronoDiffLaunchPercentWith[r] = chronoDiffLaunchPercentWith;

            // const std::string FILENAME = "complex_3_different_kernels.csv";
            // updateOrAppendCSV(FILENAME, newData);
        }
    }
    std::cout << "I am here!" << '\n';
    const std::string FILENAME = "complex_3_different_kernels_runsarg.csv";
    // for (const auto &newData : newDatas) {
    //     updateOrAppendCSV(FILENAME, newData, NUM_RUNS);
    // }
    rewriteCSV(FILENAME, newDatas, NUM_RUNS);

    return 0;
}
