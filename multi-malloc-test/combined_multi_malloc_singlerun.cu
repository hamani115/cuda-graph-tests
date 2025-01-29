// Standard headers
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
#include <algorithm>
#include <chrono>
#include <string>

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

__global__ void kernelD(float* arrayD, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayD[x] = sinf(arrayD[x]); }
}

__global__ void kernelE(int* arrayE, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){ arrayE[x] += 5; }
}

std::vector<int> generateSequence(int N) {
    std::vector<int> sequence;
    int current = 5; // Starting point
    bool multiplyByTwo = true; 
    
    while (current <= N) {
        sequence.push_back(current);
        if (multiplyByTwo) {
            current *= 2;
        } else {
            current *= 5;
        }
        multiplyByTwo = !multiplyByTwo;
    }
    return sequence;
}

// Modified runWithoutGraph and runWithGraph to store timing results in arrays for selected NSTEP:

void runWithoutGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                     std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                     std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                     int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    // Define different array sizes
    constexpr size_t arraySizeA = 1U << 20; // 1,048,576 elements
    constexpr size_t arraySizeB = 1U << 18; // 262,144 elements
    constexpr size_t arraySizeC = 1U << 16; // 65,536 elements
    constexpr size_t arraySizeD = 1U << 17; // 131,072 elements
    constexpr size_t arraySizeE = 1U << 19; // 524,288 elements

    constexpr int threadsPerBlock = 256;

    const int numBlocksA = (int)((arraySizeA + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksB = (int)((arraySizeB + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksC = (int)((arraySizeC + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksD = (int)((arraySizeD + threadsPerBlock - 1) / threadsPerBlock);
    const int numBlocksE = (int)((arraySizeE + threadsPerBlock - 1) / threadsPerBlock);

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    // double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA;
    int* h_arrayB;
    float* h_arrayD;
    int* h_arrayE;
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayA, arraySizeA * sizeof(double)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayB, arraySizeB * sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayD, arraySizeD * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayE, arraySizeE * sizeof(int)));

    for (size_t i = 0; i < arraySizeA; i++) h_arrayA[i] = initValue;
    for (size_t i = 0; i < arraySizeB; i++) h_arrayB[i] = 1;
    for (size_t i = 0; i < arraySizeD; i++) h_arrayD[i] = static_cast<float>(i)*0.01f;
    for (size_t i = 0; i < arraySizeE; i++) h_arrayE[i] = 1;

    cudaEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&firstCreateStart));
    CUDA_CHECK(cudaEventCreate(&firstCreateStop));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
    // // CUDA_CHECK(cudaMallocAsync(&d_arrayC, arraySizeC*sizeof(double), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

    CUDA_CHECK(cudaEventRecord(firstCreateStart, stream));

    // Start chrono
    const auto graphStart = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayC, arraySizeC*sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), cudaMemcpyHostToDevice, stream));

    kernelA<<<numBlocksA, threadsPerBlock, 0, stream>>>(d_arrayA, arraySizeA);
    kernelB<<<numBlocksB, threadsPerBlock, 0, stream>>>(d_arrayB, arraySizeB);
    kernelC<<<numBlocksC, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySizeC);
    kernelD<<<numBlocksD, threadsPerBlock, 0, stream>>>(d_arrayD, arraySizeD);
    kernelE<<<numBlocksE, threadsPerBlock, 0, stream>>>(d_arrayE, arraySizeE);

    CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_arrayA, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, stream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayC, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayD, stream));
    CUDA_CHECK(cudaFreeAsync(d_arrayE, stream));

    const auto graphEnd = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(firstCreateStop, stream));
    CUDA_CHECK(cudaEventSynchronize(firstCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));

    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    std::chrono::duration<double> totalTimeChrono(0.0);
    std::chrono::duration<double> totalLunchTimeChrono(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    for (int i = 1; i <= NSTEP; i++) {
        for (size_t j = 0; j < arraySizeA; j++) h_arrayA[j] = initValue;
        for (size_t j = 0; j < arraySizeB; j++) h_arrayB[j] = 1;
        for (size_t j = 0; j < arraySizeD; j++) h_arrayD[j] = static_cast<float>(j)*0.01f;
        for (size_t j = 0; j < arraySizeE; j++) h_arrayE[j] = 1;

        CUDA_CHECK(cudaEventRecord(execStart, stream));
        const auto start = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA*sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB*sizeof(int), stream));
        // CUDA_CHECK(cudaMallocAsync(&d_arrayC, arraySizeC*sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD*sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE*sizeof(int), stream));

        CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), cudaMemcpyHostToDevice, stream));

        kernelA<<<numBlocksA, threadsPerBlock, 0, stream>>>(d_arrayA, arraySizeA);
        kernelB<<<numBlocksB, threadsPerBlock, 0, stream>>>(d_arrayB, arraySizeB);
        kernelC<<<numBlocksC, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySizeC);
        kernelD<<<numBlocksD, threadsPerBlock, 0, stream>>>(d_arrayD, arraySizeD);
        kernelE<<<numBlocksE, threadsPerBlock, 0, stream>>>(d_arrayE, arraySizeE);

        CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), cudaMemcpyDeviceToHost, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFreeAsync(d_arrayA, stream));
        CUDA_CHECK(cudaFreeAsync(d_arrayB, stream));
        // CUDA_CHECK(cudaFreeAsync(d_arrayC, stream));
        CUDA_CHECK(cudaFreeAsync(d_arrayD, stream));
        CUDA_CHECK(cudaFreeAsync(d_arrayE, stream));

        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(execStop, stream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
            totalTime += elapsedTime;

            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (auto num: nsteps) {
                if (num == i) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + firstCreateTime) / i;
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    // Print out the ti(totalTime + firstCreateTime) / ime statistics
                    std::cout << "=======Setup (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
                    // std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
                    std::cout << "=======Results (No Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "First Run: " << firstCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with firstRun: " << (totalTime + firstCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without firstRun: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

                    float totalTimeWith = totalTime + firstCreateTime;
                     // CHRONO
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;
                    
                    chronoTotalTimeWithArr.push_back((totalTimeWithChrono.count()*1000.0));
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000.0);
                    chronoTotalLaunchTimeWithArr.push_back((totalLunchTimeWithChrono.count()*1000.0));
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000.0);
                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                }
            }
        }
    }

    // Verify
    constexpr double expectedA = initValue * 2.0 + 3;
    bool passed = true;
    for (size_t i = 0; i < arraySizeA; i++) {
        if (h_arrayA[i] != expectedA) {
            passed = false;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // CUDA_CHECK(cudaFreeAsync(d_arrayA, stream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayB, stream));
    // // CUDA_CHECK(cudaFreeAsync(d_arrayC, stream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayD, stream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayE, stream));

    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(firstCreateStart));
    CUDA_CHECK(cudaEventDestroy(firstCreateStop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeHost(h_arrayA));
    CUDA_CHECK(cudaFreeHost(h_arrayB));
    CUDA_CHECK(cudaFreeHost(h_arrayD));
    CUDA_CHECK(cudaFreeHost(h_arrayE));
}

void runWithGraph(std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,
                  std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,
                  std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                  int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;

    constexpr size_t arraySizeA = 1U << 20;
    constexpr size_t arraySizeB = 1U << 18;
    constexpr size_t arraySizeC = 1U << 16;
    constexpr size_t arraySizeD = 1U << 17;
    constexpr size_t arraySizeE = 1U << 19;
    constexpr int threadsPerBlock = 256;

    const int numBlocksA = (int)((arraySizeA + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksB = (int)((arraySizeB + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksC = (int)((arraySizeC + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksD = (int)((arraySizeD + threadsPerBlock - 1)/threadsPerBlock);
    const int numBlocksE = (int)((arraySizeE + threadsPerBlock - 1)/threadsPerBlock);

    constexpr double initValue = 2.0;

    double* d_arrayA;
    int* d_arrayB;
    // double* d_arrayC;
    float* d_arrayD;
    int* d_arrayE;

    double* h_arrayA;
    int* h_arrayB;
    float* h_arrayD;
    int* h_arrayE;
    
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayA, arraySizeA*sizeof(double)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayB, arraySizeB*sizeof(int)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayD, arraySizeD*sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_arrayE, arraySizeE*sizeof(int)));

    for (size_t i = 0; i < arraySizeA; i++) h_arrayA[i] = initValue;
    for (size_t i = 0; i < arraySizeB; i++) h_arrayB[i] = 1;
    for (size_t i = 0; i < arraySizeD; i++) h_arrayD[i] = static_cast<float>(i)*0.01f;
    for (size_t i = 0; i < arraySizeE; i++) h_arrayE[i] = 1;

    cudaStream_t captureStream;
    CUDA_CHECK(cudaStreamCreate(&captureStream));

    cudaEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&graphCreateStart));
    CUDA_CHECK(cudaEventCreate(&graphCreateStop));

    // CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA*sizeof(double), captureStream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB*sizeof(int), captureStream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD*sizeof(float), captureStream));
    // CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE*sizeof(int), captureStream));

    CUDA_CHECK(cudaEventRecord(graphCreateStart, captureStream));
    const auto graphStart = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySizeA*sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySizeB*sizeof(int), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayD, arraySizeD*sizeof(float), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayE, arraySizeE*sizeof(int), captureStream));

    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_arrayA, arraySizeA*sizeof(double), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayB, h_arrayB, arraySizeB*sizeof(int), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayD, h_arrayD, arraySizeD*sizeof(float), cudaMemcpyHostToDevice, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(d_arrayE, h_arrayE, arraySizeE*sizeof(int), cudaMemcpyHostToDevice, captureStream));

    kernelA<<<numBlocksA, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySizeA);
    kernelB<<<numBlocksB, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySizeB);
    kernelC<<<numBlocksC, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySizeC);
    kernelD<<<numBlocksD, threadsPerBlock, 0, captureStream>>>(d_arrayD, arraySizeD);
    kernelE<<<numBlocksE, threadsPerBlock, 0, captureStream>>>(d_arrayE, arraySizeE);

    CUDA_CHECK(cudaMemcpyAsync(h_arrayA, d_arrayA, arraySizeA*sizeof(double), cudaMemcpyDeviceToHost, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayD, d_arrayD, arraySizeD*sizeof(float), cudaMemcpyDeviceToHost, captureStream));
    CUDA_CHECK(cudaMemcpyAsync(h_arrayE, d_arrayE, arraySizeE*sizeof(int), cudaMemcpyDeviceToHost, captureStream));

    CUDA_CHECK(cudaFreeAsync(d_arrayA, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayD, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayE, captureStream));

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(captureStream, &graph));

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));

    // First Graph Launch
    CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));

    const auto graphEnd = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(graphCreateStop, captureStream));
    CUDA_CHECK(cudaEventSynchronize(graphCreateStop));
    const auto graphEnd2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, graphCreateStart, graphCreateStop));

    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    std::chrono::duration<double> totalTimeChrono(0.0);
    std::chrono::duration<double> totalLunchTimeChrono(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    for (int i = 1; i <= NSTEP; i++) {
        for (size_t j = 0; j < arraySizeA; j++) h_arrayA[j] = initValue;
        for (size_t j = 0; j < arraySizeB; j++) h_arrayB[j] = 1;
        for (size_t j = 0; j < arraySizeD; j++) h_arrayD[j] = static_cast<float>(j)*0.01f;
        for (size_t j = 0; j < arraySizeE; j++) h_arrayE[j] = 1;

        CUDA_CHECK(cudaEventRecord(execStart, captureStream));
        const auto start = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
        // CUDA_CHECK(cudaStreamSynchronize(captureStream));

        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(execStop, captureStream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        if (i >= SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
            totalTime += elapsedTime;

            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;
            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (auto num : nsteps) {
                if (num == i) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + graphCreateTime) / i;
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);
                    
                    // Print out the time statistics
                    std::cout << "=======Setup (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Iterations: " << i << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC, kernelD, kernelE" << std::endl;
                    // std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Sizes: " << arraySizeA << ", " << arraySizeB << ", " << arraySizeC << ", " << arraySizeD << ", " << arraySizeE << std::endl;
                    std::cout << "=======Results (With Graph) for NSTEP " << i << "=======" << std::endl;
                    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with Graph: " << (totalTime + graphCreateTime) / (i + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without Graph: " << (totalTime / (i - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Standard Deviation: " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
                    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;
                    float totalTimeWith = totalTime + graphCreateTime;
                    // CHRONO
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + graphCreateTimeChrono2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + graphCreateTimeChrono;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << graphCreateTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << graphCreateTimeChrono2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;

                    chronoTotalTimeWithArr.push_back((totalTimeWithChrono.count()*1000.0));
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count()*1000.0);
                    chronoTotalLaunchTimeWithArr.push_back((totalLunchTimeWithChrono.count()*1000.0));
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count()*1000.0);
                    totalTimeWithArr.push_back(totalTimeWith);
                    totalTimeWithoutArr.push_back(totalTime);
                }
            }
        }
    }

    // Verify
    constexpr double expectedA = initValue * 2.0 + 3;
    bool passed = true;
    for (size_t i = 0; i < arraySizeA; i++) {
        if (h_arrayA[i] != expectedA) {
            passed = false;
            break;
        }
    }
    if(passed){
        std::cerr << "Validation passed." << std::endl;
    }

    // CUDA_CHECK(cudaFreeAsync(d_arrayA, captureStream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayB, captureStream));
    // // CUDA_CHECK(cudaFreeAsync(d_arrayC, captureStream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayD, captureStream));
    // CUDA_CHECK(cudaFreeAsync(d_arrayE, captureStream));

    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(captureStream));

    CUDA_CHECK(cudaFreeHost(h_arrayA));
    CUDA_CHECK(cudaFreeHost(h_arrayB));
    CUDA_CHECK(cudaFreeHost(h_arrayD));
    CUDA_CHECK(cudaFreeHost(h_arrayE));
}


int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;
    const int NUM_RUNS = (argc > 3) ? std::atoi(argv[3]) : 4;

    std::cout << "==============COMPLEX DIFFERENT SIZES KERNELS TEST==============" << std::endl;
    std::vector<int> nsteps = generateSequence(NSTEP);
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
        std::cout << "==============FOR RUN" << r+1 << "==============" << std::endl;

        std::vector<float> noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr;

        runWithoutGraph(noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr,
                        NSTEP, SKIPBY);

        std::vector<float> graphTotalTimeWithArr, graphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr;

        runWithGraph(graphTotalTimeWithArr, graphTotalTimeWithoutArr, 
                     chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr,
                     chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr,
                     NSTEP, SKIPBY);

        for (int i = 0; i < (int)nsteps.size(); i++) {
            float difference = noneGraphTotalTimeWithArr[i] - graphTotalTimeWithArr[i];
            float diffPerKernel = difference / (nsteps[i] + 1);
            float diffPercentage = (difference / noneGraphTotalTimeWithArr[i]) * 100;

            float difference2 = noneGraphTotalTimeWithoutArr[i] - graphTotalTimeWithoutArr[i];
            float diffPerKernel2 = difference2 / (nsteps[i]);
            float diffPercentage2 = (difference2 / noneGraphTotalTimeWithoutArr[i]) * 100;

            float chronoDiffTotalTimeWith = chronoNoneGraphTotalTimeWithArr[i] - chronoGraphTotalTimeWithArr[i];
            float chronoDiffTotalTimeWithout = chronoNoneGraphTotalTimeWithoutArr[i] - chronoGraphTotalTimeWithoutArr[i];

            float chronoDiffPerStepWith = chronoDiffTotalTimeWith / (nsteps[i] + 1); 
            float chronoDiffPercentWith = (chronoDiffTotalTimeWith / chronoNoneGraphTotalTimeWithArr[i]) * 100;

            float chronoDiffPerStepWithout = chronoDiffTotalTimeWithout / (nsteps[i]); 
            float chronoDiffPercentWithout = (chronoDiffTotalTimeWithout / chronoNoneGraphTotalTimeWithoutArr[i]) * 100;

            float chronoDiffLaunchTimeWith = chronoNoneGraphTotalLaunchTimeWithArr[i] - chronoGraphTotalLaunchTimeWithArr[i];
            float chronoDiffLaunchTimeWithout = chronoNoneGraphTotalLaunchTimeWithoutArr[i] - chronoGraphTotalLaunchTimeWithoutArr[i];

            float chronoDiffLaunchPercentWithout = (chronoDiffLaunchTimeWithout / chronoNoneGraphTotalLaunchTimeWithoutArr[i]) * 100;
            float chronoDiffLaunchPercentWith = (chronoDiffLaunchTimeWith / chronoNoneGraphTotalLaunchTimeWithArr[i]) * 100;

            std::cout << "==============For NSTEP "<< nsteps[i] << "==============" << std::endl;
            std::cout << "=======Comparison without Graph Creation=======" << std::endl;
            std::cout << "Difference: " << difference2 << " ms" << std::endl;
            std::cout << "Difference per step: " << diffPerKernel2 << " ms" << std::endl;
            std::cout << "Difference percentage: " << diffPercentage2 << "%" << std::endl;

            std::cout << "=======Comparison=======" << std::endl;
            std::cout << "Difference: " << difference << " ms" << std::endl;
            std::cout << "Difference per step: " << diffPerKernel << " ms" << std::endl;
            std::cout << "Difference percentage: " << diffPercentage << "%" << std::endl;

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
        }
    }

    const std::string FILENAME = "complex_multi_malloc.csv";
    // for (const auto &newData : newDatas) {
    //     updateOrAppendCSV(FILENAME, newData);
    // }
    rewriteCSV(FILENAME, newDatas, NUM_RUNS);
    return 0;
}
