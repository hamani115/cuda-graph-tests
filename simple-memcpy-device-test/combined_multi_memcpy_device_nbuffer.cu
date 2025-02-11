#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include "../cuda_check.h"

// Here you can set the device ID
#define MYDEVICE 0

// Problem size
#define N (1U << 20) 
#define NSTEP 10000
#define SKIPBY 0

// ---------------------------------------------------------------------
// Kernel: Sum multiple device buffers
// ---------------------------------------------------------------------
__global__ void add_arrays(float** d_buffers, float* d_result, int nBuffers, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        float sumVal = 0.0f;
        // Sum over all buffers
        for(int bufIdx = 0; bufIdx < nBuffers; ++bufIdx) {
            sumVal += d_buffers[bufIdx][i];
        }
        d_result[i] = sumVal;
    }
}

// ---------------------------------------------------------------------
// Example function: runWithoutGraph
// Demonstrates how to do it with a dynamic number of buffers.
// ---------------------------------------------------------------------
void runWithoutGraph(int nBuffers, float* totalTimeWith, float* totalTimeWithout) {

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

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // -----------------------------------------
    // 1. Allocate an array of device pointers on the host side
    // -----------------------------------------
    std::vector<float*> devBuffers(nBuffers, nullptr);

    // 2. Allocate each device buffer
    for(int b = 0; b < nBuffers; ++b) {
        CUDA_CHECK(cudaMallocAsync(&devBuffers[b], memSize, stream));
    }

    // Also allocate the final result buffer
    float* d_result = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream));

    // 3. Create a pointer-to-pointer in device memory to hold devBuffers
    float** d_buffers = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buffers, nBuffers * sizeof(float*)));

    // Copy devBuffers array from host to device
    // This tells the kernel where each device buffer actually lives
    CUDA_CHECK(cudaMemcpyAsync(d_buffers, devBuffers.data(),
                               nBuffers * sizeof(float*),
                               cudaMemcpyHostToDevice, stream));

    // -----------------------------------------
    // Prepare to do D->D copies, etc.
    // -----------------------------------------
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timings
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 4. Copy data from host to device for the *first* buffer
    //    Then we can do device->device copies to the other buffers
    CUDA_CHECK(cudaMemcpyAsync(devBuffers[0], h_a, memSize,
                               cudaMemcpyHostToDevice, stream));

    // Start Timer for first run
    float firstTime = 0.0f;
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Perform device->device copies
    for(int b = 1; b < nBuffers; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(devBuffers[b],
                                   devBuffers[b - 1],
                                   memSize,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }

    // Single kernel launch after all memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_buffers, d_result, nBuffers, N);
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
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Repeat main loop
    for (int istep = 1; istep <= NSTEP; istep++) {
        // Start Timer for each run
        CUDA_CHECK(cudaEventRecord(start, stream));

        // D->D copies
        for(int b = 1; b < nBuffers; ++b) {
            CUDA_CHECK(cudaMemcpyAsync(devBuffers[b],
                                       devBuffers[b - 1],
                                       memSize,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        }

        // Launch kernel
        add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_buffers, d_result, nBuffers, N
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy back to host
        CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

        // Wait for completion
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Stop Timer
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Collect stats
        if (istep > SKIPBY) {
            totalTime += elapsedTime;
            count++;

            // Welford's online mean and variance
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;
        }
    }

    // Compute final statistics
    float meanTime = (totalTime + firstTime) / (NSTEP + 1 - SKIPBY);
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = std::sqrt(varianceTime);

    // Print summary
    std::cout << "======= Setup (No Graph, " << nBuffers << " buffers) =======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "First Run Time: " << firstTime << " ms" << std::endl;
    std::cout << "Average Time (including first): " << meanTime << " ms" << std::endl;
    std::cout << "Average Time (excluding first): "
              << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Std Dev: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;

    // Verify
    for (int i = 0; i < N; ++i) {
        // Each buffer is basically i, and we have nBuffers of them
        float expected = i * (float)nBuffers;
        if (std::fabs(h_result[i] - expected) > 1e-7) {
            std::cerr << "Mismatch at " << i 
                      << ": got " << h_result[i]
                      << ", expected " << expected << std::endl;
            assert(false);
        }
    }

    // Clean up
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_result));

    // Free device buffers
    for(int b = 0; b < nBuffers; ++b) {
        CUDA_CHECK(cudaFree(devBuffers[b]));
    }
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_buffers));

    // Return total times
    *totalTimeWith     = totalTime + firstTime;
    *totalTimeWithout  = totalTime;
}

// ---------------------------------------------------------------------
// Example function: runWithGraph
// Similar dynamic approach but inside a CUDA graph capture.
// ---------------------------------------------------------------------
void runWithGraph(int nBuffers, float* totalTimeWith, float* totalTimeWithout)
{
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    size_t memSize = N * sizeof(float);

    // Allocate pinned host memory
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
    }

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 1. Allocate an array of device pointers on the host side
    std::vector<float*> devBuffers(nBuffers, nullptr);
    // 2. Allocate each device buffer
    for(int b = 0; b < nBuffers; ++b) {
        CUDA_CHECK(cudaMallocAsync(&devBuffers[b], memSize, stream));
    }

    // Also allocate the final result
    float* d_result = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream));

    // 3. Allocate pointer-to-pointer in device memory
    float** d_buffers = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buffers, nBuffers * sizeof(float*)));

    // Copy the array of pointers into device memory
    CUDA_CHECK(cudaMemcpyAsync(d_buffers, devBuffers.data(),
                               nBuffers * sizeof(float*),
                               cudaMemcpyHostToDevice, stream));

    // Set up kernel dims
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy host->device
    CUDA_CHECK(cudaMemcpyAsync(devBuffers[0], h_a, memSize, cudaMemcpyHostToDevice, stream));

    // Start graph creation timer
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventRecord(start, stream));

    // Begin capture
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Device->device copies
    for(int b = 1; b < nBuffers; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(devBuffers[b],
                                   devBuffers[b - 1],
                                   memSize,
                                   cudaMemcpyDeviceToDevice, 
                                   stream));
    }

    // Kernel
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_buffers, d_result, nBuffers, N
    );

    // D->H
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // End capture
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // First graph launch
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    // End Timer for graph creation
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, start, stop));

    // Timing stats
    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Main loop
    for (int istep = 1; istep <= NSTEP; istep++) {
        // Start event
        CUDA_CHECK(cudaEventRecord(start, stream));

        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Stop event
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

        // Gather stats
        if (istep > SKIPBY) {
            totalTime += elapsedTime;
            count++;

            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;
        }
    }

    float meanTime = (totalTime + graphCreateTime) / (NSTEP + 1 - SKIPBY);
    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    double stdDevTime = std::sqrt(varianceTime);

    // Print
    std::cout << "======= Setup (With Graph, " << nBuffers << " buffers) =======" << std::endl;
    std::cout << "Iterations: " << NSTEP << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
    std::cout << "Average Time (with creation): " << meanTime << " ms" << std::endl;
    std::cout << "Average Time (without creation): "
              << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime << " ms^2" << std::endl;
    std::cout << "Std Dev: " << stdDevTime << " ms" << std::endl;
    std::cout << "Time Spread: " << lowerTime << " - " << upperTime << " ms" << std::endl;

    // Verify
    for (int i = 0; i < N; ++i) {
        float expected = i * (float)nBuffers;
        if (std::fabs(h_result[i] - expected) > 1e-7) {
            std::cerr << "Mismatch at " << i 
                      << ": got " << h_result[i]
                      << ", expected " << expected << std::endl;
            assert(false);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaGraphDestroy(graph));
    
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_result));
    
    for(int b = 0; b < nBuffers; ++b) {
        CUDA_CHECK(cudaFree(devBuffers[b]));
    }
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_buffers));

    *totalTimeWith    = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

// ---------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Default to 10 buffers, but allow the user to pass a different number
    int nBuffers = 10;
    if(argc > 1) {
        nBuffers = std::atoi(argv[1]);
    }
    std::cout << "Running with " << nBuffers << " buffers\n";

    // Measure time for non-graph
    float nonGraphTotalTime = 0.f;
    float nonGraphTotalTimeWithout = 0.f;
    runWithoutGraph(nBuffers, &nonGraphTotalTime, &nonGraphTotalTimeWithout);

    // Measure time for graph
    float graphTotalTime = 0.f;
    float graphTotalTimeWithout = 0.f;
    runWithGraph(nBuffers, &graphTotalTime, &graphTotalTimeWithout);

    // Simple comparison
    float difference   = nonGraphTotalTime - graphTotalTime;
    float diffPerRun   = difference / NSTEP;
    float diffPercent  = (difference / nonGraphTotalTime) * 100.f;

    float difference2  = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    float diffPerRun2  = difference2 / NSTEP;
    float diffPercent2 = (difference2 / nonGraphTotalTimeWithout) * 100.f;

    std::cout << "\n=======Comparison (Excluding Graph Creation)=======\n";
    std::cout << "Difference: " << difference2 << " ms\n";
    std::cout << "Difference per step: " << diffPerRun2 << " ms\n";
    std::cout << "Difference percentage: " << diffPercent2 << "%\n";

    std::cout << "=======Comparison (Including Graph Creation)=======\n";
    std::cout << "Difference: " << difference << " ms\n";
    std::cout << "Difference per step: " << diffPerRun << " ms\n";
    std::cout << "Difference percentage: " << diffPercent << "%\n";

    return 0;
}
