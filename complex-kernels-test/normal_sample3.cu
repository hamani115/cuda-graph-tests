#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(expression)                \
{                                             \
    const cudaError_t status = expression;    \
    if(status != cudaSuccess){                \
            std::cerr << "CUDA error "        \
                << status << ": "             \
                << cudaGetErrorString(status) \
                << " at " << __FILE__ << ":"  \
                << __LINE__ << std::endl;     \
    }                                         \
}

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

int main(){
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    // Set Timer for graph creation
    cudaEvent_t firstCreateStart, firstCreateStop;
    float firstCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&firstCreateStart));
    CUDA_CHECK(cudaEventCreate(&firstCreateStop));

    // Create a stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Start measuring graph creation time
    CUDA_CHECK(cudaEventRecord(firstCreateStart, stream));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_arrayA, arraySize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_arrayB, arraySize * sizeof(int)));

    // Initialize host array
    h_array.assign(h_array.size(), initValue);

    // Copy h_array to device
    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array.data(), arraySize * sizeof(double), cudaMemcpyHostToDevice, stream));

    // Launch kernels
    kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

    // Copy data back to host
    CUDA_CHECK(cudaMemcpyAsync(h_array.data(), d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // Wait for all operations to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free device memory
    CUDA_CHECK(cudaFree(d_arrayA));
    CUDA_CHECK(cudaFree(d_arrayB));

    // Stop measuring graph creation time
    CUDA_CHECK(cudaEventRecord(firstCreateStop, stream));
    CUDA_CHECK(cudaEventSynchronize(firstCreateStop));
    CUDA_CHECK(cudaEventElapsedTime(&firstCreateTime, firstCreateStart, firstCreateStop));

    // Measure execution time
    cudaEvent_t execStart, execStop;
    // float execTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    float elapsedTime = 0.0f;
    // float graphCreateTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 0;
    // Variables for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // CUDA_CHECK(cudaEventRecord(execStart, stream));

    // Execute the sequence multiple times
    constexpr int iterations = 1000;
    for(int i = 0; i < iterations; ++i){
        CUDA_CHECK(cudaEventRecord(execStart, stream));
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_arrayA, arraySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_arrayB, arraySize * sizeof(int)));

        // Initialize host array
        h_array.assign(h_array.size(), initValue);

        // Copy h_array to device
        CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array.data(), arraySize * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Launch kernels
        kernelA<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, arraySize);
        kernelB<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayB, arraySize);
        kernelC<<<numOfBlocks, threadsPerBlock, 0, stream>>>(d_arrayA, d_arrayB, arraySize);

        // Copy data back to host
        CUDA_CHECK(cudaMemcpyAsync(h_array.data(), d_arrayA, arraySize * sizeof(double), cudaMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Free device memory
        CUDA_CHECK(cudaFree(d_arrayA));
        CUDA_CHECK(cudaFree(d_arrayB));

        CUDA_CHECK(cudaEventRecord(execStop, stream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        // Time calculations
        if (i >= skipBy) {
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
            // Uncomment to see elapsed time per iteration
            // std::cout << "Elapsed time " << i << ": " << elapsedTime << " ms" << std::endl;
        }
    }

    // CUDA_CHECK(cudaEventRecord(execStop, stream));
    // CUDA_CHECK(cudaEventSynchronize(execStop));
    // CUDA_CHECK(cudaEventElapsedTime(&execTime, execStart, execStop));44

    // Calculate mean and standard deviation
    float meanTime = (totalTime + firstCreateTime) / (iterations - skipBy);
    double varianceTime3 = 0.0;
    if (count > 1) {
        varianceTime3 = M2 / (count - 1);
    }
    // Ensure variance is not negative due to floating-point errors
    if (varianceTime3 < 0.0) {
        varianceTime3 = 0.0;
    }
    double stdDevTime3 = sqrt(varianceTime3);
    
    std::cout << "=======Setup=======" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Skip By: " << skipBy << std::endl;
    std::cout << "Kernel: " << "kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "First Run: " << firstCreateTime << "ms" << std::endl;
    std::cout << "Average Time with firstRun: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without firstRun: " << ((totalTime) / (iterations - 1 - skipBy))  << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime3 << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime3 << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without firstRun: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with firstRun: " << totalTime + firstCreateTime << " ms" << std::endl;

    // std::cout << "Old measurements: " << std::endl;
    // std::cout << "First Run: " << firstCreateTime << "ms" << std::endl;
    // std::cout << "Iterations: " << iterations << std::endl;
    // std::cout << "Average Execution Time per Iteration: " << (execTime / iterations) << "ms" << std::endl;
    // std::cout << "Total Time: " << execTime + firstCreateTime << "ms" << std::endl;
    // std::cout << "New Average Execution Time per Iteration: " << ((execTime + firstCreateTime) / (iterations)) << "ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
        if(h_array[i] != expected){
            passed = false;
            std::cerr << "Validation failed! Expected " << expected << " got " << h_array[i] << " at index " << i << std::endl;
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
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
