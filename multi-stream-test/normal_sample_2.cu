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
    float execTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    CUDA_CHECK(cudaEventRecord(execStart, stream));

    // Execute the sequence multiple times
    constexpr int iterations = 1000;
    for(int i = 0; i < iterations; ++i){
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
    }

    CUDA_CHECK(cudaEventRecord(execStop, stream));
    CUDA_CHECK(cudaEventSynchronize(execStop));
    CUDA_CHECK(cudaEventElapsedTime(&execTime, execStart, execStop));

    std::cout << "First Run: " << firstCreateTime << "ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Average Execution Time per Iteration: " << (execTime / iterations) << "ms" << std::endl;
    std::cout << "Total Time: " << execTime + firstCreateTime << "ms" << std::endl;
    std::cout << "New Average Execution Time per Iteration: " << ((execTime + firstCreateTime) / (iterations + 1)) << "ms" << std::endl;

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
