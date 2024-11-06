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
    if(x < size){arrayA[x] *= 2.0;}
}

__global__ void kernelB(int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayB[x] = 3;}
}

__global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
    const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    if(x < size){arrayA[x] += arrayB[x];}
}

struct set_vector_args{
    std::vector<double>& h_array;
    double value;
};

void CUDART_CB set_vector(void* args){
    set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

    std::vector<double>& vec{h_args.h_array};
    vec.assign(vec.size(), h_args.value);
}

int main(){
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;

    // This example assumes that kernelA operates on data that needs to be initialized on
    // and copied from the host, while kernelB initializes the array that is passed to it.
    // Both arrays are then used as input to kernelC, where arrayA is also used as
    // output, that is copied back to the host, while arrayB is only read from and not modified.

    double* d_arrayA;
    int* d_arrayB;
    std::vector<double> h_array(arraySize);
    constexpr double initValue = 2.0;

    // Set Timer for graph creation
    cudaEvent_t graphCreateStart, graphCreateStop;
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&graphCreateStart));
    CUDA_CHECK(cudaEventCreate(&graphCreateStop));

    cudaStream_t captureStream;
    CUDA_CHECK(cudaStreamCreate(&captureStream));

    // Start measuring graph creation time
    CUDA_CHECK(cudaEventRecord(graphCreateStart, captureStream));

    // ##### Start capturing operations
    CUDA_CHECK(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal));

    // cudaMallocAsync and cudaMemcpyAsync are needed, to be able to assign it to a stream
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySize*sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySize*sizeof(int), captureStream));

    // Assign host function to the stream
    // Needs a custom struct to pass the arguments
    set_vector_args args{h_array, initValue};
    CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, &args));

    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array.data(), arraySize*sizeof(double), cudaMemcpyHostToDevice, captureStream));

    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    CUDA_CHECK(cudaMemcpyAsync(h_array.data(), d_arrayA, arraySize*sizeof(*d_arrayA), cudaMemcpyDeviceToHost, captureStream));

    CUDA_CHECK(cudaFreeAsync(d_arrayA, captureStream));
    CUDA_CHECK(cudaFreeAsync(d_arrayB, captureStream));

    // ###### Stop capturing
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

    // Now measure the execution time separately
    cudaEvent_t execStart, execStop;
    float execTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    CUDA_CHECK(cudaEventRecord(execStart, captureStream));

    // Launch the graph multiple times
    constexpr int iterations = 1000;
    for(int i = 0; i < iterations; ++i){
        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
    }

    CUDA_CHECK(cudaEventRecord(execStop, captureStream));
    CUDA_CHECK(cudaEventSynchronize(execStop));
    CUDA_CHECK(cudaEventElapsedTime(&execTime, execStart, execStop));

    std::cout << "Graph Creation Time: " << graphCreateTime << "ms" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Average Execution Time per Iteration: " << (execTime / iterations) << "ms" << std::endl;
    std::cout << "Total Time: " << graphCreateTime + execTime << "ms" << std::endl;
    std::cout << "Average Execution Time per Iteration: " << ((execTime + graphCreateTime) / (iterations + 1)) << "ms" << std::endl;

    // Verify results
    constexpr double expected = initValue * 2.0 + 3;
    bool passed = true;
    for(size_t i = 0; i < arraySize; ++i){
            if(h_array[i] != expected){
                    passed = false;
                    std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
                    break;
            }
    }
    if(passed){
            std::cerr << "Validation passed." << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    // Free graph and stream resources after usage
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(captureStream));

    return 0;
}
