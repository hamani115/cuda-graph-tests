#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(expression)                \
{                                             \
    const cudaError_t status = expression;    \
    if(status != cudaSuccess){                \
        std::cerr << "CUDA error "            \
                  << status << ": "           \
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

// struct set_vector_args{
//     // std::vector<double>* h_array;
//     std::vector<double>& h_array;
//     double value;
// };


// void CUDART_CB set_vector(void* args) {
//     // attempt 1
//     // set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

//     // std::vector<double>& vec{h_args.h_array};
//     // vec.assign(vec.size(), h_args.value);
//     //attempt 2
//     // set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
//     // std::vector<double>* vec = h_args->h_array;
//     // vec->assign(vec->size(), h_args->value);
//     // delete h_args;  // Free the dynamically allocated memory
//     // attempt 3
//     set_vector_args* h_args = reinterpret_cast<set_vector_args*>(args);
//     std::vector<double>& vec = h_args->h_array;
//     vec.assign(vec.size(), h_args->value);
//     // delete h_args;  // Free the dynamically allocated memory
// }

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

int main(){
    constexpr int numOfBlocks = 1024;
    constexpr int threadsPerBlock = 1024;
    constexpr size_t arraySize = 1U << 20;

    double* d_arrayA;
    int* d_arrayB;
    // std::vector<double> h_array(arraySize);
    double* h_array = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_array, arraySize * sizeof(double)));

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
    CUDA_CHECK(cudaMallocAsync(&d_arrayA, arraySize * sizeof(double), captureStream));
    CUDA_CHECK(cudaMallocAsync(&d_arrayB, arraySize * sizeof(int), captureStream));

    // Assign host function to the stream

    // Attempt 1: Needs a custom struct to pass the arguments
    // set_vector_args args{h_array, initValue};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, &args));

    // Attempt 2: Dynamically allocate args to ensure it remains valid
    // set_vector_args* args = new set_vector_args{&h_array, initValue};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, args));

    // Attempt 3: Dynamically allocate args to ensure it remains valid
    // set_vector_args* args = new set_vector_args{h_array, initValue};
    // CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, args));
    set_vector_args* args = new set_vector_args{h_array, initValue, arraySize};
    CUDA_CHECK(cudaLaunchHostFunc(captureStream, set_vector, args));


    CUDA_CHECK(cudaMemcpyAsync(d_arrayA, h_array, arraySize * sizeof(double), cudaMemcpyHostToDevice, captureStream));

    kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
    kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
    kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

    CUDA_CHECK(cudaMemcpyAsync(h_array, d_arrayA, arraySize * sizeof(*d_arrayA), cudaMemcpyDeviceToHost, captureStream));

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
    // float execTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));
    float elapsedTime = 0.0f;
    // float graphCreateTime = 0.0f;
    float totalTime = 0.0f;
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    int skipBy = 0;
    // Variables `for Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // CUDA_CHECK(cudaEventRecord(execStart, captureStream));

    // Launch the graph multiple times
    constexpr int iterations = 1000;
    for(int i = 0; i < iterations; ++i){
        CUDA_CHECK(cudaEventRecord(execStart, captureStream));

        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
        CUDA_CHECK(cudaStreamSynchronize(captureStream));

        CUDA_CHECK(cudaEventRecord(execStop, captureStream));
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
    // CUDA_CHECK(cudaStreamSynchronize(captureStream));

    // CUDA_CHECK(cudaEventRecord(execStop, captureStream));
    // CUDA_CHECK(cudaEventSynchronize(execStop));
    // CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

    // Calculate mean and standard deviation
    float meanTime = (totalTime + graphCreateTime) / (iterations - skipBy);
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
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Skip By: " << skipBy << std::endl;
    std::cout << "Kernel: " << "kernelA, kernelB, kernelC" << std::endl;
    std::cout << "Number of Blocks: " << numOfBlocks << std::endl;
    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
    std::cout << "Array Size: " << arraySize << std::endl;
    std::cout << "=======Results=======" << std::endl;
    std::cout << "Graph Creation Time: " << graphCreateTime << "ms" << std::endl;
    std::cout << "Average Time with Graph: " << meanTime << " ms" << std::endl;
    std::cout << "Average Time without Graph: " << (totalTime) / (iterations - 1 - skipBy) << " ms" << std::endl;
    std::cout << "Variance: " << varianceTime3 << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdDevTime3 << " ms" << std::endl;
    std::cout << "Time Spread: " << upperTime << " - " << lowerTime << " ms" << std::endl;
    std::cout << "Total Time without Graph Creation: " << totalTime << " ms" << std::endl;
    std::cout << "Total Time with Graph Creation: " << totalTime + graphCreateTime << " ms" << std::endl;

    // std::cout << "Old measurements: " << std::endl;
    // std::cout << "Average Execution Time per Iteration: " << (execTime / iterations) << "ms" << std::endl;
    // std::cout << "Total Time: " << graphCreateTime + execTime << "ms" << std::endl;
    // std::cout << "Average Execution Time per Iteration: " << ((execTime + graphCreateTime) / (iterations)) << "ms" << std::endl;
    

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
        std::cerr << "Validation passed."<< expected << " : " << h_array[arraySize - 1] << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(execStart));
    CUDA_CHECK(cudaEventDestroy(execStop));
    CUDA_CHECK(cudaEventDestroy(graphCreateStart));
    CUDA_CHECK(cudaEventDestroy(graphCreateStop));
    // Free graph and stream resources after usage
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaStreamDestroy(captureStream));
    delete args;
    CUDA_CHECK(cudaFreeHost(h_array));


    return 0;
}
