#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>    // For sqrt in standard deviation calculation
// 
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>



// Local headers
#include "../cuda_check.h"

#define DEFAULT_NSTEP 1000
#define DEFAULT_SKIPBY 0

struct CSVData {
    int NSTEP;
    int SKIPBY;
    float noneGraphTotalTimeWithout;
    float GraphTotalTimeWithout;
    float noneGraphTotalTimeWith;
    float GraphTotalTimeWith;
    float DiffTotalWithout;
    float DiffPerStepWithout;
    float DiffPercentWithout;
    float DiffTotalWith;
    float DiffPerStepWith;
    float DiffPercentWith;
};

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
// }

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
void runWithoutGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
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
    // int skipBy = 0;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Execute the sequence multiple times
    for(int i = 0; i < NSTEP; ++i){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }

        CUDA_CHECK(cudaEventRecord(execStart, stream));

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
    *totalTimeWith = totalTime + firstCreateTime;
    *totalTimeWithout = totalTime;
}

// Function for graph implementation
void runWithGraph(float* totalTimeWith, float* totalTimeWithout, int nstep, int skipby) {
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
    // int skipBy = 0;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // Launch the graph multiple times
    for(int i = 0; i < NSTEP; ++i){

        for (size_t j = 0; j < arraySize; j++) {
            h_array[j] = initValue;
        }
        
        CUDA_CHECK(cudaEventRecord(execStart, captureStream));

        CUDA_CHECK(cudaGraphLaunch(graphExec, captureStream));
        // CUDA_CHECK(cudaStreamSynchronize(captureStream));

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
    *totalTimeWith = totalTime + graphCreateTime;
    *totalTimeWithout = totalTime;
}

int main(int argc, char* argv[]) {
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;

    // Measure time for non-graph implementation
    // float nonGraphTotalTime = runWithoutGraph();

    // Measure time for graph implementation
    // float graphTotalTime = runWithGraph();

    std::vector<int> nsteps = generateSequence(NSTEP);
    for (const auto& num : nsteps) {

        // Measure time for non-graph implementation
        float nonGraphTotalTime, nonGraphTotalTimeWithout;
        // float nonGraphTotalTime = runWithoutGraph(N);
        runWithoutGraph(&nonGraphTotalTime, &nonGraphTotalTimeWithout, num, SKIPBY);

        // Measure time for graph implementation
        float graphTotalTime, graphTotalTimeWithout;
        // float graphTotalTime = runWithGraph(N);
        runWithGraph(&graphTotalTime, &graphTotalTimeWithout, num, SKIPBY);


        // Compute the difference
        float difference = nonGraphTotalTime - graphTotalTime;
        float diffPerKernel = difference / (num);
        float diffPercentage = (difference / nonGraphTotalTime) * 100;

        // Compute the difference for without including Graph
        float difference2 = nonGraphTotalTimeWithout - graphTotalTimeWithout;
        float diffPerKernel2 = difference2 / (num-1);
        float diffPercentage2 = (difference2 / nonGraphTotalTimeWithout) * 100;

        //-----------------------Console Output------------------------
        std::cout << "==============NSTEP " << num << "==============" << std::endl;
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
        std::vector<CSVData> csvData;
        // Open the CSV file for reading
        std::ifstream csvFileIn("timing_data.csv");
        if (csvFileIn.is_open()) {
            std::string line;
            // Skip the header line
            std::getline(csvFileIn, line);

            while (std::getline(csvFileIn, line)) {
                std::istringstream ss(line);
                CSVData data;
                std::string token;

                // Read NSTEP
                if (std::getline(ss, token, ',')) data.NSTEP = std::stoi(token);
                // Read SKIPBY
                if (std::getline(ss, token, ',')) data.SKIPBY = std::stoi(token);
                // Read noneGraphTotalTimeWithout
                if (std::getline(ss, token, ',')) data.noneGraphTotalTimeWithout = std::stod(token);
                // Read GraphTotalTimeWithout
                if (std::getline(ss, token, ',')) data.GraphTotalTimeWithout = std::stod(token);
                // Read noneGraphTotalTimeWith
                if (std::getline(ss, token, ',')) data.noneGraphTotalTimeWith = std::stod(token);
                // Read GraphTotalTimeWith
                if (std::getline(ss, token, ',')) data.GraphTotalTimeWith = std::stod(token);
                // Read DiffTotalWithout
                if (std::getline(ss, token, ',')) data.DiffTotalWithout = std::stod(token);
                // Read DiffPerStepWithout
                if (std::getline(ss, token, ',')) data.DiffPerStepWithout = std::stod(token);
                // Read DiffPercentWithout
                if (std::getline(ss, token, ',')) data.DiffPercentWithout = std::stod(token);
                // Read DiffTotalWith
                if (std::getline(ss, token, ',')) data.DiffTotalWith = std::stod(token);
                // Read DiffPerStepWith
                if (std::getline(ss, token, ',')) data.DiffPerStepWith = std::stod(token);
                // Read DiffPercentWith
                if (std::getline(ss, token, ',')) data.DiffPercentWith = std::stod(token);

                csvData.push_back(data);
            }
            csvFileIn.close();
        } else {
            std::cerr << "Unable to open file!" << std::endl;
        }

        CSVData newData;
        newData.NSTEP = num;
        newData.SKIPBY = SKIPBY;
        newData.noneGraphTotalTimeWithout = nonGraphTotalTimeWithout;
        newData.GraphTotalTimeWithout = graphTotalTimeWithout;
        newData.noneGraphTotalTimeWith = nonGraphTotalTime;
        newData.GraphTotalTimeWith = graphTotalTime;
        newData.DiffTotalWithout = difference2;
        newData.DiffPerStepWithout = diffPerKernel2;
        newData.DiffPercentWithout = diffPercentage2;
        newData.DiffTotalWith = difference;
        newData.DiffPerStepWith = diffPerKernel;
        newData.DiffPercentWith = diffPercentage;

        // Function to update or append data
        auto updateOrAppend = [](std::vector<CSVData>& dataVec, const CSVData& newData) {
            bool updated = false;
            for (auto& entry : dataVec) {
                if (entry.NSTEP == newData.NSTEP) {
                    entry = newData;
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                dataVec.push_back(newData);
            }
        };

        updateOrAppend(csvData, newData);

        // Write updated data back to CSV
        std::string tempFileName = "timing_data.tmp";
        std::ofstream tempFile(tempFileName);
        if (!tempFile.is_open()) {
            std::cerr << "Failed to open the temporary file for writing!" << std::endl;
            return -1;
        }

        tempFile << "NSTEP,SKIPBY,noneGraphTotalTimeWithout,GraphTotalTimeWithout,noneGraphTotalTimeWith,GraphTotalTimeWith,DiffTotalWithout,DiffPerStepWithout,DiffPercentWithout,DiffTotalWith,DiffPerStepWith,DiffPercentWith\n";

        for (const auto& entry : csvData) {
            tempFile << entry.NSTEP << "," 
                    << entry.SKIPBY << ','
                    << entry.noneGraphTotalTimeWithout << "," 
                    << entry.GraphTotalTimeWithout << "," 
                    << entry.noneGraphTotalTimeWith << "," 
                    << entry.GraphTotalTimeWith  << "," 
                    << entry.DiffTotalWithout << "," 
                    << entry.DiffPerStepWithout << "," 
                    << entry.DiffPercentWithout << "," 
                    << entry.DiffTotalWith << "," 
                    << entry.DiffPerStepWith << "," 
                    << entry.DiffPercentWith << "\n";
        }
        
        // Close CSV file
        tempFile.close();

        // Replace the original file with the temporary file
        std::remove("timing_data.csv");
        std::rename(tempFileName.c_str(), "timing_data.csv");

        printf("SUCCESS: ADDED TO CSV FILE\n");

    }

    return 0;
}
