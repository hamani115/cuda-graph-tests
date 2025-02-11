#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>   // For std::chrono
#include <string>

// CUDA runtime
#include <cuda_runtime.h>
#include "../cuda_check.h"

// Here you can set the device ID
#define MYDEVICE 0

// Problem size
// #define N (1U << 20) 
// #define NSTEP 10000
// #define SKIPBY 0
#define DEFAULT_NSTEP 10000
#define DEFAULT_SKIPBY 0

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
// Simple struct to store final timing results for CSV
// ---------------------------------------------------------------------
struct CSVData {
    int NSTEP;
    int SKIPBY;
    float noneGraphTotalTimeWithout[4];
    float GraphTotalTimeWithout[4];
    float noneGraphTotalTimeWith[4];
    float GraphTotalTimeWith[4];
    float DiffTotalWithout[4];
    float DiffPerStepWithout[4];
    float DiffPercentWithout[4];
    float DiffTotalWith[4];
    float DiffPerStepWith[4];
    float DiffPercentWith[4];
    float ChronoNoneGraphTotalTimeWithout[4];
    float ChronoGraphTotalTimeWithout[4];
    float ChronoNoneGraphTotalLaunchTimeWithout[4];
    float ChronoGraphTotalLaunchTimeWithout[4];
    float ChronoNoneGraphTotalTimeWith[4];
    float ChronoGraphTotalTimeWith[4];
    float ChronoNoneGraphTotalLaunchTimeWith[4];
    float ChronoGraphTotalLaunchTimeWith[4];
    float ChronoDiffTotalTimeWithout[4];
    float ChronoDiffPerStepWithout[4];
    float ChronoDiffPercentWithout[4];
    float ChronoDiffTotalTimeWith[4];
    float ChronoDiffPerStepWith[4];
    float ChronoDiffPercentWith[4];
    float ChronoDiffLaunchTimeWithout[4];
    float ChronoDiffLaunchPercentWithout[4];
    float ChronoDiffLaunchTimeWith[4];
    float ChronoDiffLaunchPercentWith[4];
};

// Helper function to read a float with error checking:
bool readFloatToken(std::istringstream &ss, float &val) {
    std::string token;
    if (!std::getline(ss, token, ',')) return false;
    val = std::stof(token);
    return true;
}

// ---------------------------------------------------------------------
// Example CSV update function (minimal version)
// ---------------------------------------------------------------------
void updateOrAppendCSV(const std::string &filename, const CSVData &newData) {
    std::vector<CSVData> csvData;
    std::ifstream csvFileIn(filename);
    if (csvFileIn.is_open()) {
        std::string line;

        // Check if file is empty or not
        if (std::getline(csvFileIn, line));
        while (std::getline(csvFileIn, line)) {
            std::istringstream ss(line);
            CSVData data;
            std::string token;
            if (!std::getline(ss, token, ',')) continue;
            data.NSTEP = std::stoi(token);
            if (!std::getline(ss, token, ',')) continue;
            data.SKIPBY = std::stoi(token);

            // Read all arrays of 4 values:
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffTotalWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffTotalWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.DiffPercentWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWithout[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWith[i])) break;
            }
            for (int i = 0; i < 4; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWith[i])) break;
            }

            csvData.push_back(data);
        }
        csvFileIn.close();
    }

    // Update or append
    bool updated = false;
    for (auto &entry : csvData) {
        if (entry.NSTEP == newData.NSTEP && entry.SKIPBY == newData.SKIPBY) {
            entry = newData;
            updated = true;
            break;
        }
    }

    if (!updated) {
        csvData.push_back(newData);
    }

    std::string tempFILENAME = "complex_3_different_kernels.tmp";
    {
        std::ofstream tempFile(tempFILENAME);
        if (!tempFile.is_open()) {
            std::cerr << "Failed to open the temporary file for writing!" << std::endl;
            return;
        }

        if (!false) {
            tempFile << "NSTEP,SKIPBY,";

            // For each metric, add the four columns with suffixes 1..4
            auto writeCols = [&](const std::string &baseName) {
                for (int i = 1; i <= 4; i++) {
                    tempFile << baseName << i << ",";
                }
            };

            writeCols("noneGraphTotalTimeWithout");
            writeCols("GraphTotalTimeWithout");
            writeCols("noneGraphTotalTimeWith");
            writeCols("GraphTotalTimeWith");
            writeCols("DiffTotalWithout");
            writeCols("DiffPerStepWithout");
            writeCols("DiffPercentWithout");
            writeCols("DiffTotalWith");
            writeCols("DiffPerStepWith");
            writeCols("DiffPercentWith");
            writeCols("ChronoNoneGraphTotalTimeWithout");
            writeCols("ChronoGraphTotalTimeWithout");
            writeCols("ChronoNoneGraphTotalLaunchTimeWithout");
            writeCols("ChronoGraphTotalLaunchTimeWithout");
            writeCols("ChronoNoneGraphTotalTimeWith");
            writeCols("ChronoGraphTotalTimeWith");
            writeCols("ChronoNoneGraphTotalLaunchTimeWith");
            writeCols("ChronoGraphTotalLaunchTimeWith");
            writeCols("ChronoDiffTotalTimeWithout");
            writeCols("ChronoDiffPerStepWithout");
            writeCols("ChronoDiffPercentWithout");
            writeCols("ChronoDiffTotalTimeWith");
            writeCols("ChronoDiffPerStepWith");
            writeCols("ChronoDiffPercentWith");
            writeCols("ChronoDiffLaunchTimeWithout");
            writeCols("ChronoDiffLaunchPercentWithout");
            writeCols("ChronoDiffLaunchTimeWith");
            writeCols("ChronoDiffLaunchPercentWith");

            // Remove last comma and add newline
            tempFile.seekp(-1, std::ios_base::cur);
            tempFile << "\n";
        }

        for (const auto &entry : csvData) {
            tempFile << entry.NSTEP << "," << entry.SKIPBY << ",";
            auto writeVals = [&](const float arr[4]) {
                for (int i = 0; i < 4; i++) {
                    tempFile << arr[i] << ",";
                }
            };

            writeVals(entry.noneGraphTotalTimeWithout);
            writeVals(entry.GraphTotalTimeWithout);
            writeVals(entry.noneGraphTotalTimeWith);
            writeVals(entry.GraphTotalTimeWith);
            writeVals(entry.DiffTotalWithout);
            writeVals(entry.DiffPerStepWithout);
            writeVals(entry.DiffPercentWithout);
            writeVals(entry.DiffTotalWith);
            writeVals(entry.DiffPerStepWith);
            writeVals(entry.DiffPercentWith);
            writeVals(entry.ChronoNoneGraphTotalTimeWithout);
            writeVals(entry.ChronoGraphTotalTimeWithout);
            writeVals(entry.ChronoNoneGraphTotalLaunchTimeWithout);
            writeVals(entry.ChronoGraphTotalLaunchTimeWithout);
            writeVals(entry.ChronoNoneGraphTotalTimeWith);
            writeVals(entry.ChronoGraphTotalTimeWith);
            writeVals(entry.ChronoNoneGraphTotalLaunchTimeWith);
            writeVals(entry.ChronoGraphTotalLaunchTimeWith);
            writeVals(entry.ChronoDiffTotalTimeWithout);
            writeVals(entry.ChronoDiffPerStepWithout);
            writeVals(entry.ChronoDiffPercentWithout);
            writeVals(entry.ChronoDiffTotalTimeWith);
            writeVals(entry.ChronoDiffPerStepWith);
            writeVals(entry.ChronoDiffPercentWith);
            writeVals(entry.ChronoDiffLaunchTimeWithout);
            writeVals(entry.ChronoDiffLaunchPercentWithout);
            writeVals(entry.ChronoDiffLaunchTimeWith);
            writeVals(entry.ChronoDiffLaunchPercentWith);

            // Remove last comma and add newline
            tempFile.seekp(-1, std::ios_base::cur);
            tempFile << "\n";
        }
    }

    std::remove(filename.c_str());
    std::rename(tempFILENAME.c_str(), filename.c_str());
    std::cout << "SUCCESS: ADDED/UPDATED CSV FILE\n";
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

// ---------------------------------------------------------------------
// Example function: runWithoutGraph
// Add CHRONO timing in addition to existing event-based timing
// ---------------------------------------------------------------------
void runWithoutGraph(int nBuffers, std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr,                         // (out) total GPU time with + w/o first
                     std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr,             // (out) total CPU time with + w/o first
                     std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr, // (out) total Launch time with + w/o first
                     int nstep, int skipby) {
    const int NSTEP = nstep;
    const int SKIPBY = skipby;
    constexpr size_t arraySize = 1U << 20;
    
    // Choose one CUDA device
    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    size_t memSize = arraySize * sizeof(float);

    // Allocate pinned host memory
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    // Initialize h_a using index 'i'
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = static_cast<float>(i);
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

    // Also allocate the final result buffer
    float* d_result = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream));

    // 3. Create pointer-to-pointer in device memory
    float** d_buffers = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buffers, nBuffers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpyAsync(d_buffers, devBuffers.data(),
                               nBuffers * sizeof(float*),
                               cudaMemcpyHostToDevice, stream));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timings
    cudaEvent_t start, stop;    
    float firstTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // ---- Chrono timing variables ----
    using clock_type = std::chrono::steady_clock;
    // auto chronoBegin = clock_type::now();
    
    // 4. Copy data from host to device for the *first* buffer
    CUDA_CHECK(cudaMemcpyAsync(devBuffers[0], h_a, memSize, cudaMemcpyHostToDevice, stream));

    // Start Timer for first run (CUDA event)
    CUDA_CHECK(cudaEventRecord(start, stream));
    auto firstIterStart = clock_type::now();

    // D->D copies
    for(int b = 1; b < nBuffers; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(devBuffers[b],
                                   devBuffers[b - 1],
                                   memSize,
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    }

    // Single kernel launch after memcpys
    add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_buffers, d_result, nBuffers, arraySize);
    CUDA_CHECK(cudaGetLastError());

    // Device to host memory copy
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // Synchronize
    // CUDA_CHECK(cudaStreamSynchronize(stream));

    // Stop Timer for first run (CUDA event)
    auto firstIterStop = clock_type::now();
    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaEventSynchronize(stop));

    auto firstIterStop2 = clock_type::now();
    CUDA_CHECK(cudaEventElapsedTime(&firstTime, start, stop));

    const std::chrono::duration<double> firstTimeChronoMS = firstIterStop - firstIterStart;
    const std::chrono::duration<double> firstTimeChronoMS2 = firstIterStop2 - firstIterStart;

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));
    
    // Variables for timing statistics
    float elapsedTime = 0.0f;
    float totalTime = 0.0f;      // accum (event) of all subsequent runs
    std::chrono::duration<double> totalTimeChrono = std::chrono::duration<double>(0.0);
    std::chrono::duration<double> totalLunchTimeChrono = std::chrono::duration<double>(0.0);
    // Range variables
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    // Std dev variables
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    std::vector<int> nsteps = generateSequence(NSTEP);

    // We'll also track chrono time for subsequent runs:
    // float totalTimeChronoSubRuns = 0.0f;

    for (int istep = 1; istep <= NSTEP; istep++) {
        // Start Timer (CUDA event)
        CUDA_CHECK(cudaEventRecord(execStart, stream));

        // D->D copies
        for(int b = 1; b < nBuffers; ++b) {
            CUDA_CHECK(cudaMemcpyAsync(devBuffers[b],
                                       devBuffers[b - 1],
                                       memSize,
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        }

        add_arrays<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_buffers, d_result, nBuffers, arraySize
        );
        // CUDA_CHECK(cudaGetLastError());

        // Copy back to host
        CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

        // CUDA_CHECK(cudaStreamSynchronize(stream));

        // Stop Timer (CUDA event)
        auto chronoStopStep = clock_type::now();
        CUDA_CHECK(cudaEventRecord(execStop, stream));
        
        CUDA_CHECK(cudaEventSynchronize(execStop));

        auto chronoStopStep2 = clock_type::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        // Chrono for this iteration
        // float thisStepChronoMS = std::chrono::duration<float, std::milli>(chronoStopStep - chronoStartStep).count();

        // Collect stats only after SKIPBY
        if (istep > SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = chronoStopStep - chronoStopStep;
            const std::chrono::duration<double> lunchExecTimeChrono = chronoStopStep2 - chronoStopStep;
            totalTime += elapsedTime;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;

            // Welford's online mean and variance
            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (const auto& num : nsteps) {
                if (num == istep) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + firstTime) / (NSTEP + 1 - SKIPBY);
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    // Print out the time statistics
                    std::cout << "=======Setup (No Graph, " << nBuffers << " buffers)  for NSTEP " << istep << "=======" << std::endl;
                    std::cout << "Iterations: " << istep << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << blocksPerGrid << std::endl; //change
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl; //change
                    std::cout << "Array Size: " << arraySize << std::endl; //change
                    std::cout << "=======Results-GPU=======" << std::endl;
                    std::cout << "First Run Time (Event): " << firstTime << " ms" << std::endl;
                    std::cout << "Average Time (Event) inc. first: " << (totalTime + firstTime) / (istep + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time (Event) exc. first: " << (totalTime / (istep - SKIPBY)) << " ms" << std::endl;
                    std::cout << "Variance (Event): " << varianceTime << " ms^2" << std::endl;
                    std::cout << "Std Dev (Event): " << stdDevTime << " ms" << std::endl;
                    std::cout << "Time Spread (Event): " << lowerTime << " - " << upperTime << " ms" << std::endl;
                    std::cout << "Total Time (Event) with first: " << totalTime + firstTime << " ms" << std::endl;
                    std::cout << "Total Time (Event) w/o first: " << totalTime << " ms" << std::endl;
                    // CHRONO
                    std::cout << "=======Results-CPU=======" << std::endl;
                    const std::chrono::duration<double> totalTimeWithChrono = totalTimeChrono + firstTimeChronoMS2;
                    const std::chrono::duration<double> totalLunchTimeWithChrono = totalLunchTimeChrono + firstTimeChronoMS;
                    std::cout << "Graph Creation Chrono Launch before Sync: " << firstTimeChronoMS.count() * 1000  << " ms" << std::endl;
                    std::cout << "Graph Creation Chrono LunchExec after Sync: " << firstTimeChronoMS2.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch without Graph Creation: " << totalLunchTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono Launch with Graph Creation: " << totalLunchTimeWithChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec without Graph Creation: " << totalTimeChrono.count() * 1000  << " ms" << std::endl;
                    std::cout << "Total Time Chrono LunchExec with Graph Creation: " << totalTimeWithChrono.count() * 1000 << " ms" << std::endl;
                    
                    // Return GPU-based time
                    totalTimeWithArr.push_back(totalTime + firstTime);
                    totalTimeWithoutArr.push_back(totalTime);
                    // Return CPU-based time
                    chronoTotalTimeWithArr.push_back(totalTimeWithChrono.count() * 1000.0);
                    chronoTotalTimeWithoutArr.push_back(totalTimeChrono.count() * 1000.0);
                    chronoTotalLaunchTimeWithArr.push_back(totalLunchTimeWithChrono.count() * 1000.0);
                    chronoTotalLaunchTimeWithoutArr.push_back(totalLunchTimeChrono.count() * 1000.0);
                }
            }
        }
    }

    // Compute final statistics
    // float meanTime = (totalTime + firstTime) / (NSTEP + 1 - SKIPBY);
    // double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    // double stdDevTime = std::sqrt(varianceTime);

    // // Also measure the entire CPU time from beginning to end
    // auto chronoEnd = clock_type::now();
    // float totalTimeChronoMS = std::chrono::duration<float, std::milli>(chronoEnd - chronoBegin).count();

    // // Print summary
    // std::cout << "======= Setup (No Graph, " << nBuffers << " buffers) =======" << std::endl;
    // std::cout << "Iterations: " << NSTEP << std::endl;
    // std::cout << "First Run Time (Event): " << firstTime << " ms" << std::endl;
    // std::cout << "Avg Time (Event) inc. first: " << meanTime << " ms" << std::endl;
    // std::cout << "Avg Time (Event) exc. first: "
    //           << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    // std::cout << "Std Dev (Event): " << stdDevTime << " ms" << std::endl;
    // std::cout << "Time Spread (Event): " << lowerTime << " - " << upperTime << " ms" << std::endl;
    // std::cout << "Total Time (Event) with first: " << firstTime + totalTime << " ms" << std::endl;
    // std::cout << "Total Time (Event) w/o first: " << totalTime << " ms" << std::endl;

    // std::cout << "---- Chrono Timings (CPU) ----" << std::endl;
    // std::cout << "First Iteration (CPU): " << firstTimeChronoMS << " ms" << std::endl;
    // std::cout << "Sum of Subsequent (CPU): " << totalTimeChronoSubRuns << " ms" << std::endl;
    // std::cout << "Total CPU Time (entire function): " << totalTimeChronoMS << " ms" << std::endl;

    // Verify correctness
    for (int i = 1; i <= arraySize; i++) {
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

    for(int b = 0; b < nBuffers; ++b) {
        CUDA_CHECK(cudaFree(devBuffers[b]));
    }
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_buffers));
}

void runWithGraph(int nBuffers, std::vector<float>& totalTimeWithArr, std::vector<float>& totalTimeWithoutArr, 
                std::vector<float>& chronoTotalTimeWithArr, std::vector<float>& chronoTotalTimeWithoutArr, 
                std::vector<float>& chronoTotalLaunchTimeWithArr, std::vector<float>& chronoTotalLaunchTimeWithoutArr,
                int nstep, int skipby) {
    
    const int NSTEP = nstep;
    const int SKIPBY = skipby;
    constexpr size_t arraySize = 1U << 20;

    CUDA_CHECK(cudaSetDevice(MYDEVICE));

    size_t memSize = arraySize * sizeof(float);

    // Allocate pinned host memory
    float* h_a;
    float* h_result;
    CUDA_CHECK(cudaMallocHost(&h_a, memSize));
    CUDA_CHECK(cudaMallocHost(&h_result, memSize));

    for (int i = 0; i < arraySize; ++i) {
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

    float* d_result = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_result, memSize, stream));

    // 3. pointer-to-pointer
    float** d_buffers = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buffers, nBuffers * sizeof(float*)));
    CUDA_CHECK(cudaMemcpyAsync(d_buffers, devBuffers.data(),
                               nBuffers * sizeof(float*),
                               cudaMemcpyHostToDevice, stream));

    // Kernel dims
    int threadsPerBlock = 1024;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Create events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy first buffer
    CUDA_CHECK(cudaMemcpyAsync(devBuffers[0], h_a, memSize, cudaMemcpyHostToDevice, stream));

    // ---- Chrono timing variables ----
    using clock_type = std::chrono::steady_clock;
    // auto chronoBegin = clock_type::now();

    // Start graph creation timer
    float graphCreateTime = 0.0f;
    CUDA_CHECK(cudaEventRecord(start, stream));
    // auto creationStart = clock_type::now();
    const auto graphStart = std::chrono::steady_clock::now();

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
        d_buffers, d_result, nBuffers, arraySize
    );

    // Copy result to host
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_result, memSize, cudaMemcpyDeviceToHost, stream));

    // End capture
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // First graph launch
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    // End Timer for graph creation
    const auto graphEnd = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaEventSynchronize(stop));

    const auto graphEnd2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventElapsedTime(&graphCreateTime, start, stop));

    const std::chrono::duration<double> graphCreateTimeChrono = graphEnd - graphStart;
    const std::chrono::duration<double> graphCreateTimeChrono2 = graphEnd2 - graphStart;

    // auto creationStop = clock_type::now();
    // float creationTimeChronoMS = std::chrono::duration<float, std::milli>(creationStop - creationStart).count();

    // Measure execution time
    cudaEvent_t execStart, execStop;
    CUDA_CHECK(cudaEventCreate(&execStart));
    CUDA_CHECK(cudaEventCreate(&execStop));

    // Timing stats for subsequent launches
    float elapsedTime = 0.0f;
    float totalTime = 0.0f;
    std::chrono::duration<double> totalTimeChrono = std::chrono::duration<double>(0.0);
    std::chrono::duration<double> totalLunchTimeChrono = std::chrono::duration<double>(0.0);
    float upperTime = 0.0f;
    float lowerTime = 0.0f;
    double mean = 0.0;
    double M2 = 0.0;
    int count = 0;

    // We'll also track CPU time for subsequent launches:
    // float totalTimeChronoSub = 0.0f;
    std::vector<int> nsteps = generateSequence(NSTEP);

    // Main loop
    for (int istep = 1; istep <= NSTEP; istep++) {
        // Start event
        CUDA_CHECK(cudaEventRecord(execStart, stream));
        // auto stepStart = clock_type::now();
        const auto start = std::chrono::steady_clock::now();

        // Launch the graph
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));

        // Stop event
        const auto end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(execStop, stream));
        CUDA_CHECK(cudaEventSynchronize(execStop));
        const auto end2 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, execStart, execStop));

        // // Chrono
        // auto stepStop = clock_type::now();
        // float stepChronoMS = std::chrono::duration<float, std::milli>(stepStop - stepStart).count();

        if (istep > SKIPBY) {
            const std::chrono::duration<double> launchTimeChrono = end - start;
            const std::chrono::duration<double> lunchExecTimeChrono = end2 - start;
            totalTime += elapsedTime;
            totalTimeChrono += lunchExecTimeChrono;
            totalLunchTimeChrono += launchTimeChrono;
            // totalTimeChronoSub += stepChronoMS;

            count++;
            double delta = elapsedTime - mean;
            mean += delta / count;
            double delta2 = elapsedTime - mean;
            M2 += delta * delta2;

            if (elapsedTime > upperTime) upperTime = elapsedTime;
            if (elapsedTime < lowerTime || lowerTime == 0.0f) lowerTime = elapsedTime;

            for (const auto& num : nsteps) {
                if (num == istep) {
                    // Calculate mean and standard deviation
                    // float meanTime = (totalTime + graphCreateTime) / i;
                    double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
                    double stdDevTime = sqrt(varianceTime);

                    // Print out the time statistics
                    std::cout << "=======Setup (With Graph) for NSTEP " << istep << "=======" << std::endl;
                    std::cout << "Iterations: " << istep << std::endl;
                    std::cout << "Skip By: " << SKIPBY << std::endl;
                    std::cout << "Kernels: kernelA, kernelB, kernelC" << std::endl;
                    std::cout << "Number of Blocks: " << blocksPerGrid << std::endl;
                    std::cout << "Threads per Block: " << threadsPerBlock << std::endl;
                    std::cout << "Array Size: " << arraySize << std::endl;
                    std::cout << "=======Results (With Graph) for NSTEP " << istep << "=======" << std::endl;
                    std::cout << "Graph Creation Time: " << graphCreateTime << " ms" << std::endl;
                    std::cout << "Average Time with Graph: " << (totalTime + graphCreateTime) / (istep + 1 - SKIPBY) << " ms" << std::endl;
                    std::cout << "Average Time without Graph: " << (totalTime / (istep - SKIPBY)) << " ms" << std::endl;
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

    // float meanTime = (totalTime + graphCreateTime) / (NSTEP + 1 - SKIPBY);
    // double varianceTime = (count > 1) ? M2 / (count - 1) : 0.0;
    // double stdDevTime = std::sqrt(varianceTime);

    // // Total CPU time for entire function
    // auto chronoEnd = clock_type::now();
    // float totalTimeChronoMS = std::chrono::duration<float, std::milli>(chronoEnd - chronoBegin).count();

    // // Print
    // std::cout << "======= Setup (With Graph, " << nBuffers << " buffers) =======" << std::endl;
    // std::cout << "Iterations: " << NSTEP << std::endl;
    // std::cout << "Graph Creation Time (Event): " << graphCreateTime << " ms" << std::endl;
    // std::cout << "Average Time (with creation, Event): " << meanTime << " ms" << std::endl;
    // std::cout << "Average Time (without creation, Event): "
    //           << (totalTime / (NSTEP - SKIPBY)) << " ms" << std::endl;
    // std::cout << "Std Dev (Event): " << stdDevTime << " ms" << std::endl;
    // std::cout << "Time Spread (Event): " << lowerTime << " - " << upperTime << " ms" << std::endl;
    // std::cout << "Total Time (Event) w/o creation: " << totalTime << " ms" << std::endl;
    // std::cout << "Total Time (Event) w/ creation:  " << totalTime + graphCreateTime << " ms" << std::endl;

    // std::cout << "---- Chrono Timings (CPU) ----" << std::endl;
    // std::cout << "Graph Creation (CPU): " << creationTimeChronoMS << " ms" << std::endl;
    // std::cout << "Sum of Subsequent Launches (CPU): " << totalTimeChronoSub << " ms" << std::endl;
    // std::cout << "Total CPU Time (entire function): " << totalTimeChronoMS << " ms" << std::endl;

    // Verify
    for (int i = 1; i <= arraySize; i++) {
        float expected = i * (float)nBuffers;
        if (std::fabs(h_result[i] - expected) > 1e-7) {
            std::cerr << "Mismatch at " << i 
                      << ": got " << h_result[i]
                      << ", expected " << expected << std::endl;
            assert(false);
        }
    }

    // Cleanup
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

    // // Return total GPU times
    // *totalTimeWith    = totalTime + graphCreateTime;
    // *totalTimeWithout = totalTime;

    // // Return total CPU times
    // *chronoTotalWith    = totalTimeChronoMS;
    // *chronoTotalWithout = totalTimeChronoSub;
}

// ---------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    const int NSTEP = (argc > 1) ? atoi(argv[1]) : DEFAULT_NSTEP;
    const int SKIPBY = (argc > 2) ? atoi(argv[2]) : DEFAULT_SKIPBY;
    const int nBuffers = (argc > 3) ? atoi(argv[3]) : 10;
    // int nBuffers = 10;
    // if(argc > 3) {
    //     nBuffers = std::atoi(argv[3]);
    // }
    std::cout << "==============SIMPLE GRAPH: MULTI-DEVICE-MEMCPY TEST==============" << std::endl;
    std::cout << "Running with " << nBuffers << " buffers\n";

    std::vector<int> nsteps = generateSequence(NSTEP);
    const int NUM_RUNS = 4;

    // Create CSVData for each nstep and initialize it
    std::vector<CSVData> newDatas(nsteps.size());
    for (auto &newData : newDatas) {
        for (int r = 0; r < NUM_RUNS; r++) {
            // Initialize arrays to some default (e.g., 0) for safety
            newData.noneGraphTotalTimeWithout[r] = 0;
            newData.GraphTotalTimeWithout[r] = 0;
            newData.noneGraphTotalTimeWith[r] = 0;
            newData.GraphTotalTimeWith[r] = 0;
            newData.DiffTotalWithout[r] = 0;
            newData.DiffPerStepWithout[r] = 0;
            newData.DiffPercentWithout[r] = 0;
            newData.DiffTotalWith[r] = 0;
            newData.DiffPerStepWith[r] = 0;
            newData.DiffPercentWith[r] = 0;
            newData.ChronoNoneGraphTotalTimeWithout[r] = 0;
            newData.ChronoGraphTotalTimeWithout[r] = 0;
            newData.ChronoNoneGraphTotalLaunchTimeWithout[r] = 0;
            newData.ChronoGraphTotalLaunchTimeWithout[r] = 0;
            newData.ChronoNoneGraphTotalTimeWith[r] = 0;
            newData.ChronoGraphTotalTimeWith[r] = 0;
            newData.ChronoNoneGraphTotalLaunchTimeWith[r] = 0;
            newData.ChronoGraphTotalLaunchTimeWith[r] = 0;
            newData.ChronoDiffTotalTimeWithout[r] = 0;
            newData.ChronoDiffPerStepWithout[r] = 0;
            newData.ChronoDiffPercentWithout[r] = 0;
            newData.ChronoDiffTotalTimeWith[r] = 0;
            newData.ChronoDiffPerStepWith[r] = 0;
            newData.ChronoDiffPercentWith[r] = 0;
            newData.ChronoDiffLaunchTimeWithout[r] = 0;
            newData.ChronoDiffLaunchPercentWithout[r] = 0;
            newData.ChronoDiffLaunchTimeWith[r] = 0;
            newData.ChronoDiffLaunchPercentWith[r] = 0;
        }
    }

    for (int r = 0; r < NUM_RUNS; r++) {
        std::cout << "==============FOR RUN"<< r+1 << "==============" << std::endl;

        std::vector<float> noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr;
        std::vector<float> chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr;
        // Measure time for non-graph implementation
        runWithoutGraph(nBuffers, noneGraphTotalTimeWithArr, noneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalTimeWithArr, chronoNoneGraphTotalTimeWithoutArr, 
                        chronoNoneGraphTotalLaunchTimeWithArr, chronoNoneGraphTotalLaunchTimeWithoutArr,
                        NSTEP, SKIPBY);

        std::vector<float> graphTotalTimeWithArr, graphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalTimeWithArr, chronoGraphTotalTimeWithoutArr;
        std::vector<float> chronoGraphTotalLaunchTimeWithArr, chronoGraphTotalLaunchTimeWithoutArr;
        // Measure time for graph implementation
        runWithGraph(nBuffers, graphTotalTimeWithoutArr, graphTotalTimeWithArr, 
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

    const std::string FILENAME = "complex_3_different_kernels_new_T4.csv";
    for (const auto &newData : newDatas) {
        updateOrAppendCSV(FILENAME, newData);
    }

    return 0;

    // // We will store final times in these variables
    // float nonGraphTotalTimeWith = 0.f;
    // float nonGraphTotalTimeWithout = 0.f;
    // float chronoNoGraphWith = 0.f;
    // float chronoNoGraphWithout = 0.f;

    // // Measure time for non-graph
    // runWithoutGraph(nBuffers,
    //                 &nonGraphTotalTimeWith,
    //                 &nonGraphTotalTimeWithout,
    //                 &chronoNoGraphWith,
    //                 &chronoNoGraphWithout);

    // float graphTotalTimeWith = 0.f;
    // float graphTotalTimeWithout = 0.f;
    // float chronoGraphWith = 0.f;
    // float chronoGraphWithout = 0.f;

    // // Measure time for graph
    // runWithGraph(nBuffers,
    //              &graphTotalTimeWith,
    //              &graphTotalTimeWithout,
    //              &chronoGraphWith,
    //              &chronoGraphWithout);

    // // Simple comparison
    // float difference   = nonGraphTotalTimeWith - graphTotalTimeWith;
    // float diffPerRun   = difference / NSTEP;
    // float diffPercent  = (difference / nonGraphTotalTimeWith) * 100.f;

    // float difference2  = nonGraphTotalTimeWithout - graphTotalTimeWithout;
    // float diffPerRun2  = difference2 / NSTEP;
    // float diffPercent2 = (difference2 / nonGraphTotalTimeWithout) * 100.f;

    // std::cout << "\n=======Comparison (Excluding Graph Creation)=======\n";
    // std::cout << "Difference: " << difference2 << " ms\n";
    // std::cout << "Difference per step: " << diffPerRun2 << " ms\n";
    // std::cout << "Difference percentage: " << diffPercent2 << "%\n";

    // std::cout << "=======Comparison (Including Graph Creation)=======\n";
    // std::cout << "Difference: " << difference << " ms\n";
    // std::cout << "Difference per step: " << diffPerRun << " ms\n";
    // std::cout << "Difference percentage: " << diffPercent << "%\n";

    // // Also do a CPU-time comparison
    // float chronoDiffWith = chronoNoGraphWith - chronoGraphWith;
    // float chronoDiffWithout = chronoNoGraphWithout - chronoGraphWithout;

    // std::cout << "\n=======CPU Chrono Comparison=======\n";
    // std::cout << "CPU total (No Graph)  with  first: " << chronoNoGraphWith    << " ms\n";
    // std::cout << "CPU total (Graph)     with  creation: " << chronoGraphWith << " ms\n";
    // std::cout << "CPU difference (with): " << chronoDiffWith << " ms\n\n";

    // std::cout << "CPU total (No Graph)  w/o first: " << chronoNoGraphWithout    << " ms\n";
    // std::cout << "CPU total (Graph)     w/o creation: " << chronoGraphWithout << " ms\n";
    // std::cout << "CPU difference (without): " << chronoDiffWithout << " ms\n\n";

    // // ------------------------------------------------------
    // // Write out to CSV (if desired)
    // // ------------------------------------------------------
    // CSVData csvEntry;
    // csvEntry.nBuffers               = nBuffers;
    // csvEntry.NSTEP                  = NSTEP;
    // csvEntry.eventTimeNoGraphWith   = nonGraphTotalTimeWith;
    // csvEntry.eventTimeNoGraphWithout= nonGraphTotalTimeWithout;
    // csvEntry.eventTimeGraphWith     = graphTotalTimeWith;
    // csvEntry.eventTimeGraphWithout  = graphTotalTimeWithout;
    // csvEntry.chronoTimeNoGraphWith  = chronoNoGraphWith;
    // csvEntry.chronoTimeNoGraphWithout = chronoNoGraphWithout;
    // csvEntry.chronoTimeGraphWith    = chronoGraphWith;
    // csvEntry.chronoTimeGraphWithout = chronoGraphWithout;

    // updateOrAppendCSV("my_sum_arrays_results.csv", csvEntry);

    // return 0;
}
