#include "csv_util.h"

#include <fstream>
#include <sstream>
#include <iostream>


//--------------------------------------------------------------------------------------
// Implementation of the CSV-writing function that overwrites the file from scratch
//--------------------------------------------------------------------------------------
void rewriteCSV(const std::string &filename, 
                const std::vector<CSVData> &allData, 
                const int runs) 
{
    const int NUM_RUNS = runs;

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open '" << filename << "' for writing!\n";
        return;
    }

    // 2) Write the header row once
    outFile << "NSTEP,SKIPBY,";
    auto writeCols = [&](const std::string &baseName) {
        for (int i = 1; i <= NUM_RUNS; i++) {
            outFile << baseName << i << ",";
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

    // Remove trailing comma and add newline
    // Move the file pointer back by 1 character, overwrite ',' with '\n'
    outFile.seekp(-1, std::ios_base::cur);
    outFile << "\n";

    // 3) Write all rows from allData
    auto writeVals = [&](const std::vector<float> &arr) {
        for (float val : arr) {
            outFile << val << ",";
        }
    };

    for (const auto &entry : allData) {
        // First two columns
        outFile << entry.NSTEP << "," << entry.SKIPBY << ",";

        // Then the arrays
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

        // Remove trailing comma and add newline
        outFile.seekp(-1, std::ios_base::cur);
        outFile << "\n";
    }

    outFile.close();
    std::cout << "SUCCESS: Rewrote CSV file '" << filename << "'\n";
}

//----------------------------------------------------------------------
// Previous Implementation
//----------------------------------------------------------------------

bool readFloatToken(std::istringstream &ss, float &val) {
    std::string token;
    if (!std::getline(ss, token, ',')) return false;
    val = std::stof(token);
    return true;
}

void updateOrAppendCSV(const std::string &filename, const CSVData &newData, const int runs) {
    const int NUM_RUNS = runs;
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
            data.noneGraphTotalTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWithout[i])) break;
            }
            data.GraphTotalTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWithout[i])) break;
            }
            data.noneGraphTotalTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.noneGraphTotalTimeWith[i])) break;
            }
            data.GraphTotalTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.GraphTotalTimeWith[i])) break;
            }
            data.DiffTotalWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffTotalWithout[i])) break;
            }
            data.DiffPerStepWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWithout[i])) break;
            }
            data.DiffPercentWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffPercentWithout[i])) break;
            }
            data.DiffTotalWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffTotalWith[i])) break;
            }
            data.DiffPerStepWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffPerStepWith[i])) break;
            }
            data.DiffPercentWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.DiffPercentWith[i])) break;
            }
            data.ChronoNoneGraphTotalTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWithout[i])) break;
            }
            data.ChronoGraphTotalTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWithout[i])) break;
            }
            data.ChronoNoneGraphTotalLaunchTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWithout[i])) break;
            }
            data.ChronoGraphTotalLaunchTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWithout[i])) break;
            }
            data.ChronoNoneGraphTotalTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalTimeWith[i])) break;
            }
            data.ChronoGraphTotalTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalTimeWith[i])) break;
            }
            data.ChronoNoneGraphTotalLaunchTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoNoneGraphTotalLaunchTimeWith[i])) break;
            }
            data.ChronoGraphTotalLaunchTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoGraphTotalLaunchTimeWith[i])) break;
            }
            data.ChronoDiffTotalTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWithout[i])) break;
            }
            data.ChronoDiffPerStepWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWithout[i])) break;
            }
            data.ChronoDiffPercentWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWithout[i])) break;
            }
            data.ChronoDiffTotalTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffTotalTimeWith[i])) break;
            }
            data.ChronoDiffPerStepWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPerStepWith[i])) break;
            }
            data.ChronoDiffPercentWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffPercentWith[i])) break;
            }
            data.ChronoDiffLaunchTimeWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWithout[i])) break;
            }
            data.ChronoDiffLaunchPercentWithout.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWithout[i])) break;
            }
            data.ChronoDiffLaunchTimeWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchTimeWith[i])) break;
            }
            data.ChronoDiffLaunchPercentWith.resize(NUM_RUNS);
            for (int i = 0; i < NUM_RUNS; i++) {
                if(!readFloatToken(ss, data.ChronoDiffLaunchPercentWith[i])) break;
            }

            csvData.push_back(data); //each line of record 
        }
        csvFileIn.close();
    }
    std::cout << "Passed fetching stage" << '\n';
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
                for (int i = 1; i <= NUM_RUNS; i++) {
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
            // auto writeVals = [&](const float arr[4]) {
            //     for (int i = 0; i < 4; i++) {
            //         tempFile << arr[i] << ",";
            //     }
            // };
            auto writeVals = [&](const std::vector<float>& arr) {
                if (arr.size() < NUM_RUNS) {
                    std::cerr << "csvData(entry) size is less than NUM_RUNS!" << '\n';
                } else if (arr.size() > NUM_RUNS) {
                    std::cerr << "csvData(entry) size is greater than NUM_RUNS!" << '\n';
                }

                for (int i = 0; i < arr.size(); i++) {
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