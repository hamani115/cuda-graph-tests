#ifndef CSV_UTIL_H
#define CSV_UTIL_H

#include <vector>
#include <string>

struct CSVData {
    int NSTEP;
    int SKIPBY;
    std::vector<float> noneGraphTotalTimeWithout;
    std::vector<float> GraphTotalTimeWithout;
    std::vector<float> noneGraphTotalTimeWith;
    std::vector<float> GraphTotalTimeWith;
    std::vector<float> DiffTotalWithout;
    std::vector<float> DiffPerStepWithout;
    std::vector<float> DiffPercentWithout;
    std::vector<float> DiffTotalWith;
    std::vector<float> DiffPerStepWith;
    std::vector<float> DiffPercentWith;
    std::vector<float> ChronoNoneGraphTotalTimeWithout;
    std::vector<float> ChronoGraphTotalTimeWithout;
    std::vector<float> ChronoNoneGraphTotalLaunchTimeWithout;
    std::vector<float> ChronoGraphTotalLaunchTimeWithout;
    std::vector<float> ChronoNoneGraphTotalTimeWith;
    std::vector<float> ChronoGraphTotalTimeWith;
    std::vector<float> ChronoNoneGraphTotalLaunchTimeWith;
    std::vector<float> ChronoGraphTotalLaunchTimeWith;
    std::vector<float> ChronoDiffTotalTimeWithout;
    std::vector<float> ChronoDiffPerStepWithout;
    std::vector<float> ChronoDiffPercentWithout;
    std::vector<float> ChronoDiffTotalTimeWith;
    std::vector<float> ChronoDiffPerStepWith;
    std::vector<float> ChronoDiffPercentWith;
    std::vector<float> ChronoDiffLaunchTimeWithout;
    std::vector<float> ChronoDiffLaunchPercentWithout;
    std::vector<float> ChronoDiffLaunchTimeWith;
    std::vector<float> ChronoDiffLaunchPercentWith;
};

void rewriteCSV(const std::string &filename, 
                const std::vector<CSVData> &allData, 
                const int runs);

void updateOrAppendCSV(const std::string &filename, const CSVData &newData, const int runs);

#endif //CSV_UTIL_H