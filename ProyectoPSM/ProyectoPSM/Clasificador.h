#pragma once

#include <string>

namespace TrainSVM {
    struct Options {
        std::string inputFolder;
        std::string outModelPath;
        std::string csvOut;
        double C = 1.0;
        double gamma = 0.0;
        bool doScale = false;
    };
}

int RunTrain(const TrainSVM::Options& opts);
int RunTrainRefiner(const TrainSVM::Options& opts, bool doLOO = true);
int RunEval(int argc, char** argv);
int RunExtractTest(int argc, char** argv);
void printEvalUsage(); // opcional