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
        bool doGridSearch = true; // nuevo: habilita Grid Search durante train
    };
}

int RunTrain(const TrainSVM::Options& opts);
int RunTrainRefiner(const TrainSVM::Options& opts, bool doLOO = true);
int RunEval(int argc, char** argv);
int RunEvalRefinerOnly(const std::string& segFolder,
    const std::string& outTxt,
    const std::string& model912,
    const std::string& model912scaler = ""); 
int RunExtractTest(int argc, char** argv);
