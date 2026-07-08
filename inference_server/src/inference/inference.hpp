#pragma once

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class Inference {
public:
    struct Result {
        int digit;
        float confidence;
        std::vector<float> probabilities;
    };

    explicit Inference(const std::string& model_path);

    Result predict(const std::vector<float>& image);
    std::vector<Result> predictBatch(const std::vector<std::vector<float>>& images);

private:
    Ort::Env env_;
    Ort::Session session_;
};
