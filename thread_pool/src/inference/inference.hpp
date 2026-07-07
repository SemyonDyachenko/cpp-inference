#pragma once
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

class Inference {
public:
    Inference(const std::string& model_path);
    std::vector<float> predict(const std::vector<float>& input);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
};