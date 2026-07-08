#include "inference.hpp"

#include <algorithm>
#include <cmath>

namespace {

Inference::Result ResultFromLogits(const float* logits, size_t num_classes) {
    std::vector<float> probabilities(logits, logits + num_classes);

    float max_logit = *std::max_element(probabilities.begin(), probabilities.end());
    float sum = 0.0f;
    for (auto& p : probabilities) {
        p = std::exp(p - max_logit);
        sum += p;
    }
    for (auto& p : probabilities) {
        p /= sum;
    }

    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    int digit = static_cast<int>(std::distance(probabilities.begin(), max_it));

    return Inference::Result{digit, *max_it, std::move(probabilities)};
}

} 

Inference::Inference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "inference_server"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr}) {
}

std::vector<Inference::Result> Inference::predictBatch(const std::vector<std::vector<float>>& images) {
    const int64_t batch_size = static_cast<int64_t>(images.size());

    std::vector<float> input_data;
    input_data.reserve(images.size() * 28 * 28);
    for (const auto& image : images) {
        input_data.insert(input_data.end(), image.begin(), image.end());
    }

    std::vector<int64_t> input_shape{batch_size, 1, 28, 28};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    const float* output_data = output_tensors.front().GetTensorData<float>();
    const size_t num_classes = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape().back();

    std::vector<Result> results;
    results.reserve(images.size());
    for (int64_t i = 0; i < batch_size; ++i) {
        results.push_back(ResultFromLogits(output_data + i * num_classes, num_classes));
    }

    return results;
}

Inference::Result Inference::predict(const std::vector<float>& image) {
    return predictBatch({image}).front();
}
