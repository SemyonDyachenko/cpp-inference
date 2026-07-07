#include "inference.hpp"
#include <stdexcept>

Inference::Inference(const std::string& model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "inference")
    , session(env, model_path.c_str(), Ort::SessionOptions{}) {
}

std::vector<float> Inference::predict(const std::vector<float>& input) {
    std::vector<int64_t> input_shape = {1, (int64_t)input.size()};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output_label"};  
    
    auto output = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // int64
    int64_t* result = output[0].GetTensorMutableData<int64_t>();
    int64_t count = output[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    // конвертируем в float для совместимости
    return std::vector<float>(result, result + count);
}