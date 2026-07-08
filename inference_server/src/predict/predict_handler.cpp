#include "predict_handler.hpp"

#include <userver/components/component_context.hpp>
#include <userver/formats/json.hpp>

namespace {

constexpr int kMaxBatchSize = 32;
constexpr int kBatchTimeoutMs = 5;

Inference* g_inference = nullptr;

}  

PredictHandler::PredictHandler(
    const userver::components::ComponentConfig& config,
    const userver::components::ComponentContext& context
)
    : HttpHandlerBase(config, context),
      batch_collector_(*g_inference, kMaxBatchSize, kBatchTimeoutMs) {
}

void PredictHandler::AppendPredict(
    userver::components::ComponentList& component_list,
    Inference& inference
) {
    g_inference = &inference;
    component_list.Append<PredictHandler>();
}

std::string PredictHandler::HandleRequestThrow(
    const userver::server::http::HttpRequest& request,
    userver::server::request::RequestContext&
) const {
    auto body = userver::formats::json::FromString(request.RequestBody());
    auto pixels_json = body["pixels"];

    std::vector<float> pixels;
    pixels.reserve(pixels_json.GetSize());
    for (const auto& value : pixels_json) {
        pixels.push_back(value.As<float>());
    }

    auto result = batch_collector_.add(std::move(pixels)).get();

    userver::formats::json::ValueBuilder response;
    response["digit"] = result.digit;
    response["confidence"] = result.confidence;
    response["probabilities"] = result.probabilities;

    return userver::formats::json::ToString(response.ExtractValue());
}
