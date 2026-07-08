#pragma once

#include <string>
#include <string_view>

#include <userver/components/component_list.hpp>
#include <userver/server/handlers/http_handler_base.hpp>

#include "../batch/batch_collector.hpp"
#include "../inference/inference.hpp"

class PredictHandler final : public userver::server::handlers::HttpHandlerBase {
public:
    static constexpr std::string_view kName = "handler-predict";

    PredictHandler(const userver::components::ComponentConfig& config,
                   const userver::components::ComponentContext& context);

    static void AppendPredict(
        userver::components::ComponentList& component_list,
        Inference& inference
    );

    std::string HandleRequestThrow(
        const userver::server::http::HttpRequest& request,
        userver::server::request::RequestContext&
    ) const override;

private:
    mutable BatchCollector batch_collector_;
};
