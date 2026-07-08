FROM --platform=linux/amd64 ghcr.io/userver-framework/ubuntu-24.04-userver:latest AS build

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/include/onnxruntime && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz && \
    tar -xzf onnxruntime-linux-x64-1.19.2.tgz && \
    cp onnxruntime-linux-x64-1.19.2/include/* /usr/local/include/onnxruntime/ && \
    cp onnxruntime-linux-x64-1.19.2/lib/* /usr/local/lib/ && \
    rm -rf onnxruntime-linux-x64-1.19.2* && \
    ldconfig

WORKDIR /app
COPY . .

RUN cmake --preset release && \
    cmake --build build-release -j$(nproc) --target inference_server

FROM --platform=linux/amd64 ghcr.io/userver-framework/ubuntu-24.04-userver:latest AS runtime

COPY --from=build /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=build /app/build-release/inference_server /app/inference_server
COPY --from=build /app/configs /app/configs
COPY --from=build /app/models /app/models

EXPOSE 8080

ENTRYPOINT ["/app/inference_server", "--config", "/app/configs/static_config.yaml"]
