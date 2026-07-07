#include "inference/inference.hpp"
#include "server.hpp"
#include <iostream>


int main() {
    TcpServer server(8080, 4, "model.onnx");
    server.run();

    return 0;
}