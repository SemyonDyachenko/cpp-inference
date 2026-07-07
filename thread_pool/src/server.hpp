#pragma once
#include "thread_pool.hpp"
#include "inference/inference.hpp"

class TcpServer {
public:
    TcpServer(int port, size_t num_threads, const std::string& model_path);
    ~TcpServer();
    void run();
private:
    int server_fd;
    ThreadPool pool;
    Inference inference;

    void handleClient(int client_fd);   
    std::vector<float> parseInput(const std::string& request);
};