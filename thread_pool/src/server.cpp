#include "server.hpp"
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

TcpServer::TcpServer(int port, size_t num_threads, const std::string& path) : pool(num_threads), inference(path) {
    server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0) {
        throw std::runtime_error("socket() failed");
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if(bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        throw std::runtime_error("bind() failed");
    }

    if(listen(server_fd, 1024) < 0) {
        throw std::runtime_error("listen() failed");
    }
}

TcpServer::~TcpServer() {
    close(server_fd);
}

void TcpServer::run() {
    while(true) {
        int client_id = accept(server_fd, nullptr, nullptr);
        pool.enqueue([client_id, this] { handleClient(client_id); });
    }
}

void TcpServer::handleClient(int client_fd) {
    char buffer[4096] = {0};
    
    while (true) {
        memset(buffer, 0, sizeof(buffer));
        ssize_t bytes = read(client_fd, buffer, sizeof(buffer));
        
        if (bytes <= 0) break; 

        std::string request(buffer);
        std::string response_body;

        auto features = parseInput(request);

        if (features.size() == 4) {
            auto result = inference.predict(features);

            std::cout << "raw output: ";
            for (auto& r : result) {
                std::cout << r << " ";
            }
            std::cout << std::endl;

            std::string classes[] = {"setosa", "versicolor", "virginica"};
            int predicted_class = (int)result[0];

            response_body = "{\"class\": \"" + classes[predicted_class] + "\", \"index\": " + std::to_string(predicted_class) + "}";
        } 
        else {
                response_body = "{\"error\": \"нужно 4 параметра: f1, f2, f3, f4\"}";
        }
        
        std::string response = 
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(response_body.size()) + "\r\n"
        "Connection: keep-alive\r\n"
        "\r\n" + response_body;
            
        write(client_fd, response.c_str(), response.size());
    }   
    
    close(client_fd);
}

std::vector<float> TcpServer::parseInput(const std::string& request) {
    std::vector<float> features;

    size_t start = request.find("?");
    if (start == std::string::npos) return features;

    std::string query = request.substr(start + 1);

    for (int i = 1; i <= 4; i++) {
        std::string key = "f" + std::to_string(i) + "=";
        size_t pos = query.find(key);
        if (pos == std::string::npos) continue;

        pos += key.size();
        size_t end = query.find("&", pos);
        std::string value = query.substr(pos, end - pos);
        features.push_back(std::stof(value));
    }
    return features;
}