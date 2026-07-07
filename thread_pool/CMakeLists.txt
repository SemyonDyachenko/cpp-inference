cmake_minimum_required(VERSION 3.10)
project(thread_pool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(ONNXRUNTIME_LIB onnxruntime)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime)


find_package(Threads REQUIRED)

include_directories(include)

add_executable(thread_pool
    src/main.cpp
    src/thread_pool.cpp
    src/server.cpp
    src/inference/inference.cpp
)

target_include_directories(thread_pool PRIVATE 
    include
    /opt/homebrew/opt/onnxruntime/include/onnxruntime
)

target_link_libraries(thread_pool PRIVATE 
    /opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib
)
target_compile_options(thread_pool PRIVATE -pthread)
