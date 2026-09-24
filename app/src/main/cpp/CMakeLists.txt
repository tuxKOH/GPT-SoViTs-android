cmake_minimum_required(VERSION 3.22.1)
project(gsv_cpu_threads LANGUAGES CXX)

add_library(gsv_cpu_threads SHARED cpu_threads.cpp)
target_link_libraries(gsv_cpu_threads dl)
