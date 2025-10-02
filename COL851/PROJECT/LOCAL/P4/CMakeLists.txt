# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(json_analysis)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(dummy main.cpp)
target_include_directories(dummy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
