cmake_minimum_required(VERSION 3.15)
project(opencl_labs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")

add_executable(test main.cpp)

add_executable(mul_matrix mul_matrix.cpp)
target_link_libraries(mul_matrix OpenCL)