cmake_minimum_required(VERSION 3.10)
project(opencl)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(mul_matrix mul_matrix.cpp)
target_link_libraries(mul_matrix OpenCL)

add_executable(prefix_sum prefix_sum.cpp)
target_link_libraries(prefix_sum OpenCL)