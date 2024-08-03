cmake_minimum_required(VERSION 3.10)
project(microtensor VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(microtensor_lib array.cpp tensor.cpp)

add_executable(lesson-01 lesson-01.cpp)
target_link_libraries(lesson-01 microtensor_lib)

add_executable(lesson-02 lesson-02.cpp)
target_link_libraries(lesson-02 microtensor_lib)
