cmake_minimum_required(VERSION 3.10)
project(microtensor VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/ff233bdd4cac0a0bf6e5cd45bda3406814cb2796.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories("/usr/include/mkl")
link_directories("/usr/lib/x86_64-linux-gnu")

# add_compile_options(-fsanitize=address)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

# Profiling
# Compile, run the program, then
# gprof ./build/lesson-04 gmon.out > analysis.txt
# add_compile_options(-pg)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

add_compile_options(-mavx)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mavx")

# add_compile_options(-fopenmp)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

set(MKL_LIBS mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)

add_library(microtensor_lib array.cpp nn.cpp tensor.cpp)

add_executable(lesson-01 lesson-01.cpp)
target_link_libraries(lesson-01 microtensor_lib ${MKL_LIBS})

add_executable(lesson-02 lesson-02.cpp)
target_link_libraries(lesson-02 microtensor_lib ${MKL_LIBS})

add_executable(lesson-03 lesson-03.cpp)
target_link_libraries(lesson-03 microtensor_lib ${MKL_LIBS})

add_executable(lesson-04 lesson-04.cpp)
target_link_libraries(lesson-04 microtensor_lib ${MKL_LIBS})

add_executable(lesson-05 lesson-05.cpp)
target_link_libraries(lesson-05 microtensor_lib ${MKL_LIBS})

add_executable(lesson-06 lesson-06.cpp)
target_link_libraries(lesson-06 microtensor_lib ${MKL_LIBS})

enable_testing()

add_executable(
  tensor_test
  tensor_test.cpp
)
target_link_libraries(
  tensor_test
  microtensor_lib
  mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl
  GTest::gtest_main
)

add_executable(
  nn_test
  nn_test.cpp
)
target_link_libraries(
  nn_test
  microtensor_lib
  mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_test)
gtest_discover_tests(nn_test)
