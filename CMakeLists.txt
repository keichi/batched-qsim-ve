cmake_minimum_required(VERSION 3.16)
project(batched-qsim CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.1.1
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

find_package(OpenMP REQUIRED)

add_library(qsim src/state_openmp.cpp)
target_link_libraries(qsim OpenMP::OpenMP_CXX)
if(CMAKE_CXX_COMPILER_ID STREQUAL "NEC")
    target_link_libraries(qsim asl_openmp)
endif()

add_executable(qsim-gate-bench src/gate_benchmark.cpp)
target_link_libraries(qsim-gate-bench qsim cxxopts OpenMP::OpenMP_CXX)

add_executable(qsim-random-circuit src/random_circuit.cpp)
target_link_libraries(qsim-random-circuit qsim cxxopts OpenMP::OpenMP_CXX)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(qsim-test src/test.cpp)
target_link_libraries(qsim-test qsim doctest OpenMP::OpenMP_CXX)

doctest_discover_tests(qsim-test)
