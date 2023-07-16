cmake_minimum_required(VERSION 3.16)
project(batched-qsim CXX)

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

add_executable(qsim-bench src/main.cpp src/gate.cpp)
target_link_libraries(qsim-bench cxxopts OpenMP::OpenMP_CXX)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(qsim-test src/test.cpp src/gate.cpp)
target_link_libraries(qsim-test doctest)

doctest_discover_tests(qsim-test)
