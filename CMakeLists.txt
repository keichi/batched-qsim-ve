cmake_minimum_required(VERSION 3.16)
project(batched-qsim CXX)

include(FetchContent)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG        v2.9
)
FetchContent_MakeAvailable(argparse)

find_package(OpenMP REQUIRED)

add_executable(qsim-bench src/main.cpp src/gate.cpp)
target_link_libraries(qsim-bench argparse OpenMP::OpenMP_CXX)
