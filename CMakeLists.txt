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

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.7/site-packages/veda/cmake)
find_package(VEDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(qsim src/state_custatevec.cu)
    target_link_libraries(qsim custatevec)
elseif(VEDA_FOUND)
    find_package(VEDA)
    enable_language(VEDA_C VEDA_CXX)
    include_directories(${VEDA_INCLUDE_DIRS})
    add_library(qsim_device SHARED src/state_veda_kernels.vc)
    target_link_libraries(qsim_device ${VEDA_DEVICE_LIBRARY})
    add_library(qsim src/state_veda.cpp src/state_veda_kernels.vc)
    target_link_libraries(qsim ${VEDA_LIBRARY})
else()
    find_package(OpenMP REQUIRED)
    add_library(qsim src/state_openmp.cpp)
    target_link_libraries(qsim OpenMP::OpenMP_CXX)
endif()

add_executable(qsim-gate-bench src/gate_benchmark.cpp)
target_link_libraries(qsim-gate-bench qsim cxxopts)

add_executable(qsim-random-circuit src/random_circuit.cpp)
target_link_libraries(qsim-random-circuit qsim cxxopts)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(qsim-test src/test.cpp)
target_link_libraries(qsim-test qsim doctest)

doctest_discover_tests(qsim-test)
