cmake_minimum_required(VERSION 3.18)
project(veqsim CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.2.1
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12
)
FetchContent_MakeAvailable(doctest)

include(CheckLanguage)
check_language(CUDA)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
list(APPEND CMAKE_MODULE_PATH $ENV{HOME}/.local/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/veda/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/veda/cmake)
find_package(VEDA)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(veqsim-core src/state_custatevec.cu)
elseif(VEDA_FOUND)
    find_package(VEDA)
    enable_language(VEDA_C VEDA_CXX)
    include_directories(${VEDA_INCLUDE_DIRS})
    add_library(veqsim-device SHARED src/state_veda_kernels.vc)
    target_link_libraries(veqsim-device ${VEDA_DEVICE_LIBRARY})
    add_library(veqsim-core src/state_veda.cpp)
    target_link_libraries(veqsim-core ${VEDA_LIBRARY})
else()
    find_package(OpenMP REQUIRED)
    add_library(veqsim-core src/state_openmp.cpp)
    target_link_libraries(veqsim-core OpenMP::OpenMP_CXX)
endif()

set_target_properties(veqsim-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(veqsim python/bindings.cpp)
target_link_libraries(veqsim PRIVATE veqsim-core)

add_executable(veqsim-gate-bench src/gate_benchmark.cpp)
target_link_libraries(veqsim-gate-bench veqsim-core cxxopts)

add_executable(veqsim-random-circuit src/random_circuit.cpp)
target_link_libraries(veqsim-random-circuit veqsim-core cxxopts)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(veqsim-test src/test.cpp)
target_link_libraries(veqsim-test veqsim-core doctest)

doctest_discover_tests(veqsim-test)
