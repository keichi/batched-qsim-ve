cmake_minimum_required(VERSION 3.18)
project(veqsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.2.1
)
FetchContent_MakeAvailable(cxxopts)

# Stop installing doctest
set(DOCTEST_NO_INSTALL ON)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12
)
FetchContent_MakeAvailable(doctest)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

include(CheckLanguage)
check_language(CUDA)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import veda; print(veda.__path__[0] + '/cmake')"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE VEDA_ROOT)
list(APPEND CMAKE_MODULE_PATH ${VEDA_ROOT})
find_package(VEDA QUIET)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(veqsim src/state_custatevec.cu)
    target_link_libraries(veqsim custatevec)
elseif(VEDA_FOUND)
    find_package(VEDA)
    enable_language(VEDA_C VEDA_CXX)
    include_directories(${VEDA_INCLUDE_DIRS})
    add_library(veqsim-device SHARED src/state_veda_kernels.vc)
    target_link_libraries(veqsim-device ${VEDA_DEVICE_LIBRARY} asl_sequential)
    add_library(veqsim src/state_veda.cpp)
    target_link_libraries(veqsim ${VEDA_LIBRARY})
else()
    find_package(OpenMP REQUIRED)
    add_library(veqsim src/state_openmp.cpp)
    target_link_libraries(veqsim OpenMP::OpenMP_CXX)
endif()

set_target_properties(veqsim PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_veqsim NB_STATIC python/bindings.cpp)
target_link_libraries(_veqsim PRIVATE veqsim)
install(TARGETS _veqsim LIBRARY DESTINATION .)

add_executable(veqsim-gate-bench src/gate_benchmark.cpp)
target_link_libraries(veqsim-gate-bench veqsim cxxopts)

add_executable(veqsim-random-circuit src/random_circuit.cpp)
target_link_libraries(veqsim-random-circuit veqsim cxxopts)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(veqsim-test src/test.cpp)
target_link_libraries(veqsim-test veqsim doctest)

doctest_discover_tests(veqsim-test)
