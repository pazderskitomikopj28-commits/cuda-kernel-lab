cmake_minimum_required(VERSION 3.24)
project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Override with -DCMAKE_CUDA_ARCHITECTURES=80,90 for a specific machine.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70;80;90)
endif()

add_library(kernel_lab STATIC src/kernels.cu)
target_include_directories(kernel_lab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(kernel_lab PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(kernel_lab PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)

add_executable(kernel_bench src/benchmark.cu)
target_link_libraries(kernel_bench PRIVATE kernel_lab)

add_executable(kernel_tests tests/test_kernels.cu)
target_link_libraries(kernel_tests PRIVATE kernel_lab)

enable_testing()
add_test(NAME kernel_tests COMMAND kernel_tests)
