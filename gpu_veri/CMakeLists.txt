cmake_minimum_required(VERSION 3.22)

project(mesh_rail_a2a_verification LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Mesh-Rail validation targets A100 (SM80). Override with
# -DCMAKE_CUDA_ARCHITECTURES=... only when doing a portability build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit 11.8 REQUIRED)
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ENV NCCL_ROOT
  PATH_SUFFIXES include)
find_library(NCCL_LIBRARY nccl
  HINTS ENV NCCL_ROOT
  PATH_SUFFIXES lib lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR
    "NCCL was not found. Install NCCL >= 2.17 or set NCCL_ROOT to its prefix.")
endif()

add_executable(a2a_bench src/a2a_bench.cu)
target_include_directories(a2a_bench PRIVATE "${NCCL_INCLUDE_DIR}")
target_link_libraries(a2a_bench PRIVATE CUDA::cudart "${NCCL_LIBRARY}")
target_compile_options(a2a_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME static_plan_validation
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_plan.py")
    add_test(NAME analysis_unit_tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_analysis.py")
    set_tests_properties(static_plan_validation analysis_unit_tests PROPERTIES
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endif()
endif()
