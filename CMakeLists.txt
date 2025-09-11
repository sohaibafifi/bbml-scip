cmake_minimum_required(VERSION 3.18)
project(bbml LANGUAGES CXX)

option(BBML_WITH_ONNX "Enable ONNXRuntime inference" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(SCIP_INCLUDE_DIR scip/scip.h HINTS $ENV{SCIP_DIR}/include ${SCIP_DIR}/include)
find_library(SCIP_LIBRARY NAMES scip HINTS $ENV{SCIP_DIR}/lib ${SCIP_DIR}/lib)
if(NOT SCIP_INCLUDE_DIR OR NOT SCIP_LIBRARY)
  message(FATAL_ERROR "SCIP not found. Set SCIP_DIR to your install prefix.")
endif()

add_subdirectory(cpp)

# Tests (only if GTest is available)
include(CTest)
enable_testing()
find_package(GTest QUIET)
if(GTest_FOUND)
  message(STATUS "GTest found, enabling tests")
  add_subdirectory(tests/cpp)
else ()
    message(STATUS "GTest not found, skipping tests")
endif()

# Example runner that loads the plugin and solves an instance
add_executable(bbml_run
  examples/bbml_run.cpp
  cpp/src/branchrule_ml.cpp
  cpp/src/nodesel_ml.cpp
  cpp/src/feature_extractor.cpp
  cpp/src/telemetry_logger.cpp
  cpp/src/onnx_runner.cpp
)
target_include_directories(bbml_run PRIVATE ${SCIP_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/cpp/include)
target_link_libraries(bbml_run PRIVATE ${SCIP_LIBRARY})
if(BBML_WITH_LP_STATS)
  target_compile_definitions(bbml_run PRIVATE BBML_WITH_LP_STATS=1)
endif()
if(BBML_WITH_ONNX)
  # mirror plugin ONNX linkage
  set(ORT_HINT_LIB_DIRS
      $ENV{ONNXRUNTIME_DIR}/lib
      ${ONNXRUNTIME_DIR}/lib
      $ENV{ONNXRUNTIME_DIR}/dist/lib
      ${ONNXRUNTIME_DIR}/dist/lib)
  set(ORT_HINT_INC_DIRS
      $ENV{ONNXRUNTIME_DIR}/include
      ${ONNXRUNTIME_DIR}/include
      $ENV{ONNXRUNTIME_DIR}/dist/include
      ${ONNXRUNTIME_DIR}/dist/include)
  find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ORT_HINT_LIB_DIRS})
  if(ONNXRUNTIME_LIB)
    target_link_libraries(bbml_run PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(bbml_run PRIVATE BBML_WITH_ONNX=1)
    target_include_directories(bbml_run PRIVATE ${ORT_HINT_INC_DIRS})
  endif()
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(bbml_run PRIVATE dl)
endif()
