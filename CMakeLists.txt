cmake_minimum_required(VERSION 3.18)
project(bbml LANGUAGES CXX)

option(BBML_WITH_ONNX "Enable ONNXRuntime inference" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}" CACHE PATH
    "Path to the ONNX Runtime install prefix")

if(BBML_WITH_ONNX AND NOT ONNXRUNTIME_DIR)
  file(GLOB BBML_ORT_BUNDLED_DIRS CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/libs/onnxruntime-*"
       "${CMAKE_SOURCE_DIR}/libs/onnxruntime*")
  foreach(_ort_dir IN LISTS BBML_ORT_BUNDLED_DIRS)
    if(EXISTS "${_ort_dir}/lib/libonnxruntime.dylib"
       OR EXISTS "${_ort_dir}/lib/libonnxruntime.so"
       OR EXISTS "${_ort_dir}/lib/onnxruntime.lib"
       OR EXISTS "${_ort_dir}/dist/lib/libonnxruntime.dylib"
       OR EXISTS "${_ort_dir}/dist/lib/libonnxruntime.so"
       OR EXISTS "${_ort_dir}/dist/lib/onnxruntime.lib")
      set(ONNXRUNTIME_DIR "${_ort_dir}" CACHE PATH
          "Path to the ONNX Runtime install prefix" FORCE)
      message(STATUS "Auto-detected ONNXRuntime under ${ONNXRUNTIME_DIR}")
      break()
    endif()
  endforeach()
endif()

if(BBML_WITH_ONNX)
  set(BBML_ORT_HINT_LIB_DIRS
      $ENV{ONNXRUNTIME_DIR}/lib
      ${ONNXRUNTIME_DIR}/lib
      $ENV{ONNXRUNTIME_DIR}/dist/lib
      ${ONNXRUNTIME_DIR}/dist/lib)
  set(BBML_ORT_HINT_INC_DIRS
      $ENV{ONNXRUNTIME_DIR}/include
      ${ONNXRUNTIME_DIR}/include
      $ENV{ONNXRUNTIME_DIR}/dist/include
      ${ONNXRUNTIME_DIR}/dist/include
      $ENV{ONNXRUNTIME_DIR}/include/onnxruntime/core/session
      ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session
      $ENV{ONNXRUNTIME_DIR}/dist/include/onnxruntime/core/session
      ${ONNXRUNTIME_DIR}/dist/include/onnxruntime/core/session)
  find_library(BBML_ONNXRUNTIME_LIB onnxruntime HINTS ${BBML_ORT_HINT_LIB_DIRS})
  if(BBML_ONNXRUNTIME_LIB)
    message(STATUS "ONNXRuntime found: ${BBML_ONNXRUNTIME_LIB}")
  else()
    message(FATAL_ERROR
            "BBML_WITH_ONNX is ON but onnxruntime was not found. "
            "Set ONNXRUNTIME_DIR to the folder containing include/ and lib/ "
            "or dist/include and dist/lib.")
  endif()
endif()

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
  target_link_libraries(bbml_run PRIVATE ${BBML_ONNXRUNTIME_LIB})
  target_compile_definitions(bbml_run PRIVATE BBML_WITH_ONNX=1)
  target_include_directories(bbml_run PRIVATE ${BBML_ORT_HINT_INC_DIRS})
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(bbml_run PRIVATE dl)
endif()
