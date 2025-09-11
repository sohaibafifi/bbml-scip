option(BBML_WITH_LP_STATS "Collect LP refactor/condition stats via LPI" OFF)

file(GLOB BBML_SRC src/*.cpp)
add_library(bbml_scip MODULE ${BBML_SRC})
set_target_properties(bbml_scip PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME "bbml_scip")
target_include_directories(bbml_scip PRIVATE ${SCIP_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(bbml_scip PRIVATE ${SCIP_LIBRARY})
if(BBML_WITH_ONNX)
  # Support both flat and 'dist' layouts for ONNXRUNTIME_DIR
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
    message(STATUS "ONNXRuntime found: ${ONNXRUNTIME_LIB}")
    target_link_libraries(bbml_scip PRIVATE ${ONNXRUNTIME_LIB})
    target_compile_definitions(bbml_scip PRIVATE BBML_WITH_ONNX=1)
    target_include_directories(bbml_scip PRIVATE ${ORT_HINT_INC_DIRS})
  else()
    message(WARNING "BBML_WITH_ONNX is ON but onnxruntime not found under ${ONNXRUNTIME_DIR}. "
                    "Set ONNXRUNTIME_DIR to the folder containing include/ and lib/ or dist/include and dist/lib.")
  endif()
else()
  message(STATUS "ONNX support disabled")
endif()
if(BBML_WITH_LP_STATS)
  target_compile_definitions(bbml_scip PRIVATE BBML_WITH_LP_STATS=1)
endif()
