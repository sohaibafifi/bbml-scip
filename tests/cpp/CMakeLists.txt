if(NOT GTest_FOUND)
  message(STATUS "GTest not found; skipping C++ unit tests build")
  return()
endif()

add_executable(cpp_tests
  test_feature_extractor.cpp
  test_branchrule_choose.cpp
  test_branchrule_params.cpp
  ${CMAKE_SOURCE_DIR}/cpp/src/branchrule_ml.cpp
  ${CMAKE_SOURCE_DIR}/cpp/src/onnx_runner.cpp
  ${CMAKE_SOURCE_DIR}/cpp/src/feature_extractor.cpp
  ${CMAKE_SOURCE_DIR}/cpp/src/telemetry_logger.cpp
)

target_include_directories(cpp_tests PRIVATE
  ${SCIP_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/include
)

target_link_libraries(cpp_tests PRIVATE GTest::gtest_main ${SCIP_LIBRARY})

add_test(NAME bbml_cpp_tests COMMAND cpp_tests)
