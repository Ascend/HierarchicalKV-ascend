cmake_minimum_required(VERSION 3.16)

project(HKV-ascend_UT)

if ("${RUN_MODE}" STREQUAL "sim")
  add_definitions(-DSIM_MODE=1)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")

add_library(test_depends INTERFACE)
target_link_libraries(test_depends INTERFACE gtest gmock gtest_main)
target_include_directories(test_depends INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

file(GLOB TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set_source_files_properties(
    ${TEST_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_score_functor.cpp
    PROPERTIES LANGUAGE ASC
)

list(REMOVE_ITEM TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)
add_executable(test_main ${TEST_SRC_FILES})
target_link_libraries(test_main PRIVATE hkv_depends test_depends)
add_test(NAME test_main COMMAND test_main)

add_executable(test_aclnn_helper ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)
target_link_libraries(test_aclnn_helper PRIVATE hkv_depends test_depends)
add_test(NAME test_aclnn_helper COMMAND test_aclnn_helper)

add_executable(test_score_functor ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_score_functor.cpp)
target_link_libraries(test_score_functor PRIVATE hkv_depends test_depends)
add_test(NAME test_score_functor COMMAND test_score_functor)
