cmake_minimum_required(VERSION 3.16)

project(HKV-ascend_CommunityTests)

if ("${RUN_MODE}" STREQUAL "sim")
  add_definitions(-DSIM_MODE=1)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")

add_library(community_test_depends INTERFACE)
target_link_libraries(community_test_depends INTERFACE gtest gmock gtest_main)

file(GLOB COMMUNITY_TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(COMMUNITY_TEST_SRC_FILE ${COMMUNITY_TEST_SRC_FILES})
  get_filename_component(COMMUNITY_TEST_NAME ${COMMUNITY_TEST_SRC_FILE} NAME_WE)
  set(COMMUNITY_TEST_TARGET "community_${COMMUNITY_TEST_NAME}")

  set_source_files_properties(
      ${COMMUNITY_TEST_SRC_FILE}
      PROPERTIES LANGUAGE ASC
  )

  add_executable(${COMMUNITY_TEST_TARGET} ${COMMUNITY_TEST_SRC_FILE})
  target_link_libraries(
      ${COMMUNITY_TEST_TARGET}
      PRIVATE hkv_depends community_test_depends
  )
  add_test(NAME ${COMMUNITY_TEST_TARGET} COMMAND ${COMMUNITY_TEST_TARGET})
endforeach()
