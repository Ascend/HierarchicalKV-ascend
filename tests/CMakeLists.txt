cmake_minimum_required(VERSION 3.16)

project(HKV_UT)

# 设置module搜索路径
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../cmake)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")

# 所有用例都依赖gtest
link_libraries(gtest gmock gtest_main)
if ("${RUN_MODE}" STREQUAL "sim")
  add_definitions(-DSIM_MODE=1)
endif()

file(GLOB TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set_source_files_properties(
    ${TEST_SRC_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_score_functor.cpp
    PROPERTIES LANGUAGE ASC
)

list(REMOVE_ITEM TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)
add_executable(test_main 
  ${TEST_SRC_FILES}
)

target_compile_options(test_main PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    -g -O2
)
target_link_libraries(test_main PRIVATE
  m
  graph_base
  tiling_api
  register
  platform
  hkv_includes
  opapi
  nnopbase
)
add_test(NAME test_main COMMAND test_main)


add_executable(test_aclnn_helper ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)
target_compile_options(test_aclnn_helper PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    -g -O2
)
target_link_libraries(test_aclnn_helper PRIVATE
  m
  graph_base
  tiling_api
  register
  platform
  hkv_includes
  opapi
  nnopbase
  -lpthread
)
add_test(NAME test_aclnn_helper COMMAND test_aclnn_helper)


add_executable(test_score_functor ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_score_functor.cpp)
target_compile_options(test_score_functor PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    -g -O2
)
target_link_libraries(test_score_functor PRIVATE
  m
  graph_base
  tiling_api
  platform
  hkv_includes
  opapi
  nnopbase
)
add_test(NAME test_score_functor COMMAND test_score_functor)
