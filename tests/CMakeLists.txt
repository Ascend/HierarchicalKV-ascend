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
list(REMOVE_ITEM TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)

add_executable(test_main ${TEST_SRC_FILES})
target_link_libraries(test_main PRIVATE
  $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
  ascendc_kernels_${RUN_MODE}
  graph_base
  hkv_includes
  tiling_api
  platform
  opapi
  nnopbase
  kernel_lib
)
target_compile_options(test_main PRIVATE
  -DUSE_DUMP_KERNEL_ASC
)
add_test(NAME test_main COMMAND test_main)

# 开启测试宏会影响其他用例
add_executable(test_aclnn_helper ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_helper.cpp)
target_link_libraries(test_aclnn_helper PRIVATE
  $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
  ascendc_kernels_${RUN_MODE}
  graph_base
  hkv_includes
  tiling_api
  platform
  opapi
  nnopbase
  -lpthread
  kernel_lib
)
target_compile_options(test_aclnn_helper PRIVATE
  -DUSE_DUMP_KERNEL_ASC
)
add_test(NAME test_aclnn_helper COMMAND test_aclnn_helper)

# test_score_functor
if("${RUN_MODE}" STREQUAL "npu" OR "${RUN_MODE}" STREQUAL "sim")
  set(TEST_KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_desired_when_missed_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_update_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_update_with_digest_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_update_without_missed_bucket_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_update_without_missed_ptr_kernel.cpp
  )

  ascendc_library(test_score_functor_kernels_${RUN_MODE} STATIC ${TEST_KERNEL_FILES})
  # 由于ascendc_library编译一定会将算子头文件安装出去，那么这里只能在安装完成后删掉冗余的测试算子头文件
  install(CODE "
    set(DELETE_DIR \${CMAKE_INSTALL_PREFIX}/include/test_score_functor_kernels_\${RUN_MODE})
    if(EXISTS \${DELETE_DIR} AND IS_DIRECTORY \${DELETE_DIR})
      message(STATUS \"Removing unnecessary directory: \${DELETE_DIR}\")
      file(REMOVE_RECURSE \${DELETE_DIR})
    endif()
  ")
endif()

# 编译测试可执行文件
add_executable(test_score_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/test_score_functor/test_score_functor.cpp
)
target_link_libraries(test_score_functor PRIVATE
  $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
  $<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:test_score_functor_kernels_${RUN_MODE}>
  ascendc_kernels_${RUN_MODE}
  graph_base
  hkv_includes
  tiling_api
  platform
  opapi
  nnopbase
  kernel_lib
)
target_compile_options(test_score_functor PRIVATE
  -DUSE_DUMP_KERNEL_ASC
)
add_test(NAME test_score_functor COMMAND test_score_functor)