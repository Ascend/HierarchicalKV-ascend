cmake_minimum_required(VERSION 3.16)

# 1. 启用ASC工程
find_package(ASC REQUIRED)
project(HierarchicalKV-ascend
  LANGUAGES ASC CXX)

# 2. cmake基础参数设置
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# 3. hkv依赖库
set(hkv_intf ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(hkv_depends INTERFACE)
target_include_directories(hkv_depends INTERFACE ${hkv_intf})
target_link_libraries(hkv_depends INTERFACE
  # c++
  m
  dl # platform依赖

  # cann
  tiling_api # platform
  platform # fe::platform
  opapi # aclnnApi
  nnopbase # aclTensor
  pthread # platform依赖
)
target_compile_options(hkv_depends INTERFACE
  # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
  -g -O2
  # 优化SIMT算子寄存器数
  -mllvm -disable-machine-licm
  # 打印寄存器使用情况
  --cce-res-usage
)

# 4. 测试工程
if(ENABLE_TEST STREQUAL 1)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of GTest" FORCE)
  set(INSTALL_GMOCK OFF CACHE BOOL "Disable installation of GMock" FORCE)
  add_subdirectory(3rdparty/gtest)
  add_subdirectory(tests)
endif()

# 5. 编译示例
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_find_miss.cpp
  PROPERTIES LANGUAGE ASC
)

add_executable(hkv_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(hkv_demo PRIVATE hkv_depends)

add_executable(hkv_benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp
)
target_link_libraries(hkv_benchmark PRIVATE hkv_depends)

add_executable(hkv_benchmark_find_miss
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_find_miss.cpp
)
target_link_libraries(hkv_benchmark_find_miss PRIVATE hkv_depends)

# 6. 安装二进制
install(TARGETS hkv_demo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS hkv_benchmark hkv_benchmark_find_miss
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
