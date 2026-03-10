cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(Ascend_c LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend910_9589" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
  CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

add_library(hkv_includes INTERFACE)
target_include_directories(hkv_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_compile_options(-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp
    PROPERTIES LANGUAGE ASC
)

add_executable(hkv_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_definitions(hkv_demo PRIVATE 
    SOC_VERSION=${SOC_VERSION}
)

target_compile_options(hkv_demo PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    -g -O2
)

target_link_libraries(hkv_demo PRIVATE
  m
  graph_base
  tiling_api
  register
  platform
  ascendalog
  dl
  hkv_includes
  opapi
  nnopbase
)

add_executable(hkv_benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp
)

target_compile_definitions(hkv_benchmark PRIVATE 
    SOC_VERSION=${SOC_VERSION}
)

target_compile_options(hkv_benchmark PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    -g -O2
    -mllvm -disable-machine-licm
    --cce-res-usage
)

target_link_libraries(hkv_benchmark PRIVATE
  m
  graph_base
  tiling_api
  register
  platform
  ascendalog
  dl
  hkv_includes
  opapi
  nnopbase
)

if(ENABLE_TEST STREQUAL 1)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of GTest" FORCE)
  set(INSTALL_GMOCK OFF CACHE BOOL "Disable installation of GMock" FORCE)
  add_subdirectory(3rdparty/gtest)
  add_subdirectory(tests)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
  DESTINATION ./
)
install(TARGETS hkv_demo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS hkv_benchmark
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
