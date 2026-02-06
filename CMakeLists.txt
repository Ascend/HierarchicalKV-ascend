cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(Ascend_c LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
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

file(GLOB KERNEL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/init_table_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/allocate_bucket_others_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/get_bucket_others_address_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/create_atomic_keys_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/init_table_kernel/create_atomic_scores_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_or_insert_ptr_kernel/find_or_insert_ptr_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_or_insert_ptr_kernel/find_or_insert_ptr_kernel_v2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_ptr_kernel/find_ptr_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_ptr_kernel/find_ptr_with_digest_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/utils_kernel/read_from_ptr_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/utils_kernel/host_nano_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/insert_or_assign_kernel/insert_or_assign_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/insert_or_assign_kernel/insert_or_assign_kernel_with_thread_1024.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/clear_kernel/clear_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/rehash_kernel/rehash_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_and_update_kernel/find_and_update_kernel_with_filter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/find_and_update_kernel/find_and_update_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/assign_scores_kernel/assign_scores_kernel_with_filter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/assign_scores_kernel/assign_scores_kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/insert_and_evict_kernel/insert_and_evict_kernel.cpp
)

if("${RUN_MODE}" STREQUAL "cpu")
  include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
  include(cmake/npu_lib.cmake)
else()
  message("invalid RUN_MODE: ${RUN_MODE}")
endif()

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/dump_kernel/dump_kernel.cpp
    PROPERTIES LANGUAGE ASC
)

add_library(kernel_lib SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/hkv_hashtable/dump_kernel/dump_kernel.cpp
)

set_target_properties(kernel_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

target_compile_options(kernel_lib PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-3101>
  -DUSE_DUMP_KERNEL_ASC
)

install(TARGETS kernel_lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_executable(hkv_demo ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(hkv_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hkv_hashtable_benchmark.cpp)

add_library(hkv_includes INTERFACE)
target_include_directories(hkv_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_compile_options(-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)

target_compile_options(hkv_demo PRIVATE
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
  -O2 -Wall -Werror -DSOC_VERSION="${SOC_VERSION}"
  -DUSE_DUMP_KERNEL_ASC
)
target_compile_options(hkv_benchmark PRIVATE
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
  -O2 -Wall -Werror -DSOC_VERSION="${SOC_VERSION}"
  -DUSE_DUMP_KERNEL_ASC
)

get_target_property(dep_options host_intf_pub INTERFACE_COMPILE_OPTIONS)
if (dep_options)
  list(FILTER dep_options EXCLUDE REGEX ".*std=.*")
  list(FILTER dep_options EXCLUDE REGEX "-fvisibility-inlines-hidden")
  list(APPEND dep_options -std=c++17 -fvisibility-inlines-hidden)
  set_target_properties(host_intf_pub PROPERTIES INTERFACE_COMPILE_OPTIONS "${dep_options}")
endif()

target_link_libraries(hkv_demo PRIVATE
  $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
  ascendc_kernels_${RUN_MODE}
  graph_base
  tiling_api
  register
  platform
  ascendalog
  dl
  hkv_includes
  opapi
  nnopbase
  kernel_lib
)
target_link_libraries(hkv_benchmark PRIVATE
  $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
  ascendc_kernels_${RUN_MODE}
  graph_base
  tiling_api
  register
  platform
  ascendalog
  dl
  hkv_includes
  opapi
  nnopbase
  kernel_lib
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
