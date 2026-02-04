if(NOT DEFINED GTEST_HOME)
  set(GTEST_HOME ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/googletest)
  message(WARNING "GTEST_HOME undefined, using default [${GTEST_HOME}]")
endif()

add_library(gtest SHARED IMPORTED)
set_target_properties(gtest
  PROPERTIES
    IMPORTED_LOCATION ${GTEST_HOME}/lib/libgtest.so
)

add_library(gmock SHARED IMPORTED)
set_target_properties(gmock
  PROPERTIES
    IMPORTED_LOCATION ${GTEST_HOME}/lib/libgmock.so
)

add_library(gmock_main SHARED IMPORTED)
set_target_properties(gmock_main
  PROPERTIES
    IMPORTED_LOCATION ${GTEST_HOME}/lib/libgmock_main.so
)

set_target_properties(gtest gmock gmock_main 
  PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${GTEST_HOME}/include
)

add_library(gtest_module INTERFACE)
target_link_libraries(gtest_module INTERFACE
  gtest
  gmock
  gmock_main
)
