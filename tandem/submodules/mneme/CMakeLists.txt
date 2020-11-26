cmake_minimum_required(VERSION 3.6)

project(mneme VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
include(cmake/doctest.cmake)

add_library(mneme-test-runner test/test_main.cpp)
target_include_directories(mneme-test-runner PUBLIC include external)

add_executable(mneme-test test/mneme.cpp)
target_compile_options(mneme-test PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(mneme-test mneme-test-runner)
doctest_discover_tests(mneme-test)

add_executable(displacements-test test/displacements.cpp)
target_compile_options(displacements-test PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(displacements-test mneme-test-runner)
doctest_discover_tests(displacements-test)

add_executable(allocators-test test/allocators.cpp)
target_compile_options(allocators-test PRIVATE -Wall -Wextra -pedantic)
target_link_libraries(allocators-test mneme-test-runner)
doctest_discover_tests(allocators-test)
