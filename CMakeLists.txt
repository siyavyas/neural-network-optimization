cmake_minimum_required(VERSION 3.29.6)
project(Optimization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/target/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/target/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/target/lib)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(test)
