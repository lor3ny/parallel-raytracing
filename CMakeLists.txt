cmake_minimum_required(VERSION 3.13)
project(tiny_renderer)
set(CMAKE_CXX_STANDARD 17)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(GLOB SOURCE_FILES 
    "${SRC_DIR}/*.cpp"
)

add_executable(renderer ${SOURCE_FILES})

target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/")