cmake_minimum_required(VERSION 3.13)
project(tiny_renderer LANGUAGES CXX )
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)   

#set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
#set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(GLOB SOURCE_FILES 
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cu"
)


add_executable(renderer ${SOURCE_FILES})


target_link_libraries(renderer MPI::MPI_CXX)

target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/")
target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/tiny_obj_loader")
target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/stb")