cmake_minimum_required(VERSION 3.13)
project(tiny_renderer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)


set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(GLOB SOURCE_FILES 
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cu"
)

add_executable(renderer ${SOURCE_FILES})

set_target_properties(renderer PROPERTIES CUDA_ARCHITECTURES OFF)

target_link_libraries(renderer CUDA::cudart)

#set_property(TARGET renderer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/")
target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/tiny_obj_loader")
target_include_directories(renderer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/libs/stb")