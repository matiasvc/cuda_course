cmake_minimum_required(VERSION 3.16)
project(cuda_hello_world LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(cuda_hello_world main.cu)

set_target_properties(
		cuda_hello_world
		PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(cuda_hello_world
                      ${CUDA_LIBRARIES})
