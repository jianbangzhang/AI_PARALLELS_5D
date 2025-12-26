cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(DataParallelism_LibTorch)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/home/whu/softwares/libtorch-shared-with-deps-2.9.0+cu126/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")

# 查找 LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 查找 MPI (用于分布式训练)
find_package(MPI REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
    src/main.cpp
    src/dp_basic.cpp
    src/dp_matrix_multiply.cpp
)

# 可执行文件
add_executable(dp_libtorch ${SOURCES})

# 链接库
target_link_libraries(dp_libtorch 
    ${TORCH_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    pthread
    dl
)

# Debug模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(flash_atten_main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# 设置 RPATH
set_target_properties(dp_libtorch PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# 编译选项
target_compile_options(dp_libtorch PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
)

# 打印配置信息
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "LibTorch Data Parallelism Configuration")
message(STATUS "==========================================")
message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch dir: ${TORCH_INSTALL_PREFIX}")
message(STATUS "CUDA available: ${TORCH_CUDA_AVAILABLE}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==========================================")
message(STATUS "")

# 安装
install(TARGETS dp_libtorch DESTINATION bin)