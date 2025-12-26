#!/bin/bash

# LibTorch编译脚本

set -e

echo "========================================"
echo "Building LibTorch Data Parallelism"
echo "========================================"

# 检查LibTorch路径
# if [ -z "$LIBTORCH_PATH" ]; then
#     echo "Error: LIBTORCH_PATH is not set!"
#     echo "Please set it to your LibTorch installation directory:"
#     echo "  export LIBTORCH_PATH=/home/whu/libtorch/share/cmake/Torch"
#     exit 1
# fi

# if [ ! -d "$LIBTORCH_PATH" ]; then
#     echo "Error: LibTorch directory not found: $LIBTORCH_PATH"
#     exit 1
# fi

# echo "LibTorch Path: $LIBTORCH_PATH"

# 检查MPI
if ! command -v mpirun &> /dev/null; then
    echo "Error: MPI not found!"
    echo "Please install MPI (OpenMPI or MPICH)"
    exit 1
fi

echo "MPI Found: $(which mpirun)"

# 创建build目录
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Configuring with CMake..."
echo "----------------------------------------"

# 配置CMake
cmake .. \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "Building..."
echo "----------------------------------------"

# 编译 (使用所有可用核心)
make -j$(nproc)

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Executable: $BUILD_DIR/dp_libtorch"
echo ""
echo "Usage examples:"
echo "  # Basic training"
echo "  mpirun -np 4 ./dp_libtorch basic train"
echo ""
echo "  # Performance benchmark"
echo "  mpirun -np 4 ./dp_libtorch basic benchmark"
echo ""
echo "  # Matrix multiplication"
echo "  mpirun -np 4 ./dp_libtorch matrix benchmark"
echo ""
echo "  # Verify correctness"
echo "  mpirun -np 4 ./dp_libtorch matrix verify"
echo ""