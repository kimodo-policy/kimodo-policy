# Source for SONIC build/run only; does not change system or conda environments.
export KIMODO_COLLECTION_ROOT="${KIMODO_COLLECTION_ROOT:-${COLLECTION_ROOT:?Set COLLECTION_ROOT to the workstation root}}"
export WBC_ROOT="${WBC_ROOT:-${REAL_WORLD_ROOT:?Source env.sh first}/external/GR00T-WholeBodyControl}"
export SONIC_NATIVE_ROOT="${SONIC_NATIVE_ROOT:-$KIMODO_COLLECTION_ROOT/native}"
export CUDA_HOME="$SONIC_NATIVE_ROOT/usr/local/cuda-11.8"
export CUDAToolkit_ROOT="$CUDA_HOME"
export TensorRT_ROOT="$SONIC_NATIVE_ROOT/tensorrt"
export onnxruntime_ROOT="$SONIC_NATIVE_ROOT/onnxruntime"
export HAS_ROS2=0
export CC="$SONIC_NATIVE_ROOT/usr/bin/gcc-10"
export CXX="$SONIC_NATIVE_ROOT/usr/bin/g++-10"
export PATH="$(dirname "${TELEOP_PYTHON:?Set TELEOP_PYTHON}"):$SONIC_NATIVE_ROOT/usr/bin:/usr/bin:/bin"
export CMAKE_PREFIX_PATH="$SONIC_NATIVE_ROOT/usr:$TensorRT_ROOT:$onnxruntime_ROOT"
export LD_LIBRARY_PATH="$SONIC_NATIVE_ROOT/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$onnxruntime_ROOT/lib"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
