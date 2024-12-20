#!/usr/bin/env bash

# Set the offload architecture variable here for easy modification
OFFLOAD_ARCH="sm_75"

echo "Compiling combined_3diffkernels_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
nvcc -arch=${OFFLOAD_ARCH} complex-kernels-test/combined_3diffkernels_singlerun.cu -o complex-kernels-test/combined_3diffkernels_singlerun

echo "Entering complex-kernels-test directory"
cd complex-kernels-test/
echo "Running combined_3diffkernels_singlerun with arguments 10000 0"
./combined_3diffkernels_singlerun 10000 0
cd ..

echo "Compiling combined_diffsize_kernels_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
nvcc -arch=${OFFLOAD_ARCH} diffsize-kernels-test/combined_diffsize_kernels_singlerun.cu -o diffsize-kernels-test/combined_diffsize_kernels_singlerun

echo "Entering diffsize-kernels-test directory"
cd diffsize-kernels-test/
echo "Running combined_diffsize_kernels_singlerun with arguments 10000 0"
./combined_diffsize_kernels_singlerun 10000 0
cd ..

echo "Compiling combined_multi_malloc_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
nvcc -arch=${OFFLOAD_ARCH} multi-malloc-test/combined_multi_malloc_singlerun.cu -o multi-malloc-test/combined_multi_malloc_singlerun

echo "Entering multi-malloc-test directory"
cd multi-malloc-test/
echo "Running combined_multi_malloc_singlerun with arguments 10000 0"
./combined_multi_malloc_singlerun 10000 0
cd ..

echo "Compiling combined_multi_stream2_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
nvcc -arch=${OFFLOAD_ARCH} multi-stream-test/combined_multi_stream2_singlerun.cu -o multi-stream-test/combined_multi_stream2_singlerun

echo "Entering multi-stream-test directory"
cd multi-stream-test/
echo "Running combined_multi_stream2_singlerun with arguments 10000 0"
./combined_multi_stream2_singlerun 10000 0
cd ..

echo "Running generate_plots.sh"
bash generate_plots.sh

echo "All steps completed successfully."

