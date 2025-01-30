#!/usr/bin/env bash

# Set the offload architecture variable here for easy modification
OFFLOAD_ARCH="sm_75"
NSTEPS=10000
SKIPBY=0
NUM_RUNS=4

# CSV Filenames
Test1_Filename="complex_3_different_kernels.csv"
Test2_Filename="complex_different_sizes_kernels.csv"
Test3_Filename="complex_multi_malloc.csv"
Test4_Filename="complex_multi_stream_kernels.csv"

# Ensure nvcc compiles both CUDA and C++ files properly
COMPILE="nvcc -arch=${OFFLOAD_ARCH} -std=c++20"

# Utility file path
CSV_UTIL="util/csv_util.cpp"

# Command
COMMAND="${COMPILE} ${CSV_UTIL}"

echo "Compiling combined_3diffkernels_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
${COMMAND} complex-kernels-test/combined_3diffkernels_singlerun.cu -o complex-kernels-test/combined_3diffkernels_singlerun

echo "Entering complex-kernels-test directory"
cd complex-kernels-test/ || exit 1
echo "Running combined_3diffkernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_3diffkernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test1_Filename}
cd ..

echo "Compiling combined_diffsize_kernels_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
${COMMAND} diffsize-kernels-test/combined_diffsize_kernels_singlerun.cu -o diffsize-kernels-test/combined_diffsize_kernels_singlerun

echo "Entering diffsize-kernels-test directory"
cd diffsize-kernels-test/ || exit 1
echo "Running combined_diffsize_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_diffsize_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test2_Filename}
cd ..

echo "Compiling combined_multi_malloc_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
${COMMAND} multi-malloc-test/combined_multi_malloc_singlerun.cu -o multi-malloc-test/combined_multi_malloc_singlerun

echo "Entering multi-malloc-test directory"
cd multi-malloc-test/ || exit 1
echo "Running combined_multi_malloc_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_malloc_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test3_Filename}
cd ..

echo "Compiling combined_multi_stream2_singlerun.cu with OFFLOAD_ARCH=${OFFLOAD_ARCH}"
${COMMAND} multi-stream-test/combined_multi_stream_singlerun.cu -o multi-stream-test/combined_multi_stream_singlerun

echo "Entering multi-stream-test directory"
cd multi-stream-test/ || exit 1
echo "Running combined_multi_stream_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_stream_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test4_Filename}
cd ..

echo "Running generate_plots.sh with NUM_RUNS=$NUM_RUNS"
bash generate_plots.sh "${NUM_RUNS}"

echo "All steps completed successfully."
