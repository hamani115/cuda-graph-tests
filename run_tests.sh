#!/usr/bin/env bash

# Set the offload architecture variable here for easy modification
ARCH="sm_75"
NSTEPS=10000
SKIPBY=0
NUM_RUNS=4

# CSV Filenames
Test1_Filename="complex_3_different_kernels.csv"
Test2_Filename="complex_different_sizes_kernels.csv"
Test3_Filename="complex_multi_malloc.csv"
Test4_Filename="complex_multi_stream_kernels.csv"

# Set Compiler and Flags
COMPILER="nvcc"

FLAGS="-arch=${ARCH} -std=c++20"
COMPILE="${COMPILER} ${FLAGS}"

# Utility file path
CSV_UTIL="util/csv_util.cpp"

# Command
COMMAND="${COMPILE} ${CSV_UTIL}"

echo "Compiling for ARCH="${ARCH}" target Nvidia GPU architecture"


echo "Compiling combined_3diff_kernels_singlerun.cu"
rm -f complex-3diff-kernels-test/combined_3diff_kernels_singlerun
${COMMAND} complex-3diff-kernels-test/combined_3diff_kernels_singlerun.cu -o complex-3diff-kernels-test/combined_3diff_kernels_singlerun

echo "Entering complex-kernels-test directory"
cd complex-3diff-kernels-test/ || exit 1
echo "Running combined_3diff_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_3diff_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test1_Filename}
cd ..

echo "Compiling combined_diffsize_kernels_singlerun.cu"
rm -f complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun
${COMMAND} complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun.cu -o complex-diffsize-kernels-test/combined_diffsize_kernels_singlerun

echo "Entering diffsize-kernels-test directory"
cd complex-diffsize-kernels-test/ || exit 1
echo "Running combined_diffsize_kernels_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_diffsize_kernels_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test2_Filename}
cd ..

echo "Compiling combined_multi_malloc_singlerun.cu"
rm -f complex-multi-malloc-test/combined_multi_malloc_singlerun
${COMMAND} complex-multi-malloc-test/combined_multi_malloc_singlerun.cu -o complex-multi-malloc-test/combined_multi_malloc_singlerun

echo "Entering multi-malloc-test directory"
cd complex-multi-malloc-test/ || exit 1
echo "Running combined_multi_malloc_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_malloc_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test3_Filename}
cd ..

echo "Compiling combined_multi_stream_singlerun.cu"
rm -f complex-multi-stream-test/combined_multi_stream_singlerun
${COMMAND} complex-multi-stream-test/combined_multi_stream_singlerun.cu -o complex-multi-stream-test/combined_multi_stream_singlerun

echo "Entering multi-stream-test directory"
cd multi-stream-test/ || exit 1
echo "Running combined_multi_stream_singlerun with arguments ${NSTEPS} ${SKIPBY} ${NUM_RUNS}"
./combined_multi_stream_singlerun ${NSTEPS} ${SKIPBY} ${NUM_RUNS} ${Test4_Filename}
cd ..

echo "Running generate_plots.sh with NUM_RUNS=$NUM_RUNS"
bash generate_plots.sh "${NUM_RUNS}"
# bash generate_plots_multitests.sh "${NUM_RUNS}"

echo "All steps completed successfully."
