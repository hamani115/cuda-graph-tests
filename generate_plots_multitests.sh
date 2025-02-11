#!/bin/bash

# Ensure the script stops on error
set -e

# Paths to CSV files (modify these paths as needed)
CSV_FILES=(
    "./multitests/complex_3_different_kernels_Nvidia T4.csv"
    "./multitests/complex_3_different_kernels_Nvidia L4.csv"
    "./multitests/complex_3_different_kernels_AMD Radeon Pro W7800.csv"
    "./multitests/complex_different_sizes_kernels_Nvidia T4.csv"
    "./multitests/complex_different_sizes_kernels_Nvidia L4.csv"
    "./multitests/complex_different_sizes_kernels_AMD Radeon Pro W7800.csv"
    "./multitests/complex_multi_stream_kernels_Nvidia T4.csv"
    "./multitests/complex_multi_stream_kernels_Nvidia L4.csv"
    "./multitests/complex_multi_stream_kernels_AMD Radeon Pro W7800.csv"
    "./multitests/complex_multi_malloc_Nvidia T4.csv"
    "./multitests/complex_multi_malloc_Nvidia L4.csv"
    "./multitests/complex_multi_malloc_AMD Radeon Pro W7800.csv"
    # "/path/to/your/file2.csv"
)

OUTPUT_DIR="./output_plots_multitests"
PYTHON_SCRIPT="./plots_multitests_generator.py"

# Check if NUM_RUNS is provided, otherwise default to 4
if [[ -z "$1" ]]; then
    NUM_RUNS=4
    echo "No number of runs provided. Defaulting to NUM_RUNS = 4."
else
    NUM_RUNS=$1
fi

mkdir -p "$OUTPUT_DIR"

# Build a command array
COMMAND=("python3" "$PYTHON_SCRIPT")

# Append each file argument
for FILE in "${CSV_FILES[@]}"; do
    if [[ -f "$FILE" ]]; then
        COMMAND+=("$FILE")
    else
        echo "Warning: File '$FILE' not found, skipping."
    fi
done

# Append the output directory arguments
COMMAND+=("--num_runs" "$NUM_RUNS" "-o" "$OUTPUT_DIR")

# Show the exact command that will be executed
echo "Executing: ${COMMAND[@]}"

# Execute the command array
"${COMMAND[@]}"

# mkdir -p "$OUTPUT_DIR"

# COMMAND="python3 $PYTHON_SCRIPT"

# for FILE in "${CSV_FILES[@]}"; do
#     if [[ -f "$FILE" ]]; then
#         COMMAND="$COMMAND $FILE"
#     else
#         echo "Warning: File '$FILE' not found, skipping."
#     fi
# done

# COMMAND="$COMMAND -o $OUTPUT_DIR"
# echo "Executing: $COMMAND"
# eval "$COMMAND"