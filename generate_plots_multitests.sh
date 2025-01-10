#!/bin/bash

# Ensure the script stops on error
set -e

# Paths to CSV files (modify these paths as needed)
CSV_FILES=(
    "./multitests/complex_3_different_kernels_T4.csv"
    "./multitests/complex_3_different_kernels_L4.csv"
    "./multitests/complex_3_different_kernels_AMD.csv"
    "./multitests/complex_different_sizes_kernels_T4.csv"
    "./multitests/complex_different_sizes_kernels_L4.csv"
    "./multitests/complex_different_sizes_kernels_AMD.csv"
    "./multitests/complex_multi_stream_kernels_T4.csv"
    "./multitests/complex_multi_stream_kernels_L4.csv"
    "./multitests/complex_multi_stream_kernels_AMD.csv"
    "./multitests/complex_multi_malloc_T4.csv"
    "./multitests/complex_multi_malloc_L4.csv"
    "./multitests/complex_multi_malloc_AMD.csv"
    # "/path/to/your/file2.csv"
)

OUTPUT_DIR="./output_plots_multitests"
PYTHON_SCRIPT="./plots_multitests_generator.py"

mkdir -p "$OUTPUT_DIR"

COMMAND="python3 $PYTHON_SCRIPT"

for FILE in "${CSV_FILES[@]}"; do
    if [[ -f "$FILE" ]]; then
        COMMAND="$COMMAND $FILE"
    else
        echo "Warning: File '$FILE' not found, skipping."
    fi
done

COMMAND="$COMMAND -o $OUTPUT_DIR"
echo "Executing: $COMMAND"
eval "$COMMAND"