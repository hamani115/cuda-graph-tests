#!/bin/bash

# Ensure the script stops on error
set -e

# Paths to CSV files (modify these paths as needed)
CSV_FILES=(
    "./complex-3diff-kernels-test/complex_3_different_kernels_Nvidia.csv"
    "./complex-diffsize-kernels-test/complex_different_sizes_kernels_Nvidia.csv"
    "./complex-multi-stream-test/complex_multi_stream_kernels_Nvidia.csv"
    "./complex-multi-malloc-test/complex_multi_malloc_Nvidia.csv"
    # "/path/to/your/file2.csv"
)

# Output directory for plots
OUTPUT_DIR="./output_plots_Nvidia"
# OUTPUT_DIR="./output_plots_combined"

# Path to the Python script
PYTHON_SCRIPT="./plots_generator.py"
# PYTHON_SCRIPT="./plot_combined_generator.py"

# Check if NUM_RUNS is provided, otherwise default to 4
if [[ -z "$1" ]]; then
    NUM_RUNS=4
    echo "No number of runs provided. Defaulting to NUM_RUNS = 4."
else
    NUM_RUNS=$1
fi


# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build the command
COMMAND="python3 $PYTHON_SCRIPT"

# Add CSV file paths to the command
for FILE in "${CSV_FILES[@]}"; do
    if [[ -f "$FILE" ]]; then
        COMMAND="$COMMAND $FILE"
    else
        echo "Warning: File '$FILE' not found, skipping."
    fi
done

# Add the output directory argument
COMMAND="$COMMAND --num_runs $NUM_RUNS -o $OUTPUT_DIR"

# Print the command for debugging purposes
echo "Executing: $COMMAND"

# Execute the command
eval "$COMMAND"

# Success message
echo "Plots generated successfully in directory: $OUTPUT_DIR"