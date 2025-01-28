# CUDA Graph Tests

This repository contains various CUDA Graph tests and corresponding scripts to compile, run, and generate plots from the results. CUDA Graphs help optimize performance by reducing CPU overhead and improving GPU utilization.

## Getting Started

### Prerequisites
- **CUDA Toolkit** installed (ensuring `nvcc` is in your PATH)
- A **C++ compiler** supporting CUDA (e.g., GCC)
- **Python 3** and any plotting libraries required by your Python scripts (e.g., `matplotlib`, `pandas`)

### Repository Structure (Simplified)
- **`complex-kernels-test/`**  
  Contains a CUDA source file (`combined_3diffkernels_singlerun.cu`) and outputs a CSV file (`complex_3_different_kernels.csv`).
- **`diffsize-kernels-test/`**  
  Contains a CUDA source file (`combined_diffsize_kernels_singlerun.cu`) and outputs a CSV file (`complex_different_sizes_kernels.csv`).
- **`multi-malloc-test/`**  
  Contains a CUDA source file (`combined_multi_malloc_singlerun.cu`) and outputs a CSV file (`complex_multi_malloc.csv`).
- **`multi-stream-test/`**  
  Contains a CUDA source file (`combined_multi_stream2_singlerun.cu`) and outputs a CSV file (`complex_multi_stream_kernels.csv`).
- **`plot_generator.py`**  
  A Python script that reads one or more CSV files and generates plots in a designated output directory.
- **`run_tests.sh`**  
  A bash script that compiles each test and runs them sequentially, producing CSV output files.
- **`generate_plots.sh`**  
  A bash script that executes `plot_generator.py` with the relevant CSV files and outputs the resulting plots.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/hamani115/cuda-graph-tests.git
    cd cuda-graph-tests
    ```

2. **Run the tests and generate plots**:
    ```bash
    bash run_tests.sh
    ```
   This single script will:
   - Compile each of the CUDA test programs (located in their respective directories).
   - Execute them with predefined arguments (generating CSV files).
   - Call `generate_plots.sh`, which in turn invokes `plot_generator.py` to create plots from the CSV results.

3. **Examine the results**:
   - **CSV files** are generated in each test’s directory (e.g., `complex_3_different_kernels.csv`).
   - **Plots** are saved to the `./output_plots/` folder by default.

## Customizing the Tests or Plots

- **Modifying Test Parameters**  
  Each test directory (e.g., `complex-kernels-test/`, `diffsize-kernels-test/`) has its own `.cu` file. Look for variables such as `skipBy`, `NSTEP`, `N`, etc. You can change these before recompiling.

- **Adding/Removing CSV Files**  
  If you add new tests producing additional CSV files or remove existing ones, update the `CSV_FILES` array in `generate_plots.sh` accordingly. Any CSV path listed there will be passed to `plot_generator.py`.

- **Using a Different Python Script**  
  If you have multiple Python scripts for plotting, modify `PYTHON_SCRIPT` in `generate_plots.sh` to point to the desired script. Ensure the new script accepts the same arguments (CSV paths, output directory), or adjust accordingly.

- **Changing the Output Directory**  
  By default, plots are saved to `./output_plots`. Update the `OUTPUT_DIR` variable in `generate_plots.sh` if you want a different location.

## Notes

- The `OFFLOAD_ARCH` (e.g., `sm_75`) is set in `run_tests.sh`. Update it to match your GPU architecture if needed.
- Ensure that your environment has the required Python packages to run the plotting scripts.
- For more detailed control, you may run each `.cu` file separately with `nvcc` and then manually call `plot_generator.py` with the produce CSV file.

---

*Feel free to open a GitHub issue or submit a PR if you encounter problems or wish to contribute improvements.*