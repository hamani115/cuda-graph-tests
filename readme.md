# CUDA Graph Tests

This repository contains tests and examples for CUDA Graphs. CUDA Graphs provide a mechanism to capture and replay a sequence of GPU operations, which can help optimize performance by reducing CPU overhead and improving GPU utilization.

<!-- ## Directory Structure

- `src/`: Contains the source code for the CUDA Graph tests.
- `include/`: Header files used in the tests.
- `data/`: Sample data files used for testing.
- `scripts/`: Utility scripts for setting up and running tests.
- `docs/`: Documentation related to the CUDA Graph tests. -->

## Getting Started

### Prerequisites

- CUDA Toolkit installed
- C++ compiler (e.g., GCC)
<!-- - CMake for building the project -->

### Building the Project

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/cuda-graph-tests.git
    cd cuda-graph-tests
    ```

2. Navigate to one of the test directories
    ```sh
    cd multi-kernel-test
    ```
    Available tests are `memcpy-device-test`, `memcpy-host-test`, and `multi-kernel-test`.

3. Compile the files with prefered naming using `nvcc` compiler:
    ```sh
    nvcc graph_matrix_multiply.cu -o graph_matrix_multiply
    ```

4. Run the compiled file and check the result output:
    ```sh
    ./graph_matrix multiply
    ```

### Code Modification for Testing

1. memcpy-device-test
    - modify `skipBy` value to change the number graph lauches to skip at the start (`default` 100 skips)
    - 
2. memcpy-host-test
    - modify `skipBy` value to change the number graph lauches to skip at the start (`default` 100 skips)
    - 
3. multi-kernel-test
    - modify `skipBy` value to change the number graph lauches to skip at the start (`default` 0 skips)
    - 

<!-- 2. Create a build directory and navigate into it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make
    ``` -->

<!-- ### Running Tests

After building the project, you can run the tests using the following command:
```sh
./run_tests
``` -->

<!-- ## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. -->

<!-- ## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->

<!-- ## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainer at your.email@example.com. -->
