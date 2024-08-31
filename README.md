# CustomGPUMemoryAllocator - A Learning Project

This project, CustomGPUMemoryAllocator, was created as a learning exercise to understand GPU memory management and CUDA programming concepts. It implements a custom memory allocation system for GPU operations, providing hands-on experience with low-level GPU memory handling.

## What I Learned

Through this project, I gained knowledge and practical experience in:

1. GPU Memory Management: Understanding how memory is allocated and managed on GPUs.
2. CUDA Programming: Writing CUDA kernels and integrating them with Python using libraries like CuPy.
3. Memory Allocation Algorithms: Implementing and optimizing algorithms for memory allocation and deallocation.
4. Fragmentation Handling: Techniques to deal with memory fragmentation, including defragmentation and coalescing.
5. Python-CUDA Integration: Using CuPy to interface between Python and CUDA.
6. Advanced Data Structures: Implementing efficient data structures for memory block management.
7. Performance Optimization: Techniques to optimize GPU memory operations for better performance.
8. Testing and Benchmarking: Creating comprehensive tests and benchmarks for GPU memory operations.

## Resources Used

In creating this project, I relied on several key resources:

1. CUDA Programming Guide: For understanding CUDA concepts and kernel programming.
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/

2. CuPy Documentation: To learn how to interface CUDA with Python.
   https://docs.cupy.dev/en/stable/

3. "Professional CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher: For in-depth CUDA programming concepts.

4. Various academic papers on GPU memory management, including:
   - "A Dynamic Memory Allocator for CUDA" by Xinxin Mei and Xiaowen Chu
   - "ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU" by Markus Steinberger et al.

5. Stack Overflow and NVIDIA Developer Forums: For troubleshooting and community insights.

6. Python Documentation: For general Python programming concepts.
   https://docs.python.org/3/

## Project Structure

The project is structured as follows:

- `src/`: Contains the main implementation files
  - `allocator.py`: The main CustomGPUMemoryAllocator class
  - `memory_manager.py`: Handles memory block management
  - `kernels.py`: CUDA kernels for memory operations
- `tests/`: Unit tests for the allocator
- `benchmarks/`: Performance benchmarking scripts
- `examples/`: Example usage of the allocator

## Key Takeaways

1. GPU memory management is complex and requires careful consideration of allocation patterns and fragmentation.
2. CUDA programming allows for powerful parallel computations but requires understanding of GPU architecture.
3. Efficient data structures are crucial for managing memory blocks and free lists.
4. Testing and benchmarking are essential for ensuring correctness and measuring performance improvements.
5. Integrating low-level GPU operations with high-level Python code requires careful use of libraries like CuPy.
