import cupy as cp

initialize_memory = cp.RawKernel(r'''
extern "C" __global__
void initialize_memory(void* memory, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = tid; i < size; i += stride) {
        ((char*)memory)[i] = 0;
    }
}
''', 'initialize_memory')

defragment_memory = cp.RawKernel(r'''
#include <cuda_runtime.h>

extern "C" __global__
void defragment_memory(void* memory, size_t total_size, long long* free_blocks, int num_free_blocks) {
    __shared__ size_t shared_offsets[256];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Calculate cumulative sizes
    if (tid < num_free_blocks) {
        size_t cumulative_size = 0;
        for (int i = 0; i <= tid; i++) {
            cumulative_size += free_blocks[i * 2 + 1];
        }
        shared_offsets[tid] = cumulative_size;
    }
    __syncthreads();

    // Move memory blocks
    for (int i = tid; i < total_size; i += stride) {
        int block_index = -1;
        for (int j = 0; j < num_free_blocks; j++) {
            if (i >= free_blocks[j * 2] && i < free_blocks[j * 2] + free_blocks[j * 2 + 1]) {
                block_index = j;
                break;
            }
        }
        
        if (block_index != -1) {
            size_t source = i;
            size_t dest = (block_index == 0) ? i : shared_offsets[block_index - 1] + (i - free_blocks[block_index * 2]);
            ((char*)memory)[dest] = ((char*)memory)[source];
        }
    }
}
''', 'defragment_memory')

coalesce_memory = cp.RawKernel(r'''
extern "C" __global__
void coalesce_memory(long long* free_blocks, int num_free_blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < num_free_blocks - 1) {
        long long current_end = free_blocks[tid * 2] + free_blocks[tid * 2 + 1];
        long long next_start = free_blocks[(tid + 1) * 2];
        
        if (current_end == next_start) {
            atomicAdd(&free_blocks[tid * 2 + 1], free_blocks[(tid + 1) * 2 + 1]);
            free_blocks[(tid + 1) * 2] = -1;  // Mark as merged
        }
    }
}
''', 'coalesce_memory')