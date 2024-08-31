import cupy as cp
import numpy as np
from .memory_manager import MemoryManager
from .kernels import initialize_memory, defragment_memory, coalesce_memory

class CustomGPUMemoryAllocator:
    def __init__(self, total_memory_size, min_block_size=256, max_block_size=1024*1024):
        self.total_memory_size = total_memory_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.memory = cp.cuda.alloc(total_memory_size)
        self.memory_manager = MemoryManager(total_memory_size, min_block_size, max_block_size)
        self.initialize_memory()

    def initialize_memory(self):
        grid_size = (self.total_memory_size + 1023) // 1024
        block_size = 1024
        initialize_memory((grid_size,), (block_size,), (self.memory, self.total_memory_size))

    def allocate(self, size):
        size = self._align_size(size)
        address = self.memory_manager.allocate(size)
        if address is None:
            self.defragment()
            address = self.memory_manager.allocate(size)
            if address is None:
                raise MemoryError("Not enough memory to allocate")
        return cp.cuda.MemoryPointer(self.memory, address)

    def free(self, ptr):
        address = ptr.ptr - self.memory.ptr
        self.memory_manager.free(address)

    def defragment(self):
        free_blocks = self.memory_manager.get_free_blocks()
        if len(free_blocks) > 1:
            grid_size = (self.total_memory_size + 1023) // 1024
            block_size = 1024
            defragment_memory((grid_size,), (block_size,), (self.memory, self.total_memory_size, 
                                                            free_blocks.astype(np.int64), len(free_blocks)))
            self.memory_manager.update_after_defragmentation(free_blocks)

    def coalesce(self):
        free_blocks = self.memory_manager.get_free_blocks()
        if len(free_blocks) > 1:
            grid_size = (len(free_blocks) + 255) // 256
            block_size = 256
            coalesce_memory((grid_size,), (block_size,), (free_blocks.astype(np.int64), len(free_blocks)))
            self.memory_manager.update_after_coalescing(free_blocks)

    def _align_size(self, size):
        return ((size + self.min_block_size - 1) // self.min_block_size) * self.min_block_size

    def __del__(self):
        if hasattr(self, 'memory'):
            self.memory.free()