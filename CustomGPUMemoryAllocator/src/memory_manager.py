import numpy as np
from collections import defaultdict
from bisect import bisect_left

class MemoryBlock:
    def __init__(self, start, size, is_free=True):
        self.start = start
        self.size = size
        self.is_free = is_free

class MemoryManager:
    def __init__(self, total_size, min_block_size, max_block_size):
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.blocks = [MemoryBlock(0, total_size)]
        self.free_blocks = defaultdict(list)
        self.free_blocks[total_size].append(0)

    def allocate(self, size):
        size_class = self._get_size_class(size)
        for block_size in range(size_class, self.max_block_size + 1):
            if self.free_blocks[block_size]:
                block_start = self.free_blocks[block_size].pop()
                block = next(b for b in self.blocks if b.start == block_start)
                if block.size > size:
                    remaining_size = block.size - size
                    new_block = MemoryBlock(block.start + size, remaining_size)
                    self.blocks.insert(self.blocks.index(block) + 1, new_block)
                    self._add_to_free_blocks(new_block)
                block.size = size
                block.is_free = False
                return block.start
        return None

    def free(self, address):
        block_index = self._find_block(address)
        if block_index != -1:
            block = self.blocks[block_index]
            block.is_free = True
            self._add_to_free_blocks(block)
            self._merge_adjacent_blocks(block_index)
        else:
            raise ValueError("Invalid address")

    def get_free_blocks(self):
        return np.array([(block.start, block.size) for block in self.blocks if block.is_free])

    def update_after_defragmentation(self, free_blocks):
        self.blocks = [MemoryBlock(start, size, True) for start, size in free_blocks]
        self.free_blocks.clear()
        for block in self.blocks:
            self._add_to_free_blocks(block)

    def update_after_coalescing(self, free_blocks):
        self.blocks = [b for b in self.blocks if not b.is_free]
        self.blocks.extend([MemoryBlock(start, size, True) for start, size in free_blocks])
        self.blocks.sort(key=lambda x: x.start)
        self.free_blocks.clear()
        for block in self.blocks:
            if block.is_free:
                self._add_to_free_blocks(block)

    def _get_size_class(self, size):
        return max(self.min_block_size, 1 << (size - 1).bit_length())

    def _add_to_free_blocks(self, block):
        size_class = self._get_size_class(block.size)
        self.free_blocks[size_class].append(block.start)

    def _find_block(self, address):
        return bisect_left(self.blocks, address, key=lambda b: b.start)

    def _merge_adjacent_blocks(self, index):
        while index > 0 and self.blocks[index - 1].is_free:
            self.blocks[index - 1].size += self.blocks[index].size
            self._remove_from_free_blocks(self.blocks[index])
            del self.blocks[index]
            index -= 1
        while index < len(self.blocks) - 1 and self.blocks[index + 1].is_free:
            self.blocks[index].size += self.blocks[index + 1].size
            self._remove_from_free_blocks(self.blocks[index + 1])
            del self.blocks[index + 1]
        self._remove_from_free_blocks(self.blocks[index])
        self._add_to_free_blocks(self.blocks[index])

    def _remove_from_free_blocks(self, block):
        size_class = self._get_size_class(block.size)
        if block.start in self.free_blocks[size_class]:
            self.free_blocks[size_class].remove(block.start)