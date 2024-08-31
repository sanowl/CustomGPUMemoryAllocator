import numpy as np
from collections import defaultdict
from bisect import bisect_left
import threading
import math

class MemoryBlock:
    __slots__ = ['start', 'size', 'is_free', 'buddy']
    def __init__(self, start, size, is_free=True, buddy=None):
        self.start = start
        self.size = size
        self.is_free = is_free
        self.buddy = buddy

class MemoryManager:
    def __init__(self, total_size, min_block_size, max_block_size):
        self.total_size = total_size
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.blocks = [MemoryBlock(0, total_size)]
        self.free_lists = [[] for _ in range(self._get_num_size_classes())]
        self.allocated_blocks = {}
        self.lock = threading.Lock()
        self._initialize_buddy_system()

    def _initialize_buddy_system(self):
        size = self.total_size
        while size >= self.min_block_size:
            self._split_block(self.blocks[0])
            size //= 2

    def _get_num_size_classes(self):
        return int(math.log2(self.max_block_size // self.min_block_size)) + 1

    def _get_size_class(self, size):
        return min(max(0, int(math.log2(size)) - int(math.log2(self.min_block_size))),
                   self._get_num_size_classes() - 1)

    def _split_block(self, block):
        if block.size <= self.min_block_size:
            return

        size_class = self._get_size_class(block.size)
        if size_class == 0:
            return

        new_size = block.size // 2
        buddy = MemoryBlock(block.start + new_size, new_size, True, block)
        block.size = new_size
        block.buddy = buddy

        self.blocks.insert(self.blocks.index(block) + 1, buddy)
        self._add_to_free_list(block)
        self._add_to_free_list(buddy)

    def _merge_buddies(self, block):
        if not block.buddy or not block.buddy.is_free:
            return block

        if block.start > block.buddy.start:
            block, block.buddy = block.buddy, block

        merged = MemoryBlock(block.start, block.size * 2, True)
        self.blocks.remove(block.buddy)
        self.blocks[self.blocks.index(block)] = merged
        
        self._remove_from_free_list(block)
        self._remove_from_free_list(block.buddy)
        self._add_to_free_list(merged)

        return self._merge_buddies(merged)

    def _add_to_free_list(self, block):
        size_class = self._get_size_class(block.size)
        self.free_lists[size_class].append(block)

    def _remove_from_free_list(self, block):
        size_class = self._get_size_class(block.size)
        self.free_lists[size_class].remove(block)

    def allocate(self, size):
        with self.lock:
            size_class = self._get_size_class(size)
            for i in range(size_class, self._get_num_size_classes()):
                if self.free_lists[i]:
                    block = self.free_lists[i].pop()
                    while block.size > size and block.size > self.min_block_size:
                        self._split_block(block)
                    block.is_free = False
                    self.allocated_blocks[block.start] = block
                    return block.start
            raise MemoryError("Not enough memory to allocate")

    def free(self, address):
        with self.lock:
            if address not in self.allocated_blocks:
                raise ValueError("Invalid address")
            block = self.allocated_blocks.pop(address)
            block.is_free = True
            merged_block = self._merge_buddies(block)
            self._add_to_free_list(merged_block)

    def get_free_blocks(self):
        return np.array([(block.start, block.size) for block in self.blocks if block.is_free])

    def defragment(self):
        with self.lock:
            self.blocks.sort(key=lambda x: x.start)
            i = 0
            while i < len(self.blocks) - 1:
                if self.blocks[i].is_free and self.blocks[i+1].is_free:
                    self.blocks[i].size += self.blocks[i+1].size
                    self._remove_from_free_list(self.blocks[i])
                    self._remove_from_free_list(self.blocks[i+1])
                    del self.blocks[i+1]
                    self._add_to_free_list(self.blocks[i])
                else:
                    i += 1

    def get_memory_usage(self):
        total = self.total_size
        used = sum(block.size for block in self.blocks if not block.is_free)
        return {
            "total": total,
            "used": used,
            "free": total - used,
            "fragmentation": 1 - (max(block.size for block in self.blocks if block.is_free) / (total - used))
        }

    def _find_block(self, address):
        return bisect_left(self.blocks, address, key=lambda b: b.start)

    def __str__(self):
        return f"MemoryManager(total={self.total_size}, used={self.get_memory_usage()['used']}, free={self.get_memory_usage()['free']})"

    def __repr__(self):
        return self.__str__()