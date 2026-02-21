from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0 # 引用计数
        self.hash = -1 # 该块所存储 token 序列的哈希值，用于缓存匹配
        self.token_ids = [] # 该块所存储的 token 序列

    def update(self, hash: int, token_ids: list[int]): # 更新块的哈希值和 token 内容（通常在块填满时调用）
        self.hash = hash
        self.token_ids = token_ids

    def reset(self): # 重置块状态，使其变为可用状态
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager: # 负责所有 Block 的生命周期管理（分配、释放、查找）

    def __init__(self, num_blocks: int, block_size: int): 
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 预先创建 num_blocks 个 Block 对象
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值到块 ID 的映射，用于快速查找缓存匹配的块
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲块队列，用于分配新块（双端队列）
        self.used_block_ids: set[int] = set() # 已用块集合，用于快速查找已用块

    # 使用 xxhash 算法计算 token 序列的哈希
    # 当 block 满时，才会计算哈希值，用于缓存匹配
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): # prefix：前一个 block 的哈希值，用于增量哈希计算
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前一个 Block 的哈希值（整数）转换为 8 字节（64位）的二进制数据，'little' 表示小端字节序
            h.update(prefix.to_bytes(8, "little"))
        # 将 token_ids 列表转换为 numpy 数组，再转为紧凑的字节流
        h.update(np.array(token_ids).tobytes())
        # 返回一个 64 位的整数作为该 Block 的唯一指纹
        return h.intdigest()

    # 新序列分配块，返回具体的 block
    def _allocate_block(self, block_id: int) -> Block: 
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    # 释放块，将其标记为空闲，返回具体的 block
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查是否有足够的空闲块来分配给序列
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 为序列分配物理内存块（***）
    '''
    假设 block_size=16 ，有两个请求：

    1. Req A : [Token 0-31] (占 2 个块)
    2. Req B : [Token 0-15] + [Token 16'-31'] (前 16 个 token 与 A 相同)
    Req A 分配过程 :

    1. Block 0 (Token 0-15): 计算 Hash A1 -> 未命中 -> 分配物理块 #100 -> 记录 Hash A1 -> #100。
    2. Block 1 (Token 16-31): 计算 Hash A2 (基于 A1) -> 未命中 -> 分配物理块 #101 -> 记录 Hash A2 -> #101。
    Req B 分配过程 :

    1. Block 0 (Token 0-15): 计算 Hash B1 (即 A1) -> 命中 Hash A1 -> #100 -> 复用物理块 #100 (ref_count=2)。
    2. Block 1 (Token 16'-31'): 计算 Hash B2 (基于 B1) -> 未命中 (内容不同) -> 分配物理块 #102。
    '''
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks): # 遍历序列需要的逻辑块数
            token_ids = seq.block(i) # 获取第 i 个逻辑块的 token 序列
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 计算当前逻辑块的哈希值（包含前面 block 的 hash）
            block_id = self.hash_to_block_id.get(h, -1) # 检查 hash_to_block_id 中是否存在对应的物理块。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: 
                cache_miss = True # 缓存未命中
            if cache_miss:
                block_id = self.free_block_ids[0] # 从空闲块队列中获取第一个空闲块
                block = self._allocate_block(block_id) # 分配新块
            else:
                seq.num_cached_tokens += self.block_size # 增加已缓存 token 数量
                if block_id in self.used_block_ids: # 存在且内容匹配 (token_ids 一致)，则 缓存命中 (Cache Hit) 。
                    block = self.blocks[block_id]
                    block.ref_count += 1 # 增加其 ref_count
                else:
                    block = self._allocate_block(block_id) # 重新分配新块
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id # 添加映射
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): # 遍历序列持有的所有块，将引用计数减 1
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0: # 引用计数为 0 时，释放块
                self._deallocate_block(block_id) 
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool: # 检查是否有足够的空闲块来追加序列
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence): # 尝试追加序列到最后一个逻辑块
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id) # 分配新块
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
