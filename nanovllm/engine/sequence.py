from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum): # 枚举类，用于标识序列的状态
    WAITING = auto() # auto 自动分配枚举值
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256 # 分页内存管理的块大小（每个块存储 256 个 Token）
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = [] # 逻辑块表，用于映射到物理内存块（这里初始化为空，通常由调度器填充）
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property # 装饰器，将方法转换为属性，调用时无需加括号
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self): # 完成 token 数量 = 总 token 数量 - 提示 token 数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): # 获取 prompt token 列表
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): # 获取 completion token 列表
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 已缓存的块数量 = 已缓存 token 数量 // 块大小
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): #当前序列占用的逻辑块数量 = (总 token 数量 + 块大小 - 1) // 块大小
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self): # 最后一个块的 token 数量 = 总 token 数量 - (总块数量 - 1) * 块大小
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 第 i 个逻辑块中的所有 Token ID
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int): # 推理阶段追加一个 token 到序列中
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    '''
    seq 状态序列化，用于保存和恢复序列状态
    如果序列已经开始生成（ num_completion_tokens > 0 ），它不会传输完整的 token_ids 列表，而是只传输 last_token
    '''
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    '''
    seq 状态反序列化，用于恢复序列状态
    恢复状态时，如果是生成阶段，它只恢复 last_token
    '''
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
