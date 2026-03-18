from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

'''
Scheduler 负责决定哪些 Sequence 应该被推理
- 时间上，Scheduler 通过一定的策略决定哪一个 Sequence 应当优先被推理
- 空间上，Scheduler 通过 BlockManager 完成每一个 Sequence 的 KV Cache 分配、销毁。
'''
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 分页内存管理
        self.waiting: deque[Sequence] = deque() # 等待队列
        self.running: deque[Sequence] = deque() # 运行队列

    # 等待队列和运行队列都为空时，推理完成
    def is_finished(self):
        return not self.waiting and not self.running

    # 添加一个 Sequence 到等待队列
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    '''
    - 接收到新请求后，将其加入等待队列。
    - 执行 step，分为 prefill 和 decode 两个阶段。
    - 即将执行的请求分配 blocks，信息记录在 block table 中。
    - 打包当前需要执行的请求数组 seqs。
    - 前向计算完成后进行后处理，释放已结束请求占用的 KV cache 资源。
    - 将 token ids 返回给 tokenizer 解码。
    '''
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 从 **等待队列** 头部取出 Sequence 到运行队列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            # 分配 KV Cache 并更新 Sequence 状态
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            # 更新等待队列和运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 从 **运行队列** 头部取出 Sequence 到运行队列
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 若 block 不够，触发抢占 preempt ：把某个 running 序列挪回 waiting 并释放其 block
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮调度结果按原顺序放回 running 头部
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占一个 Sequence，将其状态设置为 WAITING，并将其从运行队列中移除，加入等待队列头部
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 后处理 Sequence，将其状态设置为 FINISHED，同时从运行队列中移除
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
