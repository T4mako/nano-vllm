import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str # 模型路径或名称，必须是本地存在的目录
    max_num_batched_tokens: int = 16384 # 一个批次中允许处理的最大 token 总数，用于控制显存占用和计算负载
    max_num_seqs: int = 512 # 一个批次中允许并发处理的最大序列（请求）数量
    max_model_len: int = 4096 # 模型的最大上下文长度（输入+输出），会与模型自身的限制取较小值
    gpu_memory_utilization: float = 0.9 # 预留给 KV Cache 的 GPU 显存比例（0.0 - 1.0），剩余部分用于模型权重和激活值
    tensor_parallel_size: int = 1 # 张量并行度，即使用的 GPU 数量，范围通常是 1 到 8
    enforce_eager: bool = False # 是否强制使用 PyTorch Eager 模式执行，设为 True 可用于调试，但会降低性能
    hf_config: AutoConfig | None = None # HuggingFace 的配置对象，在 __post_init__ 中自动加载，无需手动指定
    eos: int = -1 # 结束符 (End of Sentence) 的 Token ID
    kvcache_block_size: int = 256 # PagedAttention 中每个 KV Cache 块的大小，通常需要是 16 或 32 的倍数
    num_kvcache_blocks: int = -1 # KV Cache 的块总数量，通常根据 gpu_memory_utilization 自动计算，-1 表示自动

    def __post_init__(self):
        assert os.path.isdir(self.model) # 确保模型路径存在且为目录
        assert self.kvcache_block_size % 256 == 0 # 确保 KV Cache 块大小是 256 的倍数（可能是为了对齐或特定硬件优化）
        assert 1 <= self.tensor_parallel_size <= 8 # 确保张量并行度在合理范围内
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 自动加载 HuggingFace 模型配置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 确保最大模型长度不超过模型本身的限制
        assert self.max_num_batched_tokens >= self.max_model_len # 确保批处理的 token 总数足以容纳至少一个最长序列
