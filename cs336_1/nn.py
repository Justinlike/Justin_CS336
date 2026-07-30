import torch
import torch.nn as nn
import math

from anyio import sleep_until
from functorch.einops import rearrange
from numba.cuda.cudadrv.driver import device_memory_depends
from torch.nn import factory_kwargs

"""
full connection layer
    1. init: define a matrix of W
    2. forward: matrix multiplay of W and input x
"""

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 1. define weights of W (shape: out * in)
        # create it with correct device and device
        factory_kwargs = {"device": device, "dtype": dtype}

        # nn.Parameter 的作用是告诉 Pytorch “这个张量是模型的一部分，它需要通过训练来学习的权重 (Weights)”
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 2. initialize weights (notmalization normally use xavier initialization) Xavier initialize
        # sigma^2 = 2 / (d_in + d_out)
        std = (2.0 / (in_features + out_features)) ** 0.5
        # PDF 要求截断在 [-3 sigma, 3 sigma]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a = -3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 einsum 处理，适应各种 Batch 维度情况
        # "...i" means the shape of x, and the las dim is in_feature
        # "oi" means the shape of weight, (out_features * in_features)
        # "...o" means the shape of output, and the last dim is out_features
        return torch.einsum("...i, oi -> ...o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a = -3*std, b = 3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]

def silu_fn(in_features):

    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        # W1 and W2 是并行升维层 d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # W2 是降维层: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        # shape: [..., d_ff]

        return self.w2(gate * signal)

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # 1. Initialize learnable parameter
        # weight(gamma): 缩放参数 必须初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        # bias(beta): 平移参数 必须初始化为全 0, LayerNorm 特有
        self.bias = nn.Parameter(torch.zeros(d_model, **factory_kwargs))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, d_model]
        in_dtype = x.dtype

        # 2. 转换为 float32 以确保计算均值和方差时的数值稳定性(防止溢出)
        x_float = x.to(torch.float32)

        # 3. 计算均值
        mean = x_float.mean(-1, keepdim = True)

        # 4. 计算标准差
        std = x_float.std(-1, keepdim = True)

        # 5. 归一化
        x_normed = (x_float - mean) / (std + self.eps)

        # 6. 应用可学习的增益 weight 和偏置 bias
        result = x_normed * self.weight + self.bias

        # 7. 转换回输入的类型
        return result.to(in_dtype)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # 1. Initialize learnable parameter
        # weight(gamma): 缩放参数 必须初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, d_model]
        in_dtype = x.dtype

        # 2. 转换为 float32 以确保计算均值和方差时的数值稳定性(防止溢出)
        x_float = x.to(torch.float32)

        # 3. 计算均方根 (Root Mean Square)
        # 公式: rms = sqrt(mean(x^2) + eps)
        # dim=-1 表示在隐藏层维度计算，keepdim=True 方便后续除法自动广播
        ms = x_float.pow(2).mean(-1, keepdim = True)
        rms = torch.sqrt(ms + self.eps)

        # 4. 应用可学习的增益 weight
        result = (x_float / rms) * self.weight

        # 5. 转换回输入的类型
        return result.to(in_dtype)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        initialize RoPE block

        :param theta: standard frequency (default 10000)
        :param d_k: dimension of Head (oven)
        :param max_seq_len: maximun sequence length
        """

        super().__init__()
        self.d_k = d_k

        # 1. 计算频率 Omega_k = theta^(-2k / d)
        # 我们只需要计算 d_k/2 个频率，因为旋转是承兑进行的
        # arange(0, d_k, 2) 产生 [0, 2, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers) # shape (d_k/2, )

        # 2. 创建位置序列 [0, 1, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, device=device).float() # shape (max_seq_len, )

        # 3. 计算所有位置的所有角度 (外积) this is theta_{i,k}
        # freqs_matrix shape: (max_seq_len, d_k/2)
        freqs_matrix = torch.outer(t, freqs)

        # 4. 预计算 cos 和 sin 并作为 buffer 注册
        # 使用 persistent=False 确保这些缓存不会保在 stat_dict 中 (因为可以随时生成)
        # cos_cached means cos(theta_{i,k}), and analogously for sin.
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. 提取 cos/sin (..., seq, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐
        # 只有当 x 是 4D (Batch, Head, S, d_k) 且 cos 是 3D (含Batch维度) 时，才需要手动插入 Head
        # 对于 test_rope 这种 3D x 2D cos 的情况，Pytorch 会自动左侧补1，无需操作
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一直
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分并旋转
        # ...（ellipsis）：代表任意数量的前置维度，保持不变。
        # 不管张量是 1D、2D、3D 或更多维，... 会把除了最后一维以外的所有维都保留原样。
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * cos + x_odd * sin

        return output

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 1. 数值稳定 减去指定维度的最大值
    # dim=-1 通常是 Transformer 中指定隐藏层或词表维度
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    # 2. 计算指数
    exp_x = torch.exp(x_stable)

    # 3. 计算分母的各指数之和
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # 4. 计算最终结果
    return exp_x / sum_exp

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    :param Q: (Batch, ..., n, d_k)
    :param K: (Batch, ..., m, d_k)
    :param V: (Batch, ..., m, d_v)
    :param mask: (..., n, m) 或者是可以广播到这个形状的布尔张量
    :return: 下一个 ID 的加权V的概率分布 (..., n, d_v)
    e.g. softamx 结果为 tensor([0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337])

    """

    d_k = Q.size(-1)

    # 1. 计算分数：Q @ K^T / sqrt(d_k)
    # 变换最后两个维度进行矩阵乘法
    # 形状变化: (..., n, d_k) @ (..., m, d_k)^T -> (..., n, m)
    scores = torch.einsum(",,,nk, ...mk, ...nm", Q, K) / math.sqrt(d_k)

    # 2. 应用掩码 mask
    if mask is not None:
        # PDF 要求：把 mask 为 False 的地方填入 -inf
        # 注意：使用一个足够小的负数，通常 float("-inf") 在 torch 中是安全的
        scores = scores.masked_fill(mask == False, float('-inf'))

    # 3. Softamx 归一化 (在最后一个维度上)
    probs = softmax(scores, dim=-1)

    # 4. 对 Value 加权求和
    # 形状变化 (..., n, m) @ (..., m, d_v) _> (..., n, d_v)
    output = torch.einsum("...nm, ...mk, ...nk", probs, V)

    return output

class CasualSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True,
                 max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 1. 定义 Q, K, V 的投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 2. 定义输出投影
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 3. 实例化 RoPE
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b, s, d = x.shape

        # step 1&2: project and divide head
        # q = self.q_proj(x).view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        q = rearrange(self.q_proj(x), "... s (h, q) -> ... h s q", h = self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h, q) -> ... h s q", h = self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h, q) -> ... h s q", h = self.num_heads)

        # step 3: utilize RoPE
        # 只有当模块存在时才应用
        if self.rope is not None:
            # 如果没传位置，且 RoPE 需要位置，则生成默认位置
            if token_positions is None:
                # 适配各种 Batch 维度，使用 expand 比 repeat 更高效
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(s, device=x.device).expand(*batch_dims, s)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # step 4: 生成下三角矩阵 因果掩码
        mask = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        # step 5: SDPA (SDPA 内部应能处理 mask 为 None 的情况)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # step 6&7: 合并与输出投影
        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")
        return self.output_proj(attn_out)


import torch
import torch.nn as nn
# from .nn import Embedding, RMSNorm, Linear, CasualSelfAttention, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 theta: float, device=None, dtype=None,
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",    # options: "pre", "post"
                 ffn_type: str = "swiglu",  # options: "swiglu", "ffn"
                 ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        # 1. 初始化 Attention
        # RoPE 的开关由外部传入的 theta 是否为 None 控制
        self.attn = CasualSelfAttention(
            d_model = d_model,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            theta = theta,
            device = device,
            dtype = dtype,
        )

        # 2. 初始化 Norm 层 (Abalation 1)
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            # 禁用 Norm，使用 nn.Identity 占位，直接返回输入，不做任何改变
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # 初始化 FFN (Abalation 4)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # 标准 FFN: x -> Linear -> SiLU -> Linear -> out
            # notice: 为了公平对比，通常 SiLU FFN 的 d_ff 应该是 4 * d_model
            # 这里使用传入的 d_ff
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.SiLU(),
                Linear(d_ff, d_model, device=device, dtype=dtype),
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Pre-Norm (Llama default, and is baseline of the assignment)
        # formula: x = x + SubLayer(Norm(x))
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))

        # Post-Norm (ori transformer, Ablation 2)
        # formula: x = Norm(x + SubLayer(x))
        elif self.norm_mode == "post":
            x = x + self.attn(x, token_positions=token_positions)
            x = self.ln2(x + self.ffn(x))

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None, dtype=None,
                 # 实验参数
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",    # options: "pre", "post"
                 ffn_type: str = "swiglu",  # options: "swiglu", "silu"
                ):
        super().__init__()
        self.context_length = context_length

        # 1. token embedding
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 2. stack Transformer Blocks
        # 将实验参数传入每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta,
                device=device, dtype=dtype,
                use_rms_norm=use_rms_norm,
                norm_mode=norm_mode,
                ffn_type=ffn_type
            )
            for _ in range(num_layers)
        ])

        # 3. 最终的输出层
        # 如果全局禁用了 Norm，这里的 Final Norm 也要变成 Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()

        # 最后用一个 Linear 层映射回词表大小 (LM Head)
        self.lm_head = nn.Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape

        # 准备位置信息用于 RoPE
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expend(b, s)

        # 1. Embedding
        x = self.token_embeddings(token_ids)

        # 2. 逐层通过 Transformer
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 3. 最终归一化  (如果 use_rms_norm = False, 这里就是直通)
        x = self.ln_final(x)

        # 4. 投影到词表空间得到 logits
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            eos_token_id: int,
            temperature: float = 1.0,
            top_p: float = 1.0
    ):
        """
        从模型生成文本 ID 序列

        :param prompt: 提示词 ID （Batch， Seq_len）
        :param max_new_tokens: 最多生成的词数
        :param eos_token_id: 停止生成的 Token ID （如 <|endoftext|>）
        :param temperature: 温度系数 (越高月随机，越低越确定)
        :param top_p:  核采样阈值
        :return:
        """

        # 设置为评估模式
        self.eval()

        # 将输入拷贝一份，避免修改原始数据
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # 1. 裁剪输入：模型智能处理 context_length 长度的内容
            # 如果提示词序列过长，只取最后的 context_length 个词
            idx_cond = generated[:, -self.context_length:]

            # 2. 前向传播得到 Logits
            # 我们只关心最后一个时间步的预测
            logits = self.forward(idx_cond) # (Batch, T, Vocab)
            logits = logits[:, -1, :]       # (Batch, Vocab)

            # 3. 应用温度 (Temperature)
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8) # (Batch, T, Vocab)

            # 4. 应用 Top-P (Nucleus Sampling) 过滤
            if top_p < 1.0:
                logits = self._top_p


    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数，执行 Top-P 截断"""
        # 对词表分值进行降序排序, sorted_indices 是排列后对应原始的索引
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 计算累计概率分布 cumsum是累计求和函数
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # 创建掩码，我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True (即需要过滤)
        sorted_indices_to_remove = cumulative_probs > p

        # 关键修正：确保至少保留第一个词 （最高概率词）
        # 并且我们要保留第一个 “使概率超过p” 的那个词
        # 做法就是把标记为向右移动一位
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始此表索引位置
        # 将 sorted_indices_to_remove 的值按照 sorted_indices 的索引写入 sorted_indices_to_remove的dim=1的位置
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits

