import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    calculate cross entropy loss

    :param logits: 预测的分数，形状为(..., vocab_size)
    :param targets: 目标 ID，形状为(...)
    :return:
        平均损失标量
    """

    # 1. 提取维度信息
    # 假设最后一维是 vocab_size, 前面所有的维度都是 batch-like 维度
    # 在transformer训练中，输入维度通常是 (Batch, Seq_len, Vocab_size)

    vocab_size = logits.size(-1)

    # 2. 数值稳定
    # dim=-1. 表示在词表维度中找最大，keepdim=True 方便广播减法 [1, 2, 3] -> 3 -> [3]
    # m (Batch, Seq_len, 1)
    m = torch.max(logits, dim=-1, keepdim=True).values

    # 3. 提取目标位置的 Logits (o_j)
    # 使用gather函数从logits中根据targets提取对应的分值
    # logits: (Batch, Seq, Vocab) -> targets: (Batch, Seq)
    # 需要将 targets 升维为 (Batch, Seq, 1) 才能使用 gather

    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 4. 计算 Log-Sum-Exp
    # 使用公式 M + log(sum(exp(o-M)))
    # 注意：为了防止sum结果为 0，导致log(-inf)，这里的减法已经保证 exp 的最大值是 e^0-1
    # shift_logits (Batch, Seq, Vocab)
    shift_logits = logits - m
    # Log_Sum_Exp (Batch, Seq)
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shift_logits), dim=-1))

    # 5. 计算单个 Token 的损失 log_sum_exp-o_j
    loss = log_sum_exp - target_logits

    # 6. 返回整个Batch的平均值
    return torch.mean(loss)