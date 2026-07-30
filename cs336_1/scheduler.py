import math

def get_lr_cosine_schdule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    """
    计算带预测的余弦退火学习率

    :param int: 当前迭代次数 t
    :param max_learning_rate:  最大学习率
    :param min_learning_rate:  最小学习率
    :param warmup_iters:  预热部署 T_w
    :param cosine_cycle_iters:  总退火部署 T_c
    :return:
    """

    # 1. 预热阶段： 线性增长
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    # 2. 退火后阶段
    if it > cosine_cycle_iters:
        return min_learning_rate

    # 3. 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # 计算余弦系数，从 1.0 讲到 0.0
    # math.cos(math.pi * decay_ratio) 的范围是 [1, -1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # 最终学习率 = min + 系数 * (max - min)
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)