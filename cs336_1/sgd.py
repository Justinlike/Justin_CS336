import torch
import math
from collections.abc import Callable
from typing import Optional

from anyio import sleep_until


# 1. 按照 PDF 实现带有衰减逻辑的 SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """

        :param params: 传入模型需要优化的参数，通常是 model.parameters()。
        :param lr: defaults: 一个字典，存储默认超参数（如学习率 lr）
        调用 super().__init__ 后，PyTorch 会将参数组织在 self.param_groups 中
        """

        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 获取该参数对应的状态字典 (用于记录步数t)
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0

                t = state["t"]
                grad = p.grad.data

                # 执行 PDF 中的更新公式 p = p - (lr / sqrt(t+1)) * grad
                p.data -= (lr / math.sqrt(t+1)) * grad

                state["t"] += 1

        return loss

# 2. 运行学习率调试实验 (Problem learning_rate tuning)
def run_experiment(learning_rate: float):
    print(f"test learning rate {learning_rate}")

    # 初始化参数 (10 x 10)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=learning_rate)

    for t in range(10):
        opt.zero_grad()
        # 目标函数 loss = weights^2 的平均值 (极小值点应在全 0 处)
        loss = (weights ** 2).mean()
        print(f"Iter {t}: loss = {loss.item():.4f}")

        loss.backward()
        opt.step()

# 测试不同学习率
lrs_to_test = [1e1, 1e2, 1e3, 1e4]

for lr in lrs_to_test:
    run_experiment(lr)

"""
test learning rate 10.0
Iter 0: loss = 23.9290
Iter 1: loss = 15.3146
Iter 2: loss = 11.2892
Iter 3: loss = 8.8326
Iter 4: loss = 7.1544
Iter 5: loss = 5.9318
Iter 6: loss = 5.0027
Iter 7: loss = 4.2750
Iter 8: loss = 3.6918
Iter 9: loss = 3.2159
test learning rate 100.0
Iter 0: loss = 26.9829
Iter 1: loss = 26.9829
Iter 2: loss = 4.6295
Iter 3: loss = 0.1108
Iter 4: loss = 0.0000
Iter 5: loss = 0.0000
Iter 6: loss = 0.0000
Iter 7: loss = 0.0000
Iter 8: loss = 0.0000
Iter 9: loss = 0.0000
test learning rate 1000.0
Iter 0: loss = 28.5757
Iter 1: loss = 10315.8301
Iter 2: loss = 1781706.2500
Iter 3: loss = 198195808.0000
Iter 4: loss = 16053858304.0000
Iter 5: loss = 1013182300160.0000
Iter 6: loss = 52013467435008.0000
Iter 7: loss = 2237841975279616.0000
Iter 8: loss = 82482080960741376.0000
Iter 9: loss = 2648591295039143936.0000
test learning rate 10000.0
Iter 0: loss = 23.4684
Iter 1: loss = 929373.7500
Iter 2: loss = 18325536768.0000
Iter 3: loss = 240126705795072.0000
Iter 4: loss = 2353481824388251648.0000
Iter 5: loss = 18409203193115698003968.0000
Iter 6: loss = 119740213859178281781166080.0000
Iter 7: loss = 666246499418131003698592088064.0000
Iter 8: loss = 3237677120760991615119228292562944.0000
Iter 9: loss = inf
"""
