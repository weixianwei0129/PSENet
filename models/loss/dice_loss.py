import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        """
        input: 置信度，概率值
        target: 0-1 mask
        mask: 预测错误的地方 + 内容是###的地方构成的mask

        """
        batch_size = input.size(0)

        input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)  # 论文中的公式（5）
        loss = 1 - d  # 论文中的lc，公式（6）

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss
