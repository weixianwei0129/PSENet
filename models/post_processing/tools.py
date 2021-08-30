import torch
import numpy as np
from .pse import pse


# 后处理部分
def get_results(out, kernel_num, min_area):
    score = torch.sigmoid(out[:, 0, :, :])

    kernels = out[:, :kernel_num, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    score = score.data.cpu().numpy()[0].astype(np.float32)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

    label = pse(kernels, min_area)
    return score, label
