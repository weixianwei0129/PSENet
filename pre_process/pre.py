import cv2
import numpy as np


def scale_aligned_short(img, short_size):
    """根据短边进行resize,并调整为32的倍数"""
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def preprocess_img(img, short_size=736, mean=None, std=None):
    """图像预处理

    Args:
        img:   (h, w, c) BGR img
        short_size: size of resize
        mean:  (3, 1) channel means
        std:   (3, 1) channel std

    Returns:

    """
    img = scale_aligned_short(img, short_size=short_size)
    if mean is None:
        mean = np.reshape([0.485, 0.456, 0.406], (1, 1, 3))
    if std is None:
        std = np.reshape([0.229, 0.224, 0.225], (1, 1, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    img = np.ascontiguousarray(img).astype(np.float32)
    assert len(img.shape) == 4
    return img
