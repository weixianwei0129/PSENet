import glob
import os

import cv2
import yaml
import torch
import numpy as np
from easydict import EasyDict

from models.psenet import PSENet
from dataset.polygon import get_ann
from models.post_processing.tools import get_results


def iou_single(a, b, n_class=2):
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()

        miou.append(torch.sum(inter) / (torch.sum(union) + 1e-4))
    miou = sum(miou) / len(miou)
    return miou


def scale_aligned_short(img, short_size=736):
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
        mean = np.reshape([0.485, 0.456, 0.406], (3, 1))
    if std is None:
        std = np.reshape([0.229, 0.224, 0.225], (3, 1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    img = np.ascontiguousarray(img)
    assert len(img.shape) == 4
    return img


@torch.no_grad()
def do_infer(model, img, cfg):
    height, width = img.shape[:2]
    model.eval()
    processed_img = preprocess_img(img)
    out = model(processed_img)
    score, label = get_results(out, cfg.evaluation.kernel_num, cfg.evaluation.min_area)
    score = cv2.resize(score, (width, height))
    label = cv2.resize(label, (width, height))
    return score, label


def load_model(root_path):
    cfg_path = os.path.join(root_path, "config.yaml")
    weights = os.path.join(root_path, "last.pt")
    cfg = EasyDict(yaml.safe_load(cfg_path))

    model = PSENet(**cfg.model)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
    return model, cfg


def main():
    model, cfg = load_model('checkpoints/v1.3.0')
    all_path = glob.glob("/Users/weixianwei/Dataset/open/MSRA-TD500/test/*.JPG")
    prs = np.zeros((11,))
    rcs = np.zeros((11,))
    f1s = np.zeros((11,))
    ious = np.zeros((11,))
    for img_path in all_path:
        img = cv2.imread(img_path)
        gt_path = img_path.repalce('.JPG', '.TXT')
        text_regions, words = get_ann(img, gt_path)
        height, width = img.shape[:2]
        # =======构建gt_text=======
        # 记录全部的文本区域: 有文本为[文本的序号], 没有文本为0
        gt_instance = np.zeros((height, width), dtype='uint8')
        for idx, points in enumerate(text_regions):
            points = np.reshape(points, (-1, 2)) * np.array([width, height]).T
            points = np.int32(points)
            text_regions[idx] = points
            cv2.fillPoly(gt_instance, [points], idx + 1)
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_text = gt_text.flatten()
        # =======model infer======
        score, label = do_infer(model, img, cfg)
        for index in range(0, 11):
            threshold = index / 10
            score = score[score > threshold].astype(int).flatten()
            iou = iou_single(score, label)
            tp = np.sum(score == 1 and gt_text == 1)
            tn = np.sum(score == 0 and gt_text == 0)
            fp = np.sum(score == 0 and gt_text == 1)
            fn = np.sum(score == 1 and gt_text == 0)
            precision = tp / (tp + fp + 1e-4)
            recall = tp / (tp + fn + 1e-4)
            f1 = (2 * precision * recall) / (recall + precision)
            prs[index] += precision
            rcs[index] += recall
            f1s[index] += f1
            ious[index] += iou

    prs /= 11.0
    rcs /= 11.0
    f1s /= 11.0
    ious /= 11.0
    print(prs, rcs, f1s, ious)
