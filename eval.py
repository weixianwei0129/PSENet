import glob
import os

import cv2
import yaml
import torch
import numpy as np
from easydict import EasyDict
import matplotlib.pyplot as plt
from collections import OrderedDict

from models.psenet import PSENet
from dataset.polygon import get_ann
from models.post_processing.tools import get_results

cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


def iou_single(a, b, n_class=2):
    miou = []
    for i in range(1, n_class):
        inter = ((a == i) & (b == i)).astype(float)
        union = ((a == i) | (b == i)).astype(float)
        miou.append(np.sum(inter) / (np.sum(union) + 1e-4))
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
        mean = np.reshape([0.485, 0.456, 0.406], (1, 1, 3))
    if std is None:
        std = np.reshape([0.229, 0.224, 0.225], (1, 1, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[None, ...]
    img = np.ascontiguousarray(img).astype(np.float32)
    assert len(img.shape) == 4
    img = torch.from_numpy(img)
    return img.cuda() if cuda == 'cuda' else img


@torch.no_grad()
def do_infer(model, img, cfg):
    model.eval()
    processed_img = preprocess_img(img)
    out = model(processed_img)
    score, label = get_results(out, cfg.evaluation.kernel_num, cfg.evaluation.min_area)
    height, width = img.shape[:2]
    score = cv2.resize(score.astype(np.float32), (width, height))
    label = cv2.resize(label.astype(np.float32), (width, height))
    return score, label


def load_model(root_path):
    cfg_path = os.path.join(root_path, "config.yaml")
    weights = os.path.join(root_path, "ckpt/best.pt")
    cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    model = PSENet(**cfg.model)

    checkpoint = torch.load(weights)
    d = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        name = key[7:]  # key 中删除前七个字符
        d[name] = value
    model.load_state_dict(d)
    if cuda == 'cuda':
        model = model.cuda()
    return model, cfg


def main():
    model, cfg = load_model('/data/weixianwei/models/psenet/uniform/v1.3.0/')
    all_path = glob.glob("/data/weixianwei/psenet/data/MSRA-TD500/test//*.JPG")
    prs, rcs, ious = [], [], []
    all_path.sort()
    for img_path in all_path:

        # Load Data
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        gt_path = img_path.replace('.JPG', '.TXT')
        text_regions, words = get_ann(img, gt_path)
        gt_text = np.zeros((height, width), dtype='uint8')
        for idx, points in enumerate(text_regions):
            points = np.reshape(points, (-1, 2)) * np.array([width, height]).T
            points = np.int32(points)
            text_regions[idx] = points
            cv2.fillPoly(gt_text, [points], 1)
        gt_text = gt_text.flatten().astype(int)

        # Inference
        _, label = do_infer(model, img, cfg)

        # Metric
        predict = (label > 0).astype(int).flatten()
        iou = iou_single(predict, gt_text)
        tp = np.sum(np.logical_and(predict == 1, gt_text == 1))
        fp = np.sum(np.logical_and(predict == 1, gt_text == 0))
        fn = np.sum(np.logical_and(predict == 0, gt_text == 1))

        precision = tp / (tp + fp + 1e-4)
        recall = tp / (tp + fn + 1e-4)

        if iou < 0.5:
            print(f"{img_path}>>{iou:.4f} | {words}")

            mask = np.zeros_like(img, dtype=np.uint8)
            for idx, points in enumerate(text_regions):
                cv2.fillPoly(mask, [points], (0, 255, 0))
            gt_img = np.clip(0.3 * mask + img, 0, 255).astype(np.uint8)
            predict_mask = np.zeros_like(img, dtype=np.uint8)
            predict_mask[..., 1] = (label / np.max(label) * 255).astype(np.uint8)
            gt_img = np.clip(0.3 * predict_mask + gt_img, 0, 255).astype(np.uint8)

            basename = os.path.basename(img_path)
            cv2.imwrite(basename, gt_img)

        prs.append(precision)
        rcs.append(recall)
        ious.append(iou)

    plt.subplot(221)
    plt.title('prs')
    plt.hist(prs)
    plt.subplot(222)
    plt.title('rcs')
    plt.hist(rcs)
    plt.subplot(212)
    plt.title('ious')
    plt.hist(ious)
    plt.tight_layout()
    plt.savefig('res.png')


if __name__ == '__main__':
    main()
