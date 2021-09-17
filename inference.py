import glob
import os

import cv2
import yaml
import torch
import numpy as np
from easydict import EasyDict
from collections import OrderedDict

from models.psenet import PSENet
from pre_process.pre import preprocess_img
from post_process.post import postprocess_bitmap
from models.post_processing.tools import get_pse_label

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def do_infer(model, img, cfg):
    model.eval()
    processed_img = preprocess_img(img)
    tensor = torch.from_numpy(processed_img).to(device)
    out = model(tensor)
    score, label = get_pse_label(out, cfg.evaluation.kernel_num, cfg.evaluation.min_area)
    return score, label


def load_model(root_path):
    cfg_path = os.path.join(root_path, "config.yaml")
    weights = os.path.join(root_path, "ckpt/best.pt")
    cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    model = PSENet(**cfg.model)

    checkpoint = torch.load(weights, map_location=torch.device(device))
    d = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        name = key[7:]  # key 中删除前七个字符
        d[name] = value
    model.load_state_dict(d)
    model = model.to(device)
    return model, cfg


def main():
    root_path = "checkpoints/v1.5.0"
    model, cfg = load_model(root_path)
    all_path = glob.glob("/Users/weixianwei/Dataset/open/MSRA-TD500/test/*.JPG")
    all_path.sort()
    for img_path in all_path:
        # Load Data
        img = cv2.imread(img_path)
        # Inference
        score, label = do_infer(model, img, cfg)
        np.save(img_path + '.npy', [score, label])

        infos = postprocess_bitmap(img, label)
        for i, info in enumerate(infos):
            cv2.imshow(str(i), info["textImg"])
        cv2.imshow('img', img)
        cv2.waitKey(0)
        for i in range(len(infos)):
            cv2.destroyWindow(str(i))


if __name__ == '__main__':
    main()
