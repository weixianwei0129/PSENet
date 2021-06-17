import glob
import json
import os
import sys

import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from mmcv import Config
import torchvision.transforms as transforms

from models import build_model
from models.utils import fuse_module


def scale_aligned_short(img, short_size=736):
    # print('original img_size:', img.shape)
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    # print('img_size:', img.shape)
    return img


def pre_process(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = img[:, :, [2, 1, 0]]

    img_meta = dict(
        org_img_size=[np.array(img.shape[:2])]
    )

    img = scale_aligned_short(img, 736)
    img_meta.update(dict(
        img_size=[np.array(img.shape[:2])]
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    data = dict(
        imgs=img[None, ...],
        img_metas=img_meta
    )
    return data


def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    # sys.stdout.flush()

    device = "cpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device!")

    # model
    model = build_model(cfg.model)
    if device == "cuda":
        model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))

            if device == "cuda":
                checkpoint = torch.load(args.checkpoint)
            else:
                checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model.eval()

    all_path = glob.glob("D:/wxw/PSENet/data/ICDAR2015/Challenge4/ch4_test_images/*.jpg")
    print("total: ", len(all_path))
    for idx, path in enumerate(all_path):
        data = pre_process(path)
        data.update(dict(
            cfg=cfg
        ))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        # vision
        image_data = data['imgs'].numpy()
        image_data = np.transpose(image_data[0], [1, 2, 0])
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        image_data = (image_data * 255).astype(np.uint8)
        image_data = image_data[:, :, ::-1]
        org_img_size = data['img_metas']['org_img_size'][0]
        image_data = cv2.resize(image_data, (org_img_size[1], org_img_size[0]))
        for box in outputs['bboxes']:
            box = np.reshape(box, (-1, 1, 2))
            image_data = cv2.polylines(image_data, [box], 1, (0, 0, 255))
        cv2.imshow("image", image_data)
        key = cv2.waitKey(0)
        if key == 113:
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
