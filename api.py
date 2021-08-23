import glob
import os
import time
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from mmcv import Config
import torchvision.transforms as transforms

from models import build_model
from models.utils import fuse_module
from post_process.utils import split_and_merge, draw_info

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device!")

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


def get_model(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))

    # model
    model = build_model(cfg.model)
    if DEVICE == "cuda":
        model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))

            if DEVICE == "cuda":
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
    return cfg, model


def inference(path, cfg, model):
    data = pre_process(path)
    if DEVICE == "cuda":
        data['imgs'] = data['imgs'].cuda()
    data.update(dict(
        cfg=cfg
    ))
    # forward
    with torch.no_grad():
        outputs = model(**data)

    # vision
    if DEVICE == "cuda":
        data['imgs'] = data['imgs'].cpu()
    image_data = data['imgs'].numpy()
    image_data = np.transpose(image_data[0], [1, 2, 0])
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    image_data = (image_data * 255).astype(np.uint8)
    image_data = image_data[:, :, ::-1]
    org_img_size = data['img_metas']['org_img_size'][0]
    image_data = cv2.resize(image_data, (org_img_size[1], org_img_size[0]))
    return image_data, outputs['bboxes']


def main(args):
    # load model
    cfg, model = get_model(args)
    # all_path = glob.glob("D:/dataset/pse_dataset/test_data/*.jpg")
    all_path = glob.glob("/data/weixianwei/psenet/test_data/*.jpg")
    # do infer
    print("total: ", len(all_path))
    all_path.sort()
    total_time = 0
    for idx, path in enumerate(all_path):
        basename = os.path.basename(path)
        print("============")
        t1 = time.time()
        # infer model
        image_data, bboxes = inference(path, cfg, model)
        # post process
        all_text = split_and_merge(bboxes)
        print(f"cost Time is {time.time() - t1}")
        total_time += time.time() - t1
        # vision
        image_data_show = image_data.copy()
        for box in bboxes:
            box = np.reshape(box, (-1, 1, 2))
            cv2.polylines(image_data_show, [box], True, (0, 0, 222), 1)
        cv2.imwrite(f"tmp/{basename+'_model.jpg'}", image_data_show)

        # for box in all_text:
        #     draw_info(image_data, None, box)
        # cv2.imwrite(f"tmp/{basename+'_pst.jpg'}", image_data)
        # cv2.imshow("im", image_data)
        # cv2.imwrite(f"{idx}.jpg", image_data)
        # key = cv2.waitKey(0)
        # if key == 113:
        #     exit()

        # for box in outputs:
        #     box = np.reshape(box, (-1, 1, 2))
        #     image_data = cv2.polylines(image_data, [box], 1, (0, 0, 255))
        # cv2.imshow("image", image_data)
        # key = cv2.waitKey(0)
        # if key == 113:
        #     exit()
    print(f"mean time is {total_time/(1e-4+len(all_path))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument(
        '--config',
        type=str,
        help='config file path',
        default="config/psenet/psenet_r50_ic15_736.py"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        # default="checkpoints/v1.1_600ep.pth.tar",
        default="/data/weixianwei/psenet/train_models/psenet_r50_ic15_736_v1.1/checkpoint_600ep.pth.tar",

    )
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
