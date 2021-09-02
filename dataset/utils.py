import numpy as np
import imgaug.augmenters as iaa
import random
import cv2
import pyclipper

random.seed(123456)


# Random aug
def random_color_aug(image):
    """
    输入uint8数据[0,255]
    输出uint8数据[0,255]
    """
    # jpeg 图像质量
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    image = sometimes(iaa.JpegCompression(compression=(60, 80)))(image=image)
    image = sometimes(iaa.AddToHueAndSaturation((-60, 60)))(image=image)
    k = np.random.randint(3, 8)
    image = sometimes(iaa.MotionBlur(k, angle=[-90, 90]))(image=image)
    image = image.astype(np.uint8)
    return image


def random_horizontal_flip(imgs):
    """随机水平翻转"""
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    """根据图像中心随机旋转 [-10,10]度"""
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        if i == 2:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST, borderValue=1)
        else:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST, borderValue=0)
        imgs[i] = img_rotation
    return imgs


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


def resize_h(img, rh):
    h, w = img.shape[:2]
    rw = int(np.ceil((w / h) * rh / 32) * 32)
    return cv2.resize(img, (rw, rh))


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736):
    """随机更新长宽比,并调整为32的倍数"""
    h, w = img.shape[0:2]

    scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
    scale = (scale * short_size) / min(h, w)

    aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
    h_scale = scale * np.sqrt(aspect)
    w_scale = scale / np.sqrt(aspect)

    img = scale_aligned(img, h_scale, w_scale)
    return img


def random_crop_padding(imgs, target_size):
    """随机crop
    imgs: [图片, gt mask, training mask, kernels]
    """
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            if idx == 2:
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(1,))
            else:
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def random_crop(imgs, iterations=100):
    h, w = imgs[0].shape[0:2]
    th = tw = int(random.uniform(0.5, 0.9) * min(h, w))
    cnt = 0
    while cnt < iterations:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        gt_text = imgs[1][i:i + th, j:j + tw].copy()
        training_mask = imgs[2][i:i + th, j:j + tw].copy()
        valid_gt_text = gt_text * training_mask
        # 文字区域大于100个像素
        if valid_gt_text.sum() > 100:
            for idx in range(len(imgs)):
                if len(imgs[idx].shape) == 3:
                    imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
                else:
                    imgs[idx] = imgs[idx][i:i + th, j:j + tw]
            return imgs
        else:
            cnt += 1
    return imgs


# =========SHRINK=======
def shrink(text_regions, rate, max_shr=20):
    rate = rate * rate
    shrunk_text_regions = []
    for bbox in text_regions:
        area = cv2.contourArea(bbox)
        peri = cv2.arcLength(bbox, True)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrunk_bbox = pco.Execute(-offset)
            if len(shrunk_bbox) == 0:
                shrunk_text_regions.append(bbox)
                continue

            shrunk_bbox = np.array(shrunk_bbox)[0]
            if shrunk_bbox.shape[0] <= 2:
                shrunk_text_regions.append(bbox)
                continue

            shrunk_text_regions.append(shrunk_bbox)
        except Exception as e:
            print('area:', area, 'peri:', peri)
            shrunk_text_regions.append(bbox)

    return shrunk_text_regions


def crop_img(img):
    h, w = img.shape[:2]
    crop_side = int(min(h, w) * random.uniform(0.5, 0.8))
    if crop_side < 256:
        return img
    x = np.random.randint(0, w - crop_side)
    y = np.random.randint(0, h - crop_side)
    if len(img.shape) == 3:
        return img[y:y + crop_side, x:x + crop_side, :]
    else:
        return img[y:y + crop_side, x:x + crop_side]
