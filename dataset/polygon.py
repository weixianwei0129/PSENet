import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

from dataset.utils import shrink, random_rotate, crop_img
from dataset.utils import random_color_aug, scale_aligned_short, clip_polygon

train_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
# train_root_dir = '/Users/weixianwei/Dataset/open/MSRA-TD500/'
train_data_dir = os.path.join(train_root_dir, 'train')
train_gt_dir = os.path.join(train_root_dir, 'train')

test_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
test_data_dir = os.path.join(train_root_dir, 'test')
test_gt_dir = os.path.join(train_root_dir, 'test')

img_postfix = "JPG"
gt_postfix = "TXT"


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_ann(img, gt_path):
    """
    如果img is None, 则返回像素坐标
    """
    if img is None:
        h, w = 1.0, 1.0
    else:
        h, w = img.shape[0:2]
    lines = open(gt_path, 'r').readlines()
    text_regions = []
    words = []
    for idx, line in enumerate(lines):
        sp = line.strip().split(',')
        word = sp[-1]
        if word[0] == '#':
            words.append('###')
        else:
            words.append(word)
        location = [int(x) for x in sp[:-1]]
        location = np.array(location) / ([w * 1.0, h * 1.0] * int(len(location) / 2))
        text_regions.append(location)
    return text_regions, words


class PolygonDataSet(data.Dataset):
    def __init__(self,
                 data_type='train',
                 short_size=736,
                 kernel_num=7,
                 min_scale=0.4,
                 use_mosaic=True):
        self.data_type = data_type
        self.use_mosaic = use_mosaic
        self.short_size = short_size
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        if self.data_type == 'train':
            data_pattern = os.path.join(train_data_dir, f"*.{img_postfix}")
            gt_pattern = os.path.join(train_gt_dir, f"*.{gt_postfix}")
        elif self.data_type == 'test':
            data_pattern = os.path.join(test_data_dir, f"*.{img_postfix}")
            gt_pattern = os.path.join(test_gt_dir, f"*.{gt_postfix}")
        else:
            print('Error: data_type must be train or test!')
            raise
        self.img_paths = glob.glob(data_pattern)
        self.img_paths.sort()
        self.gt_paths = glob.glob(gt_pattern)
        self.gt_paths.sort()
        assert len(self.gt_paths) == len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        do_crop = np.random.randint(0, 10) < 3 and self.data_type == 'train'

        if self.data_type == 'train' and not do_crop and \
                np.random.uniform(0, 10) < self.use_mosaic:
            if np.random.choice([True, True, False]):
                # mosaic
                img, text_regions, words = self.mosaic(index)
            else:
                img, text_regions, words = self.mix_up(index)
        else:
            img_path = self.img_paths[index]
            gt_path = self.gt_paths[index]
            img = get_img(img_path)  # (h,w,c-rgb)
            text_regions, words = get_ann(img, gt_path)

        if self.data_type == 'train':
            img = cv2.resize(img, (self.short_size, self.short_size))
        else:
            img = scale_aligned_short(img, self.short_size)
        if do_crop:
            img, text_regions = crop_img(img, text_regions, self.short_size)
        height, width = img.shape[:2]

        # =======构建label map=======
        # 记录全部的文本区域: 有文本为[文本的序号], 没有文本为0
        gt_instance = np.zeros((height, width), dtype='uint8')
        # 记录难识别文本的区域: 难识别的文本为0, 其他为1
        training_mask = np.ones((height, width), dtype='uint8')
        selected_text_regions = []
        for idx, points in enumerate(text_regions):
            ignore = words[idx] == '###'
            if do_crop:
                # 计算被裁减掉的多边形区域的最终形状
                points = clip_polygon([points], height, width)
                if not len(points):
                    continue
            points = np.reshape(points, (-1, 2)) * np.array([width, height]).T
            points = np.int32(points)
            # # 如果文本面积很小并且的长宽太小(看不清楚),那么不要
            # area = cv2.contourArea(points)
            # shape = cv2.minAreaRect(points)[1]
            # if area < 200 and min(shape) < 8:
            #     ignore = True

            if ignore:
                cv2.fillPoly(training_mask, [points], 0)
            else:
                # 制作label
                cv2.fillPoly(gt_instance, [points], min(idx + 1, 100))
                selected_text_regions.append(points)

        text_regions = selected_text_regions

        gt_kernels = []
        for i in range(1, self.kernel_num):
            gt_kernel = np.zeros((height, width), dtype='uint8')
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            kernel_text_regions = shrink(text_regions, rate)
            for kernel_points in kernel_text_regions:
                cv2.fillPoly(gt_kernel, [kernel_points.astype(int)], 1)
            gt_kernels.append(gt_kernel)

        # =========数据增强=========
        if self.data_type == 'train':
            if np.random.uniform(0, 10) > 5:
                img = random_color_aug(img)
            matrices = [img, gt_instance, training_mask] + gt_kernels
            matrices = random_rotate(matrices)
            img, gt_instance, training_mask, gt_kernels = matrices[0], matrices[1], matrices[2], matrices[3:]

        # gt_text 不区分文本实例
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
        )
        return data

    def mix_up(self, index):
        indexes = np.random.choice(range(len(self.img_paths)), size=(2,)).tolist()
        if index not in indexes:
            indexes[0] = index
        img1 = get_img(self.img_paths[indexes[0]])
        texts_regions1, words1 = get_ann(img1, self.gt_paths[indexes[0]])
        img1 = cv2.resize(img1, (self.short_size, self.short_size))
        img2 = get_img(self.img_paths[indexes[1]])
        texts_regions2, words2 = get_ann(img2, self.gt_paths[indexes[1]])
        img2 = cv2.resize(img2, (self.short_size, self.short_size))
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        return img, texts_regions1 + texts_regions2, words1 + words2

    def mosaic(self, index):
        """
        田字格画布用a表示
        4张图片用b表示

        """
        text_regions4 = []
        words4 = []
        s = self.short_size
        xc, yc = np.random.uniform(s // 2, s * 3 // 2, size=[2, ]).astype(int)
        indices = [index] + np.random.choice(range(len(self.img_paths)), size=(3,)).tolist()
        for i, index in enumerate(indices):
            img = get_img(self.img_paths[index])
            texts_regions, words = get_ann(img, self.gt_paths[index])
            img = scale_aligned_short(img, int(self.short_size * np.random.uniform(0.9, 2)))
            h, w, c = img.shape

            if i == 0:  # a图的左上图, 图b右下角与中心对齐
                img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(0, xc - w), max(0, yc - h), xc, yc
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = w - wa, h - ha, w, h
            elif i == 1:  # a图的右上图, 图片b左下角与中心对齐
                x1a, y1a, x2a, y2a = xc, max(0, yc - h), min(s * 2, xc + w), yc
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = 0, h - ha, min(w, wa), h
            elif i == 2:  # a图的左下图, 图片b的右上角与中心对齐
                x1a, y1a, x2a, y2a = max(0, xc - w), yc, xc, min(s * 2, yc + h)
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = w - wa, 0, w, min(ha, h)
            elif i == 3:  # a图右下图, 图b的左上角与中心对齐
                x1a, y1a, x2a, y2a = xc, yc, min(s * 2, xc + w), min(s * 2, yc + h)
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = 0, 0, min(w, wa), min(h, ha)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            for points, word in zip(texts_regions, words):
                num = points.shape[0] // 2
                points = points * np.array([w, h] * num) + np.array([padw, padh] * num)
                points = points / np.array([s * 2, s * 2] * num)
                text_regions4.append(points)
                words4.append(word)

        return img4, text_regions4, words4


if __name__ == '__main__':
    import sys

    sys.path.append(os.path.dirname(__file__) + '/../')
    dataset = PolygonDataSet(data_type='train')
    print("total: ", len(dataset))
    batch_size = 2
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
    )
    for data in loader:
        for b in range(batch_size):
            img = data['imgs'][b].numpy().transpose((1, 2, 0))[..., ::-1]
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
            gt_text = (data['gt_texts'][b].numpy() * 255).astype(np.uint8)
            gt_kernels = data['gt_kernels'][b].numpy()
            train_mask = (data['training_masks'][b].numpy() * 255).astype(np.uint8)
            gt_kernels = (np.concatenate(gt_kernels, axis=0) * 255).astype(np.uint8)
            mask = np.where(gt_text[:, :, None] > 0, img, 0).astype(np.uint8)
            concat = [img, np.stack([gt_text] * 3, axis=-1), np.stack([train_mask] * 3, axis=-1), mask]
            concat = np.concatenate(concat, axis=1)
            cv2.imshow("img.jpg", concat)
            cv2.imwrite("img.jpg", concat)

            # cv2.imshow("gt_text.jpg", gt_text)
            # cv2.imshow("gt_kernels.jpg", gt_kernels)
            # cv2.imshow("mask.jpg", mask)
            cv2.waitKey(0)
