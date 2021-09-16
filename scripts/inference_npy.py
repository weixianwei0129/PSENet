import cv2
import glob
import numpy as np
from tqdm import tqdm
from post_process.post import postprocess_bitmap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Times New Roman']


def main():
    all_path = glob.glob("/Users/weixianwei/Desktop/test/*.JPG")
    all_path.sort()
    for index, img_path in tqdm(enumerate(all_path)):
        if index < 0:
            continue
        # print(index)
        # Load Data
        img = cv2.imread(img_path)
        # Inference
        score, label = np.load(img_path + '.npy', allow_pickle=True)
        label = label.astype(int)

        h, w = img.shape[:2]
        mask = (label / np.max(label) * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h))
        zeros = np.zeros_like(mask, dtype=np.uint8)
        mask = np.stack([mask, mask, zeros], axis=-1)
        img = np.clip(mask * .5 + img * .5, 0, 255).astype(np.uint8)

        infos = postprocess_bitmap(img, label)

        count = len(infos)
        if not count:
            continue
        fig = plt.figure()
        gs = GridSpec(nrows=count, ncols=2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(img)
        for i in range(count):
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(infos[i].get('textImg', img))
        plt.savefig(img_path + '_res.png')
        del fig
        # plt.show()
        # plt.pause(0)


if __name__ == '__main__':
    main()
