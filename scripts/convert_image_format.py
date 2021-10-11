import glob
import cv2
import os

all_src = glob.glob("/Users/weixianwei/Dataset/ymm/bankcard/text_det/card2/*.png")
for src in all_src:
    img = cv2.imread(src)
    dst = src.replace('.png', '.jpg')
    cv2.imwrite(dst, img)