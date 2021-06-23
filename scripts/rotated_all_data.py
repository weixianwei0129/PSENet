import cv2
import glob
import os
import shutil
import json

import json
import cv2
import glob
import numpy as np
from configs import config
from tqdm import tqdm


def rotate_image_90(image_data, cls):
    """旋转中心坐标，逆时针旋转"""
    if cls == 0:
        pass
    elif cls == 1:
        image_data = np.rot90(image_data)
        image_data = np.flip(image_data, axis=1)
        image_data = np.flip(image_data, axis=0)
    elif cls == 2:
        image_data = np.flip(image_data, axis=0)
    else:
        image_data = np.rot90(image_data)
    return image_data


def parse_json(json_path):
    """解析json文件，获取旋转角度，其中逆时针为正，顺时针为负"""
    info = json.load(open(json_path, encoding='utf-8'))
    points = info.get("shapes", [{}])[0].get("points", [])
    if not points:
        return None
    left_x, left_y = points[0]
    right_x, right_y = points[1]
    width = abs(left_x - right_x)
    height = abs(left_y - right_y)
    if width > height:  # 0 or 180
        if left_x > right_x:
            angle = 180
            cls = 2
        else:
            angle = 0
            cls = 0
    else:  # 90 or -90
        if left_y > right_y:
            angle = 90
            cls = 1
        else:
            angle = -90
            cls = 3

    return cls, angle, [left_x, left_y, right_x, right_y]


if __name__ == '__main__':
    all_json_path = glob.glob("D:/dataset/text_images_copy/*/*/*.json")
    all_json_path.sort()
    for json_path in tqdm(all_json_path):
        try:
            cls, _, [left_x, left_y, right_x, right_y] = parse_json(json_path)
            left_x, left_y, right_x, right_y = [int(x) for x in [left_x, left_y, right_x, right_y]]
            image_path = json_path.replace(".json", ".jpg")
            image_data = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # cv2.putText(image_data, '1', (left_x, left_y), 1, 1, (0, 0, 222), 1)
            # cv2.putText(image_data, '2', (right_x, right_y), 1, 1, (0, 0, 222), 1)
            # cv2.imshow('original', image_data)

            image_data = rotate_image_90(image_data, cls)
            new_path = image_path.replace("text_images_copy", "text_images_rotated")
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path))
            # if os.path.exists(new_path):
            #     print("Exist!=>", new_path)
            #     continue
            cv2.imwrite(new_path, image_data)
            # print("cls: ", cls)
            # cv2.imshow("image_data", image_data)
            # cv2.waitKey(0)
        except Exception as e:
            print(json_path)
            print(f"why? {e}")
