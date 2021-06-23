import json
import os

import numpy as np
import cv2
import glob

all_json = glob.glob("D:/dataset/pse_dataset/ymm_v1.1/*.json")


def clockwise_from_upper_left(pts):
    """clockwise_from_upper_left

    Args:
        pts: list [[x1,y1],[x2,y2]...] 4x2

    Returns:

    """
    # 先分上下
    pts.sort(key=lambda y: y[1])
    # 再分左右
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[-2:] = sorted(pts[-2:], key=lambda x: x[0], reverse=True)
    return pts


for json_file in all_json:
    try:
        print('-------------')
        print(json_file)
        info = json.load(open(json_file, encoding='utf-8'))
        shapes = info.get("shapes", [])
        pts = []
        strings = ""
        for idx, shape in enumerate(shapes):
            label = shape.get("label", None)
            if label != '1':
                continue
            points = shape.get("points", [])
            assert len(points) == 4
            points = clockwise_from_upper_left(points)
            points = np.int0(points).flatten().tolist()
            strings += ",".join([str(x) for x in points])
            strings += f",Id{str(idx + 1).zfill(2)}\n"
            pts.append(points)
        strings = strings[:-1]
        basename = os.path.basename(json_file).split('.')[0]
        with open(json_file.replace(basename + '.json', f"gt_{basename}.txt"), "w") as fo:
            fo.writelines(strings)
        # debug
        image_data = cv2.imread(json_file.replace(".json", ".jpg"))
        for points in pts:
            points = np.reshape(points, (-1, 1, 2))
            cv2.polylines(image_data, [points], 1, (0, 0, 222), 1)
        cv2.imshow("", image_data)
        key = cv2.waitKey(0)
        if key == 113:
            break
    except Exception as e:
        print("why?:", e)
