import json
import numpy as np
import glob
import cv2
import json

all_js = glob.glob("/Users/weixianwei/Dataset/ymm/bankcard/已完成/*/*.json")

for js in all_js:
    info = json.load(open(js))
    height = info['imageHeight']
    width = info['imageWidth']
    shapes = info.get("shapes", [])
    collection = ""
    for shape in shapes:
        if shape.get('label', None) != "1":
            continue
        # [[x,y][x,y]]
        points = shape.get('points', None)
        if points is None:
            continue
        if shape.get("shape_type", '') == "rectangle":
            assert len(points) == 2
            [x1, y1], [x2, y2] = points
            points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]
        for x, y in points:
            collection += f"{int(x)},{int(y)},"
        collection += "word\n"
    with open(js.replace('.json', '.txt'), 'w') as fo:
        fo.writelines(collection)