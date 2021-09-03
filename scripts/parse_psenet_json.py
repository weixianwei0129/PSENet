import json
import glob
import numpy as np
import cv2

all_json = glob.glob("/Users/weixianwei/Dataset/open/MSRA-TD500/train/*.json")
all_json.sort()
for jsP in all_json:
    info = json.load(open(jsP))
    shapes = info['shapes']
    boxes = [np.array(shape.get('points', [])).reshape(-1).astype(np.int32).flatten().tolist() for shape in shapes]
    strings = ""
    for idx, box in enumerate(boxes):
        strings += ','.join([str(x) for x in box])
        strings += f",ID{idx}\n"
    open(jsP.replace(".json", ".TXT"), 'w').writelines(strings)

    # # debug
    # imP = jsP.replace('.json', '.JPG')
    # img = cv2.imread(imP)
    # for box in boxes:
    #     box = np.reshape(box, (-1, 2))
    #     cv2.polylines(img, [box], 1, (0, 0, 222), 1)
    # cv2.imshow('img.jpg', img)
    # cv2.waitKey(0)
