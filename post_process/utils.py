import os
import cv2
import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np


class BOX(object):
    def __init__(self, box):
        """
        box: 轮廓点
        """
        box = np.reshape(box, (-1, 1, 2))
        rect = cv2.minAreaRect(box)
        self.center, _, _ = rect
        box = np.int0(cv2.boxPoints(rect))
        x1, y1, x2, y2, x3, y3, x4, y4 = box.flatten().tolist()
        box = [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]
        self.box = np.reshape(clockwise_from_upper_left(box), (-1, 2))
        self.p0, self.p1, self.p2, self.p3 = self.box
        self.height = cmp_distance_points(self.p0, self.p3)
        self.width = cmp_distance_points(self.p0, self.p1)
        self.left = (self.p0 + self.p3) / 2
        self.right = (self.p1 + self.p2) / 2
        diff = self.right - self.left
        self.k = diff[1] / diff[0]
        self.b = self.left[1] - self.left[0] * self.k

    def covered_area(self, mask):
        min_x = min(np.int0(self.box[:, 0]))
        max_x = max(np.int0(self.box[:, 0]))
        min_y = min(np.int0(self.box[:, 1]))
        max_y = max(np.int0(self.box[:, 1]))
        return np.mean(mask[min_y:max_y, min_x:max_x, 0])


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


def distance_point_to_line(p, k, b):
    """计算点到直线的距离"""
    x, y = p
    return abs(x * k - y + b) / np.sqrt(k ** 2 + 1)


def cmp_distance_points(p1, p2):
    """计算点和点之间的距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def merge_box(line):
    res = []
    b1 = line.pop(0)
    while line:
        b2 = line[0]
        # 两个框的距离
        distance = cmp_distance_points(b1.right, b2.left)
        if b1.right[0] > b2.left[0] or distance < 30:
            # b1收集一个框
            b1 = BOX([b1.p0, b1.p1, b2.p0, b2.p1, b2.p2, b2.p3, b1.p2, b1.p3])
            line.pop(0)
        else:
            print(f"distance: {distance}")
            # 把b1放入结果中，搜索下一个b1
            res.append(b1)
            b1 = line.pop(0)
    res.append(b1)
    return res


def merge_line(line):
    obj_list = []

    # op 对每一行的小片段(box)进行处理：
    line.sort(key=lambda x: x.left[0])
    obj = []
    base_box = line[0]
    # 对每一行的文本进行拼接
    for box in line:
        dif_height = abs(box.height - base_box.height)
        if dif_height < 5:
            obj.append(box)
        else:
            # op
            obj_list.append(obj)
            base_box = box
            obj = [box]
    if len(obj) != 0:
        obj_list.append(obj)
    index = 0

    while index < len(obj_list):
        obj = obj_list[index]
        # 对obj进行拼接，如果obj中有一个以上的box，则对其合并
        if len(obj) > 1:
            obj_list[index] = merge_box(obj)
        index += 1
    return obj_list


def split_and_merge(bboxes):
    # 创建BOX类
    box_list = []
    for box in bboxes:
        # 最小外接矩形框，有方向角
        box = BOX(box)
        box_list.append(box)
    if len(box_list) == 0:
        return []
    # 按照中心点高度排序
    box_list.sort(key=lambda x: x.center[1])
    # 收集相同高度的框
    all_lines = []
    same_line = []
    base_box = box_list[0]
    for box in box_list:
        # 计算与base box的高度差
        dif_height = distance_point_to_line(box.center, base_box.k, base_box.b)
        if dif_height < 1:
            print(f"dif_height: {dif_height}")
            same_line.append(box)
        else:
            all_lines += merge_box(same_line)
            base_box = box
            same_line = [box]
    if len(same_line) != 0:
        all_lines += merge_box(same_line)
    return all_lines
    # print(all_lines)
    # exit()
    # all_text = []
    # for boxes in all_lines:
    #     all_text.extend(boxes)
    # return all_text


def draw_info(image_data, mask, box, color=(0, 0, 222)):
    # vision
    if mask is not None:
        cv2.fillConvexPoly(mask, box.box, (255, 255, 255))
        cv2.circle(mask, np.int0(box.center), 2, color)
        cv2.line(mask, np.int0(box.left), np.int0(box.right), color, 2)
        for idx, (x, y) in enumerate(box.box.tolist()):
            cv2.putText(mask, str(idx), (x, y), 1, 1, color, 1)
    # image data
    box = np.reshape(box.box, (-1, 1, 2))
    cv2.polylines(image_data, [box], 1, (0, 0, 255))
    return image_data


if __name__ == '__main__':
    image_data = cv2.imdecode(np.fromfile("dataset/1.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = np.zeros_like(image_data, dtype=np.uint8)
    bboxes = np.load("dataset/1.npy")
    all_text = split_and_merge(bboxes)
    # vision
    color = (0, 0, 255)
    # all_text = np.reshape(all_text, (-1)).flatten()
    for box in all_text:
        color = (255 - np.array(color)).tolist()
        draw_info(image_data, mask, box, color)
    cv2.imshow("im", image_data)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
