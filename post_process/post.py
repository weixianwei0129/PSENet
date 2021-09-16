import cv2
import pyclipper
import numpy as np


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def postprocess_bitmap(img, labels):
    # h, w
    ratio = np.array(img.shape[:2]) / np.array(labels.shape[:2])
    results = []
    for i in range(1, np.max(labels) + 1):
        label = ((labels == i).astype(int) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            box, _ = get_mini_boxes(cnt)
            # w, h
            box = (np.reshape(box, [-1, 2]) * ratio[::-1].T).astype(int)
            p0, p1, p2, p3 = box
            h = int(distance(p0, p3))
            w = int(distance(p0, p1))

            pts1 = np.float32([p0, p1, p3])
            pts2 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
            dst = cv2.warpAffine(img, cv2.getAffineTransform(pts1, pts2), (w, h))
            results.append(
                dict(
                    textImg=dst,
                    Polygon=cnt.astype(int).tolist(),
                    minRect=box,
                )
            )
    return results


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
