import cv2
import time
import numpy as np


def ufunc_4(S1, S2, TAG):
    t1 = time.time()
    # indices 四邻域 x-1 x+1 y-1 y+1，如果等于TAG 则赋值为label
    for h in range(1, S1.shape[0] - 1):
        for w in range(1, S1.shape[1] - 1):
            label = S1[h][w]
            if label != 0:
                if S2[h][w - 1] == TAG:
                    S2[h][w - 1] = label
                if S2[h][w + 1] == TAG:
                    S2[h][w + 1] = label
                if S2[h - 1][w] == TAG:
                    S2[h - 1][w] = label
                if S2[h + 1][w] == TAG:
                    S2[h + 1][w] = label
    print("cost time: ", time.time() - t1)


def scale_expand_kernel(S1, S2):
    TAG = 10240
    S2[S2 == 255] = TAG
    mask = (S1 != 0)
    S2[mask] = S1[mask]
    cond = True
    while (cond):
        before = np.count_nonzero(S1 == 0)
        ufunc_4(S1, S2, TAG)
        S1[S2 != TAG] = S2[S2 != TAG]
        after = np.count_nonzero(S1 == 0)
        if before <= after:
            cond = False

    return S1


def filter_label_by_area(labelimge, num_label, area=5):
    for i in range(1, num_label + 1):
        if np.count_nonzero(labelimge == i) <= area:
            labelimge[labelimge == i] == 0
    return labelimge


def scale_expand_kernels(kernels, filter=False):
    '''
    args:
        kernels : S(0,1,2,..n) scale kernels , Sn is the largest kernel
    '''
    S = kernels[0]
    num_label, labelimage = cv2.connectedComponents(S.astype('uint8'))
    if filter:
        labelimage = filter_label_by_area(labelimage, num_label)
    for Si in kernels[1:]:
        labelimage = scale_expand_kernel(labelimage, Si)
    return num_label, labelimage


def fit_boundingRect_2(num_label, labelImage):
    rects = []
    for label in range(1, num_label + 1):
        points = np.array(np.where(labelImage == label)[::-1]).T
        x, y, w, h = cv2.boundingRect(points)
        rect = np.array([x, y, x + w, y + h])
        rects.append(rect)
    return rects


newres1 = np.load("../kernels.npy")
print(newres1.shape)
mask = np.zeros_like(newres1)[0, ...]
# res1 = res[0]
# res1[res1 > 0.9] = 1
# res1[res1 <= 0.9] = 0
# for i in range(7):
#     cv2.imshow("c", res[i] * 255)
#     cv2.waitKey(0)

# newres1 = []
# for i in range(2, 5):
#     n = np.logical_and(res1[:, :, 5], res1[:, :, i]) * 255
#     newres1.append(n)
# newres1.append(res1[:, :, 5] * 255)
num_label, labelimage = scale_expand_kernels(newres1, filter=False)
rects = fit_boundingRect_2(num_label, labelimage)
for x1, y1, x2, y2 in rects:
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 1)
cv2.imshow("m", mask)
cv2.waitKey(0)
