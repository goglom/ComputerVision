import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

nums = []
for i in range(10):
    nums.append(cv.imread(str(i) + '.jpg'))

ex = cv.imread('t1.jpg')
canvas = ex.copy()
pts = []
for i in range(10):
    pts.append([])

for i in range(10):
    w, h, _ = nums[i].shape[::-1]
    res = cv.matchTemplate(ex, nums[i], cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        pts[i].append(pt)
        cv.rectangle(canvas, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)

resVec = []
for x in range(len(pts)):
    for y in range(len(pts[x])):
        resVec.append([pts[x][y][0], x])

resVec.sort()
time = str(resVec[0][1]) + str(resVec[1][1]) +':' + str(resVec[2][1]) + str(resVec[3][1])

plt.figure()
plt.title(f"Time: {time}")
plt.imshow(canvas)
plt.axis('off')
plt.show()
