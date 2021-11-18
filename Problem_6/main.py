import cv2 as cv
import numpy as np


def detect(cnt, img):
    preds = {}
    w, h, _ = img.shape
    orig_canvas = np.zeros((w, h), dtype=np.uint8)
    fig_canvas = np.zeros((w, h), dtype=np.uint8)
    cv.drawContours(orig_canvas, [cnt], -1, 255, -1)

    def get_proba():
        res = 1 - np.count_nonzero(np.bitwise_xor(fig_canvas, orig_canvas)) / (w * h)
        fig_canvas.fill(0)
        return res

    ellipse = cv.fitEllipse(cnt)
    cv.ellipse(fig_canvas, ellipse, 255 ,-1)
    cv.ellipse(img, ellipse, (255, 0, 0) , 3)
    preds["ellipse"] = get_proba()
    preds["circle"] = preds["ellipse"] * min(ellipse[1]) / max(ellipse[1])

    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(fig_canvas, [box], -1, 255, -1)
    cv.drawContours(img, [box], -1, (0, 0, 255), 3)
    
    preds["rectangle"] = get_proba()
    preds["square"] = preds["rectangle"] *  min(rect[1]) / max(rect[1])

    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 3:
        cv.drawContours(fig_canvas, [approx], -1, 255, -1)
        cv.drawContours(img, [approx], -1, (255, 255, 0), 3)
        
        preds["triangle"] = get_proba()
    else:
        preds["triangle"] = 0.0
    
    return preds



img = cv.imread('vase.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

edged = cv.Canny(img_gray, 50, 200)
ks = 20
kernel = cv.getStructuringElement(cv.MORPH_RECT, (ks, ks))
closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
contours = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

max_area = -1
S = img.shape[0] * img.shape[1]
for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    if max_area < area <= 0.95 * S:
        cnt = contours[i]
        max_area = area

_, size, _ = cv.minAreaRect(cnt)
img_size = img.shape[:2]

size_factor = max(max(size) / max(img_size), min(size) / min(img_size))

if (size_factor < 0.33):
    print("Small object", end="")
elif (size_factor < 0.66):
    print("Medium object", end="")
else:
    print("Big object", end="")
print(f": {size_factor * 100:.2f} %")


preds = detect(cnt, img)

print("The most likely shape of this object: ", max(preds, key=preds.get))
print(*preds.items(), sep='\n')

cv.drawContours(img, [cnt], -1, (0, 255, 0), 3)
cv.imshow('contours', img)
cv.waitKey(0)
