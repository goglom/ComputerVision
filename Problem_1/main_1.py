import cv2

img = cv2.imread('text.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (640, 480))


trsh = cv2.adaptiveThreshold(img, 100, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 7, 2)

cv2.imshow('before', img)
cv2.imshow('after', trsh)

cv2.waitKey(0)

