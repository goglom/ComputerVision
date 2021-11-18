import cv2

img = cv2.imread('mik.jpg', cv2.IMREAD_GRAYSCALE)

#trsh = cv2.adaptiveThreshold(img, 50, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 9, 0)

print(img.shape[0])
cv2.imshow('image', img)
cv2.waitKey(0)