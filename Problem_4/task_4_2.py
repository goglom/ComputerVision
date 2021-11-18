from matplotlib import pyplot as plt
import cv2

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (400, 400))

"""
https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html <- OCV docs


"""

eq_img = cv2.equalizeHist(img)

histr = cv2.calcHist([img], [0], None, [256], [0, 256])
eq_histr = cv2.calcHist([eq_img], [0], None, [256], [0, 256])
plt.plot(histr, label='Source')
plt.plot(eq_histr, label='Equalized')
plt.legend()
plt.grid()

cv2.imshow('Source image', img)
cv2.imshow('Equalized Image', eq_img)

cv2.waitKey(0)

plt.show()