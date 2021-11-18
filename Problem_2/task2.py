import cv2

img = cv2.imread('mike.jpg')
x_size, y_size = img.shape[0], img.shape[1]

cv2.circle(img, (x_size//2 + 45, y_size//2 - 120), 150, (255, 0, 0), 5)
cv2.circle(img, (x_size//2, y_size//2 - 200), 20, (0, 0, 255), 2)
cv2.circle(img, (x_size//2 + 75, y_size//2 - 200), 20, (0, 0, 255), 2)
mouse_center = (x_size//2 + 45, y_size//2 - 60)
cv2.ellipse(img, mouse_center, (60, 20), 0, 0, -180, (0, 0, 0), 4)

cv2.namedWindow('mike', cv2.WINDOW_NORMAL)
cv2.imshow("mike", img)
cv2.waitKey(0)