#pip install opencv-python
# pip install numpy

import cv2
import numpy as np
# Loads the image
image = cv2.imread("img/test.jpg")


# Converts to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applies Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Applies adaptive thresholding
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C = Adaptive method, cv2.THRESH_BINARY = Threshold type
# 11 = area size for local mean must be odd to be centered, 2 = Constant subtracted from mean
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.ones((3, 3), np.uint8)
# Opening: removes small white noise
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: fills small black holes inside text
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

combined = cv2.hconcat([gray, binary])

# Shows both images together
cv2.imshow("Grayscale (Left) | Adaptive Threshold (Right)", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
