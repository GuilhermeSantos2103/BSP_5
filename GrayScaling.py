import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Resize images for alignment
height = 400
gray_resized = cv2.resize(gray, (int(gray.shape[1] * height / gray.shape[0]), height))
binary_resized = cv2.resize(binary, (int(binary.shape[1] * height / binary.shape[0]), height))
cleaned_resized = cv2.resize(cleaned, (int(cleaned.shape[1] * height / cleaned.shape[0]), height))

# Combines all three images horizontally
combined = cv2.hconcat([gray_resized, binary_resized, cleaned_resized])

plt.figure(figsize=(14, 6))
plt.imshow(combined, cmap="gray")
plt.title("Grayscale | Adaptive Threshold | Morphology")
plt.axis("off")
plt.show()

# Inverts cleaned image for contour detection
cleaned_inv = cv2.bitwise_not(cleaned)  

# Finds contours on the inverted cleaned binary image
contours, hierarchy = cv2.findContours(cleaned_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Makes a copy to draw bounding boxes
image_contours = image.copy()

# Loops through contours and draws rectangles around each detected symbol
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # filters out tiny contours that are likely noise
    if w > 2 and h > 2:
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)



plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.title("Contours Detection")
plt.axis("off")
plt.show()

