# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# pip install easyocr

import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

'''
Read image for one line 
IMAGE_PATH = 'img/Hello.jpg'

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
result = reader.readtext(IMAGE_PATH)
print(result)

# Draw results 
img = cv2.imread(IMAGE_PATH)

# Make sure we actually got a detection
if len(result) > 0:
    top_left = tuple(map(int, result[0][0][0]))
    bottom_right = tuple(map(int, result[0][0][2]))
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw rectangle and text
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    img = cv2.putText(img, text, top_left, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Convert color for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Visualize
plt.imshow(img)
plt.axis('off')
plt.show()
'''


# for multiple lines 
IMAGE_PATH = 'img/out_of_service.jpg'

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
result = reader.readtext(IMAGE_PATH)
print(result)


# Draw results 
img = cv2.imread(IMAGE_PATH)

for detection in result:
    top_left = tuple([int(val) for val in detection [0][0]])
    bottom_right = tuple([int(val) for val in detection [0][0]])
    text = detection [1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    img = cv2.putText(img, text, top_left, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()