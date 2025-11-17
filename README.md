# OCR Preprocessing and Contour Detection

The file Grayscaling.py performs image preprocessing and contour-based symbol detection using OpenCV, NumPy, and Matplotlib.  
It prepares handwritten or printed mathematical expressions for OCR by cleaning noise and isolating individual characters.

The script steps through:
- Grayscale conversion  
- Gaussian noise reduction  
- Adaptive thresholding  
- Morphological opening and closing  
- Contour detection  
- Visual output via Matplotlib  

---

## Requirements

bash
pip install opencv-python==4.12.0
pip install numpy==2.1.1
pip install matplotlib==3.10.7


How to Run the Script

Run the file normally:
python text_processing.py



Two visual windows will appear:

- Grayscale → Adaptive Threshold → Morphology pipeline

- Contour Detection (characters marked with red bounding boxes, this works best with test_2)


## What the Script Does (Step-by-Step)

### 1. Load Image
The script loads the input image located at `img/test.jpg` using OpenCV.

### 2. Convert to Grayscale
The original BGR image is converted into a grayscale image, making it easier to process and threshold.

### 3. Apply Gaussian Blur
A `(5 × 5)` Gaussian kernel is applied to reduce small noise and smooth the image before thresholding.

### 4. Adaptive Thresholding
The blurred grayscale image is converted into a binary black-and-white image using:

- `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`
- Block size: 11
- Constant subtracted from mean: 2

This produces clear separation between the handwriting and the background.

### 5. Morphological Operations
Two morphology steps are applied:

- Opening — removes isolated white noise pixels.
- Closing — fills small black gaps inside the characters.

These operations help clean the text for better contour detection.

### 6. Resize & Combine Preprocessing Visuals
Grayscale, binary, and cleaned images are resized to the same height and concatenated horizontally using `cv2.hconcat`, allowing a clear visual comparison of each preprocessing stage.

### 7. Invert Clean Image
The cleaned image is inverted (white becomes black and vice versa) to prepare it for contour detection, since OpenCV detects white objects on black backgrounds.

### 8. Contour Detection
Contours are extracted using:

- `cv2.RETR_EXTERNAL` — retrieves only external contours  
- `cv2.CHAIN_APPROX_SIMPLE` — compresses contour points for efficiency  

Small noise contours are filtered out.

### 9. Draw Bounding Boxes
For each valid contour, a red rectangle is drawn on a copy of the original image:

- Uses `cv2.boundingRect(cnt)`
- Ignores very small contours (width/height ≤ 2)

This isolates individual digits or symbols.

### 10. Show Results
Two figures are displayed:

1. **Grayscale | Adaptive Threshold | Morphology**  
2. **Contours Detection** (bounding boxes drawn on the original image)



