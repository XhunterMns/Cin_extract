import cv2
import pytesseract

import os

image_path = 'static/files/cin.jpg'
image_path = os.path.abspath(image_path)  
print("Looking for image at:", image_path)

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found at", image_path)
        return ""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Optional denoise
    gray = cv2.medianBlur(gray, 3)

    # Extract text
    text = pytesseract.image_to_string(gray, lang='arabic')
    print("Extracted Text:")
    print(text)

    return text