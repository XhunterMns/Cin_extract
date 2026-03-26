import os
import sys

import cv2
import numpy as np
import tempfile


def read_image(path_or_bytes):
    """
    Accept either:
      - filesystem path (str)
      - bytes/bytearray with image file contents (jpg/png ...)
    Return BGR numpy array or None.
    """
    # raw bytes
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # path (str)
    if not isinstance(path_or_bytes, str):
        return None

    # expand ~ and relative to project
    p = os.path.expanduser(path_or_bytes)
    if not os.path.isabs(p):
        base = os.path.abspath(os.path.dirname(__file__))  # module dir
        p = os.path.join(base, p)

    if not os.path.exists(p):
        return None

    img = cv2.imread(p)
    return img


def resize_for_ocr(image, target_width=1600):
    height, width = image.shape[:2]
    if width >= target_width:
        return image

    scale = target_width / float(width)
    new_size = (target_width, max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def to_greyscale(image):
    if image is None:
        raise ValueError("to_greyscale received None as image")

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image)


def denoise(image):
    return cv2.fastNlMeansDenoising(image, None, 15, 7, 21)


def sharpen(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1.6, gaussian, -0.6, 0)


def _foreground_mask(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def crop_to_text_region(image, padding=20):
    gray = to_greyscale(image)
    mask = _foreground_mask(gray)
    points = cv2.findNonZero(mask)

    if points is None:
        return image

    x, y, w, h = cv2.boundingRect(points)
    if w < image.shape[1] * 0.35 or h < image.shape[0] * 0.25:
        return image

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + w + padding)
    y1 = min(image.shape[0], y + h + padding)
    return image[y0:y1, x0:x1]


def deskew_image(gray_image):
    mask = _foreground_mask(gray_image)
    points = cv2.findNonZero(mask)

    if points is None:
        return gray_image

    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5:
        return gray_image

    height, width = gray_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def build_ocr_variants(image):
    gray = to_greyscale(image)
    normalized = normalize_contrast(gray)
    cleaned = denoise(normalized)
    sharpened = sharpen(cleaned)
    deskewed = deskew_image(sharpened)

    otsu = cv2.threshold(
        deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    adaptive = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    return {
        "gray": gray,
        "normalized": normalized,
        "cleaned": cleaned,
        "deskewed": deskewed,
        "otsu": otsu,
        "adaptive": adaptive,
    }


def preprocess_image(file_path, save_debug=False, debug_prefix=None):
    image = read_image(file_path)
    image = resize_for_ocr(image)
    image = crop_to_text_region(image)
    variants = build_ocr_variants(image)

    if save_debug:
        if not debug_prefix:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            debug_prefix = f"{base_name}_preprocessed"

        output_dir = os.path.dirname(os.path.abspath(file_path))
        for variant_name, variant_image in variants.items():
            out_path = os.path.join(output_dir, f"{debug_prefix}_{variant_name}.png")
            cv2.imwrite(out_path, variant_image)

    return variants["deskewed"]


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "static/files/cin.jpg"
    try:
        image = read_image(input_path)
        image = resize_for_ocr(image)
        image = crop_to_text_region(image)
        variants = build_ocr_variants(image)

        if not os.path.isabs(input_path):
            base_dir = os.path.abspath(os.path.dirname(__file__))
            input_path_abs = os.path.join(base_dir, input_path)
        else:
            input_path_abs = input_path

        base_name = os.path.splitext(os.path.basename(input_path_abs))[0]
        out_dir = os.path.dirname(input_path_abs)

        for variant_name, variant_image in variants.items():
            out_path = os.path.join(out_dir, f"{base_name}_{variant_name}.png")
            cv2.imwrite(out_path, variant_image)
            print(f"Wrote: {out_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
