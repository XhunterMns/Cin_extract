import os
import cv2

# ultralytics YOLO is optional; keep defensive import
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

DEFAULT_IMAGE_PATH = os.path.abspath("static/files/cin.jpg")
_MODEL_CACHE = {}


def _load_model(model_path=None):
    model_path = model_path or "yolov8n.pt"
    if not _YOLO_AVAILABLE:
        return None
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        m = YOLO(model_path)
        _MODEL_CACHE[model_path] = m
        return m
    except Exception:
        return None


def detect_text_regions(image_path=DEFAULT_IMAGE_PATH, conf_thresh=0.25, model_path=None):
    """
    Return list of boxes (x1,y1,x2,y2,score) for candidate regions detected by YOLO.
    If YOLO not available or there are no detections, returns [].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]

    model = _load_model(model_path)
    if model is None:
        return []

    try:
        results = model(image_path)  # returns list-like
    except Exception:
        return []

    boxes = []
    # results may be list of Results
    for res in results:
        # res.boxes: contain xyxy and conf
        for box in getattr(res, "boxes", []):
            xyxy = box.xyxy.cpu().numpy().flatten() if hasattr(box.xyxy, "cpu") else box.xyxy.numpy().flatten()
            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") and hasattr(box.conf, "cpu") else float(getattr(box, "conf", 0.0))
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def crop_relative(image, top, bottom, left, right):
    height, width = image.shape[:2]
    y1 = max(0, int(height * top))
    y2 = min(height, int(height * bottom))
    x1 = max(0, int(width * left))
    x2 = min(width, int(width * right))
    return image[y1:y2, x1:x2]


def extract_fields_by_region(image_path=DEFAULT_IMAGE_PATH):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    fields = {
        "header": (0.00, 0.23, 0.05, 0.95),
        "cin_number": (0.18, 0.38, 0.36, 0.80),
        "name_block": (0.28, 0.58, 0.22, 0.93),
        "birth_block": (0.52, 0.76, 0.23, 0.93),
        "address_block": (0.68, 0.96, 0.18, 0.95),
    }

    results = {}
    for field_name, region in fields.items():
        crop = crop_relative(image, *region)
        variants = {"raw": crop}
        best = choose_best_ocr(variants, psms=(6, 7, 11))
        results[field_name] = normalize_text(best["text"])

    return results


if __name__ == "__main__":
    extracted = extract_fields_by_region()
    print("=== Extracted CIN Info ===")
    for field, text in extracted.items():
        print(f"{field}: {text}")
