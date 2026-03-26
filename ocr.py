import os
import re
import cv2
import tempfile
import pytesseract
import numpy as np
import base64

from image_processing import (
    read_image,
    resize_for_ocr,
    to_greyscale,
    normalize_contrast,
    denoise,
    sharpen,
    crop_to_text_region,
    deskew_image,
)
import yolo as yolo_mod

DEFAULT_IMAGE_PATH = os.path.abspath("static/files/cin.jpg")
DEFAULT_LANGUAGES = ("ara", "fra+ara", "ara+fra")
DEFAULT_PSMS = (11, 6, 3)

ARABIC_KEYWORDS = (
    "الجمهورية",
    "التونسية",
    "بطاقة",
    "التعريف",
    "الوطنية",
    "اللقب",
    "الاسم",
    "الولادة",
    "بنت",
    "بن",
)


def normalize_text(text):
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("|", " ").replace("_", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_score(text, confidence):
    if not text or not text.strip():
        return -1.0
    score = float(confidence or 0.0)
    score += len(re.findall(r"[\u0600-\u06FF]", text)) * 0.35
    score += len(re.findall(r"\d", text)) * 0.25
    score += len(re.findall(r"[٠-٩]", text)) * 0.25
    lowered = text.lower()
    for keyword in ARABIC_KEYWORDS:
        if keyword in lowered:
            score += 2.0
    if re.search(r"\b\d{8}\b", lowered):
        score += 2.0
    if re.search(r"(19|20)\d{2}", lowered):
        score += 0.8
    return score


def ocr_with_confidence(image, lang="ara", psm=11):
    """Return dict with text + average confidence from pytesseract on the provided image (numpy BGR or gray)."""
    gray = image if len(image.shape) == 2 else to_greyscale(image)
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(gray, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words = []
    line_buckets = {}
    confidences = []
    for idx, (t, conf, line_num) in enumerate(zip(data.get("text", []), data.get("conf", []), data.get("line_num", []))):
        if not t or t.strip() == "":
            continue
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        if conf_f >= 0:
            confidences.append(conf_f)
        line_buckets.setdefault(int(line_num), []).append(t)
        words.append(t)
    ordered_lines = [" ".join(line_buckets[k]) for k in sorted(line_buckets.keys())] if line_buckets else []
    normalized = normalize_text("\n".join(ordered_lines))
    avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
    return {"text": normalized, "confidence": avg_conf, "lang": lang, "psm": psm}


def build_variants_from_image(image):
    """Return dict of preprocessed variants for OCR."""
    variants = {}
    try:
        variants["raw"] = image.copy()
        variants["resized"] = resize_for_ocr(image)
        gray = to_greyscale(variants["resized"])
        variants["gray"] = gray
        variants["clahe"] = normalize_contrast(gray)
        variants["denoised"] = denoise(variants["clahe"])
        try:
            ds = deskew_image(variants["denoised"])
            if ds is not None:
                variants["deskewed"] = ds
        except Exception:
            pass
        # additional quick binarization
        otsu = cv2.threshold(variants.get("deskewed", variants["denoised"]), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants["otsu"] = otsu
    except Exception:
        # best-effort fallback
        variants["raw"] = image
    return variants


def choose_best_ocr(variants, languages=None, psms=None):
    languages = languages or DEFAULT_LANGUAGES
    psms = psms or DEFAULT_PSMS
    best_result = {"text": "", "score": -1e6, "confidence": 0.0, "variant": None, "lang": None, "psm": None, "image": None}
    preferred_order = ["deskewed", "otsu", "clahe", "gray", "resized", "raw"]
    for vname in preferred_order:
        img = variants.get(vname)
        if img is None:
            continue
        for lang in languages:
            for psm in psms:
                try:
                    res = ocr_with_confidence(img, lang=lang, psm=psm)
                except Exception:
                    continue
                txt = res.get("text", "")
                conf = res.get("confidence", 0.0)
                sc = text_score(txt, conf)
                if sc > best_result["score"]:
                    best_result.update({
                        "text": txt,
                        "score": sc,
                        "confidence": conf,
                        "variant": vname,
                        "lang": lang,
                        "psm": psm,
                        "image": img.copy() if hasattr(img, "copy") else img
                    })
    return best_result


def extract_text_from_image(image_input, debug=False):
    """
    Accepts path / bytes / ndarray. If debug=True the returned dict will contain
    'preprocessed_image' with a data URL of the best variant used for OCR.
    """
    # normalize to ndarray
    img = None
    temp_path = None
    try:
        if isinstance(image_input, (bytes, bytearray)):
            img = read_image(image_input)
            # write temp file for YOLO if needed later
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_path = tf.name
            tf.close()
            cv2.imwrite(temp_path, img)
            image_path_for_yolo = temp_path
        elif isinstance(image_input, str):
            # path
            image_path_for_yolo = image_input
            img = read_image(image_input)
        else:
            # assume ndarray
            img = image_input
            # make temp for YOLO
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_path = tf.name
            tf.close()
            cv2.imwrite(temp_path, img)
            image_path_for_yolo = temp_path

        if img is None:
            raise FileNotFoundError("image not found or could not be decoded")

        # run YOLO detection using a temporary path (yolo.detect_text_regions expects path)
        regions = []
        try:
            regions = yolo_mod.detect_text_regions(image_path=image_path_for_yolo, conf_thresh=0.25)
        except Exception:
            regions = []

        # if no regions found, crop whole image and run OCR
        if not regions:
            try:
                cropped = crop_to_text_region(img)
                if cropped is None:
                    cropped = resize_for_ocr(img)
            except Exception:
                cropped = resize_for_ocr(img)
            variants = build_variants_from_image(cropped)
            best = choose_best_ocr(variants)
            best.update({"region": None})
            return best

        # run OCR per region
        results = []
        for (x1, y1, x2, y2, score) in regions:
            crop = img[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            variants = build_variants_from_image(crop)
            best = choose_best_ocr(variants)
            best.update({"region": (x1, y1, x2, y2), "region_score": score})
            results.append(best)

        if results:
            results.sort(key=lambda r: r["score"], reverse=True)
            chosen = results[0]
        else:
            # fallback full-image OCR
            variants = build_variants_from_image(resize_for_ocr(img))
            chosen = choose_best_ocr(variants)

        # attach debug preprocessed image data URL if requested
        if debug:
            img_for_export = chosen.get("image")
            if img_for_export is None:
                # fallback to a resized full image
                img_for_export = resize_for_ocr(img)
            try:
                ok, enc = cv2.imencode('.jpg', img_for_export)
                if ok:
                    b64 = base64.b64encode(enc.tobytes()).decode('ascii')
                    chosen["preprocessed_image"] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                chosen["preprocessed_image"] = None

        # remove the ndarray from returned dict to avoid huge JSON if debug not requested
        if "image" in chosen and not debug:
            chosen.pop("image", None)

        return chosen
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def extract_fields(text):
    """Very small heuristic field extraction from OCR text. Extend as needed."""
    out = {}
    if not text:
        return out
    # find 8-digit CIN
    m = re.search(r"\b(\d{8})\b", text)
    if m:
        out["cin_number"] = m.group(1)
    # date
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    if m2:
        out["year"] = m2.group(0)
    # fallback: first two non-empty lines as name / other
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 1:
        out["line1"] = lines[0]
    if len(lines) >= 2:
        out["line2"] = lines[1]
    return out


if __name__ == "__main__":
    image_path = DEFAULT_IMAGE_PATH
    print(f"Looking for image at: {image_path}")

    result = extract_text_from_image(image_path)
    print("Best OCR configuration:")
    print(
        f"variant={result['variant']}, lang={result['lang']}, "
        f"psm={result['psm']}, confidence={result['confidence']:.2f}"
    )
    print("Extracted Text:")
    print(result["text"])

    fields = extract_fields(result["text"])
    print("Detected fields:")
    for key, value in fields.items():
        print(f"{key}: {value}")
