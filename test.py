import cv2
import numpy as np
import easyocr
import re
import json

class TunisianCINExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)

    def preprocess(self, path):
        img = cv2.imread(path)
        if img is None: return None
        img = cv2.resize(img, (1200, 800))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # INCREASE CONTRAST: This makes the blue ink much darker
        # compared to the white background
        alpha = 2.0 # Contrast control
        beta = -50  # Brightness control
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        return adjusted

    def extract(self, path):
        processed_img = self.preprocess(path)
        results = self.reader.readtext(processed_img, detail=1)
        
        data = {"cin": None, "nom": None, "prenom": None, "birth_year": None}
        
        # Reference coordinates to filter out headers
        y_cin = 0
        y_birth = 800

        # Step 1: Find Anchors (CIN and Year)
        for (bbox, text, prob) in results:
            clean_digits = re.sub(r'\D', '', text)
            if len(clean_digits) == 8:
                data["cin"] = clean_digits
                y_cin = (bbox[0][1] + bbox[2][1]) / 2
            
            year_match = re.search(r'\b(19|20)\d{2}\b', text)
            if year_match:
                data["birth_year"] = year_match.group(0)
                y_birth = (bbox[0][1] + bbox[2][1]) / 2

        # Step 2: Extract Names between the Anchors
        # We search specifically for strings that AREN'T the labels
        potential_fields = []
        labels_to_ignore = ["اللقب", "الاسم", "الإسم", "لقب", "اسم", "الام"]

        for (bbox, text, prob) in results:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            x_center = (bbox[0][0] + bbox[1][0]) / 2
            
            # 1. Filter by "The Fence": must be below CIN and above Birth Date
            if y_cin < y_center < y_birth and x_center > 350:
                clean_ar = re.sub(r'[^\u0600-\u06FF ]', '', text).strip()
                
                # 2. Check if this specific box is JUST a label
                is_label = any(label == clean_ar for label in labels_to_ignore)
                
                if not is_label and len(clean_ar) > 2:
                    # 3. If the box contains both label and name (e.g. "الاسم دليلة")
                    # We strip the label out
                    for label in labels_to_ignore:
                        clean_ar = clean_ar.replace(label, "").strip()
                    
                    if len(clean_ar) > 2:
                        potential_fields.append({'text': clean_ar, 'y': y_center})

        # Step 3: Map by position
        potential_fields.sort(key=lambda x: x['y'])
        
        if len(potential_fields) >= 1:
            data["nom"] = potential_fields[0]['text']
        if len(potential_fields) >= 2:
            data["prenom"] = potential_fields[1]['text']

        return data

# --- RUN ---
path = "/home/chahine/Desktop/ML project/static/files/cin4.jpg"
extractor = TunisianCINExtractor()
print(json.dumps(extractor.extract(path), indent=4, ensure_ascii=False))

