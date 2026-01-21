import cv2
import numpy as np
from ultralytics import YOLO
import config

class VehicleDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.yolo = YOLO("yolov8n.pt") 
        
        self.class_map = {1: 'JEDNOSLAD', 2: 'OSOBOWY', 3: 'JEDNOSLAD', 5: 'BUS', 7: 'CIEZAROWY'}

    def get_blobs(self, frame):
        """Szybkie wykrywanie ruchu (MOG2)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, self.kernel)
        dilation = cv2.dilate(opening, self.kernel, iterations=2)
        _, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if config.MIN_CONTOUR_AREA < area < config.MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    blobs.append({'pos': (cx, cy), 'bbox': (x, y, w, h)})
        return blobs, dilation

    def classify_and_crop(self, frame, bbox):
        """AI analizuje wycinek i zwraca najlepszą klatkę"""
        x, y, w, h = bbox
        
        pad_h = int(h * 1.6)
        pad_w = int(w * 1.6)
        
        y1, y2 = max(0, y), min(frame.shape[0], y+pad_h)
        x1, x2 = max(0, x), min(frame.shape[1], x+pad_w)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0: return "OSOBOWY", None

        results = self.yolo(roi, conf=0.15, classes=[1,2,3,5,7], verbose=False)[0]
        label = "OSOBOWY"
        max_conf = 0.0

        for box in results.boxes:
            c_id = int(box.cls[0])
            conf = box.conf[0].item()

            if c_id not in self.class_map:
                continue

            candidate = self.class_map[c_id]

            if candidate == "CIEZAROWY":
                if w < 260 or h < 180:
                    continue

            if candidate == "BUS":
                if w < 230 or h < 200:
                    continue

            if conf > max_conf:
                max_conf = conf
                label = candidate

        # Fail-safe dla wielkich naczep
        if (w > 280 or h > 280) and label == "OSOBOWY": 
            label = "CIEZAROWY"
        return label, roi