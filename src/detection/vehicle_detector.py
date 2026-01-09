import cv2
import numpy as np
from ultralytics import YOLO
import config

class VehicleDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.yolo = YOLO('yolov8n.pt')
        # Mapowanie klas YOLO na Twoje kategorie
        self.class_map = {1: 'JEDNOSLAD', 2: 'OSOBOWY', 3: 'JEDNOSLAD', 5: 'BUS', 7: 'CIEZAROWY'}

    def get_blobs(self, frame):
        """Szybkie wykrywanie ruchu (MOG2)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)
        _, bins = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(bins, self.kernel, iterations=2)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        # PANORAMA: Bardzo szerokie pole dla ciężarówek
        pad_h = int(h * 1.2)
        pad_w = int(w * 1.2)
        
        y1, y2 = max(0, y), min(frame.shape[0], y+pad_h)
        x1, x2 = max(0, x), min(frame.shape[1]+w, x+pad_w)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0: return "OSOBOWY", None

        results = self.yolo(roi, verbose=False, conf=0.15)[0]
        label = "OSOBOWY"
        max_conf = 0
        for box in results.boxes:
            c_id = int(box.cls[0])
            conf = box.conf[0].item()
            if c_id in self.class_map:
                boost = 1.5 if c_id in [5, 7] else 1.0
                if (conf * boost) > max_conf:
                    max_conf = conf * boost
                    label = self.class_map[c_id]

        # Fail-safe dla wielkich naczep
        #if w > 280 and label == "OSOBOWY": label = "CIEZAROWY"
        return label, roi