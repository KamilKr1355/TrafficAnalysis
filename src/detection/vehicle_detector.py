import cv2

class VehicleDetector:
    def __init__(self, min_area):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50
        )
        self.min_area = min_area

    def detect(self, frame):
        mask = self.bg_subtractor.apply(frame)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

        return detections
