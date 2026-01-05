import cv2

class VehicleDetector:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def get_centroids(self, frame, min_area=80):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_small)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel_big)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    centroids.append((cx, cy))
        return centroids, closing