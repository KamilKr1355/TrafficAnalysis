import cv2
from src.config import *
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.centroid_tracker import CentroidTracker
from src.analysis.direction_analyzer import DirectionAnalyzer
from src.visualization.drawer import draw

cap = cv2.VideoCapture(VIDEO_PATH)

detector = VehicleDetector(MIN_CONTOUR_AREA)
tracker = CentroidTracker()
analyzer = DirectionAnalyzer()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    detections = detector.detect(frame)
    tracked = tracker.update(detections)
    counters = analyzer.analyze(tracked)

    frame = draw(frame, tracked, counters)

    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
