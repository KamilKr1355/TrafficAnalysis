import cv2

class Drawer:
    @staticmethod
    def draw_traffic_lines(frame, lines):
        cv2.line(frame, (0, lines['up']['outer']), (frame.shape[1], lines['up']['outer']), (0, 255, 255), 2)
        cv2.line(frame, (0, lines['up']['inner']), (frame.shape[1], lines['up']['inner']), (0, 0, 255), 2)
        cv2.line(frame, (0, lines['down']['outer']), (frame.shape[1], lines['down']['outer']), (0, 165, 255), 2)
        cv2.line(frame, (0, lines['down']['inner']), (frame.shape[1], lines['down']['inner']), (255, 255, 0), 2)
        cv2.line(frame, (lines['left']['outer'], 0), (lines['left']['outer'], frame.shape[0]), (255, 0, 0), 2)
        cv2.line(frame, (lines['left']['inner'], 0), (lines['left']['inner'], frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (lines['right']['outer'], 0), (lines['right']['outer'], frame.shape[0]), (42, 42, 165), 2)
        cv2.line(frame, (lines['right']['inner'], 0), (lines['right']['inner'], frame.shape[0]), (200, 222, 245), 2)

    @staticmethod
    def draw_ui(frame, counts):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (180, 130), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, (k, v) in enumerate(counts.items()):
            cv2.putText(frame, f"{k.upper()}: {v}", (10, 25 + i*25), 1, 1.2, (255, 255, 255), 2)