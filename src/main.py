import sys
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

from detection.vehicle_detector import VehicleDetector
from tracking.centroid_tracker import CentroidTracker
from analysis.direction_analyzer import DirectionAnalyzer
from visualization.drawer import Drawer

def main():
    video_path = os.path.join(os.path.dirname(current_dir), 'data', 'videos', 'video.mp4')
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)

    m_out, m_inn = 15, 60
    lines_config = {
        'up':    {'outer': m_out,          'inner': m_inn},
        'down':  {'outer': height - m_out, 'inner': height - m_inn},
        'left':  {'outer': m_out,          'inner': m_inn},
        'right': {'outer': width - m_out,  'inner': width - m_inn}
    }

    detector = VehicleDetector()
    tracker = CentroidTracker(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzer = DirectionAnalyzer(lines_config)
    
    frame_idx = 0
    print("Analiza w toku... Nacisnij ESC aby zakonczyc i zapisac raport.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (width, height))
        centroids, _ = detector.get_centroids(frame)
        tracker.update(frame_idx, centroids)

        for car_id in tracker.car_ids:
            pos = tracker.get_pos(frame_idx, car_id)
            if pos != '' and pos is not None:
                # PRZEKAZUJEMY frame_idx i fps do logowania czasu
                analyzer.check_crossing(int(car_id), pos, frame_idx, fps)
                cv2.circle(frame, tuple(pos), 3, (0, 0, 255), -1)

        Drawer.draw_traffic_lines(frame, lines_config)
        Drawer.draw_ui(frame, analyzer.counts)
        
        cv2.imshow('Traffic Counter Pro', frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27: break

    # ZAPISYWANIE DANYCH PO ZAKONCZENIU
    print("\n--- ANALIZA ZAKONCZONA ---")
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    summary_path = os.path.join(data_dir, 'raport_sumaryczny.csv')
    log_path = os.path.join(data_dir, 'log_przejazdow.csv')
    
    analyzer.save_results(summary_path, log_path)
    
    print(f"Zapisano sumy w: {summary_path}")
    print(f"Zapisano log zdarzen w: {log_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()