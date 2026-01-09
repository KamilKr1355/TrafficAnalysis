import sys, os, cv2
# Ścieżki muszą być ustawione przed importem config
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.append(current_dir)

import config
from detection.vehicle_detector import VehicleDetector
from tracking.centroid_tracker import CentroidTracker
from analysis.direction_analyzer import DirectionAnalyzer
from visualization.drawer import Drawer

def main():
    video_abs_path = os.path.join(os.path.dirname(current_dir), config.VIDEO_PATH)
    cap = cv2.VideoCapture(video_abs_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Zmienna do przechowania tła wykresu
    first_frame_bg = None

    out,inn = 30,100

    lines_config = {
        'up':    {'outer': out, 'inner': inn},
        'down':  {'outer': config.FRAME_HEIGHT - out, 'inner': config.FRAME_HEIGHT - inn},
        'left':  {'outer': out, 'inner': inn},
        'right': {'outer': config.FRAME_WIDTH - out, 'inner': config.FRAME_WIDTH - inn}
    }

    det, trk, anz = VehicleDetector(), CentroidTracker(cap.get(cv2.CAP_PROP_FRAME_COUNT)), DirectionAnalyzer(lines_config)
    static_img, timer = None, 0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Skalowanie klatki
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        # Zapisujemy pierwszą klatkę jako tło dla wykresu
        if first_frame_bg is None:
            first_frame_bg = frame.copy()

        if frame_idx % 2 != 0: # Skok dla wydajności
            frame_idx += 1; continue

        blobs, debug_mask = det.get_blobs(frame)
        trk.update(frame_idx, blobs)

        for c_id in list(trk.active_cars_pos.keys()):
            c_pos = trk.active_cars_pos[c_id]
            p_pos = trk.df.at[frame_idx - 2, str(c_id)] if frame_idx >= 2 else None
            bbox = next((b['bbox'] for b in blobs if b['pos'] == tuple(c_pos)), (0,0,0,0))
            
            img = anz.check_crossing(c_id, p_pos if p_pos != "" else None, c_pos, bbox, frame, det, frame_idx, fps)
            if img is not None:
                h_i, w_i = img.shape[:2]
                scale = min(450/w_i, 350/h_i)
                static_img, timer = cv2.resize(img, (0,0), fx=scale, fy=scale), 50

            cv2.circle(frame, tuple(c_pos), 5, (0, 0, 255), -1)

        if static_img is not None and timer > 0:
            h_s, w_s = static_img.shape[:2]
            y_o, x_o = 20, config.FRAME_WIDTH - w_s - 20
            if y_o + h_s < config.FRAME_HEIGHT:
                frame[y_o:y_o+h_s, x_o:x_o+w_s] = static_img
                cv2.rectangle(frame, (x_o-1, y_o-1), (x_o+w_s+1, y_o+h_s+1), (0, 255, 0), 2)
            timer -= 1
        elif timer <= 0: static_img = None

        Drawer.draw_traffic_lines(frame, lines_config)
        Drawer.draw_ui(frame, anz.counts)
        cv2.imshow('Traffic Analysis ULTIMATE + PLOT', frame)
        cv2.imshow('MOG2 Debug', cv2.resize(debug_mask, (640, 360)))
        
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27: break

    # --- ZAKOŃCZENIE I EKSPORT ---
    print("\nGenerowanie raportów i wykresu podsumowującego...")
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    
    # 1. Zapis CSV
    anz.save_results(os.path.join(data_dir, 'raport_typy.csv'), os.path.join(data_dir, 'log_przejazdow.csv'))
    
    # 2. Zapis WYKRESU (summary_plot.png)
    if first_frame_bg is not None:
        Drawer.save_summary_plot(trk.df, first_frame_bg, os.path.join(data_dir, 'summary_plot.png'))
    
    cap.release(); cv2.destroyAllWindows()
    print("Gotowe! Wszystkie pliki znajdziesz w folderze /data.")

if __name__ == "__main__":
    main()