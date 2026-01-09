import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class CentroidTracker:
    def __init__(self, max_frames, max_radius=120):
        self.max_radius = max_radius 
        self.df = pd.DataFrame(index=range(int(max_frames)))
        self.active_cars_pos = {}
        self.total_cars = 0
        self.disappeared = {}
        self.max_disappeared = 12 

    def update(self, frame_idx, current_detections):
        cxx = [d['pos'][0] for d in current_detections]
        cyy = [d['pos'][1] for d in current_detections]
        used_indices = []

        if not self.active_cars_pos:
            for det in current_detections:
                self._add_new_car(frame_idx, det['pos'][0], det['pos'][1])
            return

        ids = list(self.active_cars_pos.keys())
        for car_id in ids:
            prev_pos = self.active_cars_pos[car_id]
            distances = [np.sqrt((prev_pos[0]-cx)**2 + (prev_pos[1]-cy)**2) for cx, cy in zip(cxx, cyy)]
            
            if distances:
                m_idx = np.argmin(distances)
                if distances[m_idx] < self.max_radius and m_idx not in used_indices:
                    self.active_cars_pos[car_id] = [cxx[m_idx], cyy[m_idx]]
                    self.df.at[frame_idx, str(car_id)] = [cxx[m_idx], cyy[m_idx]]
                    self.disappeared[car_id] = 0
                    used_indices.append(m_idx)
                else: self.disappeared[car_id] = self.disappeared.get(car_id, 0) + 1
            else: self.disappeared[car_id] = self.disappeared.get(car_id, 0) + 1

            if self.disappeared.get(car_id, 0) > self.max_disappeared:
                del self.active_cars_pos[car_id]
                del self.disappeared[car_id]

        for i, det in enumerate(current_detections):
            if i not in used_indices:
                self._add_new_car(frame_idx, det['pos'][0], det['pos'][1])

    def _add_new_car(self, frame_idx, x, y):
        new_id = str(self.total_cars)
        self.df[new_id] = ""
        self.df.at[frame_idx, new_id] = [x, y]
        self.active_cars_pos[int(new_id)] = [x, y]
        self.total_cars += 1
        if self.total_cars % 50 == 0: self.df = self.df.copy()