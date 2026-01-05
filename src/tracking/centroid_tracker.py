import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class CentroidTracker:
    def __init__(self, max_frames, max_radius=50):
        self.df = pd.DataFrame(index=range(int(max_frames)))
        self.car_ids = []
        self.total_cars = 0
        self.max_radius = max_radius

    def update(self, frame_idx, current_centroids):
        cxx = [c[0] for c in current_centroids]
        cyy = [c[1] for c in current_centroids]
        used_indices = []

        if frame_idx == 0 or not self.car_ids:
            for i in range(len(cxx)):
                self._add_new_car(frame_idx, cxx[i], cyy[i])
            return

        for car_id in self.car_ids:
            prev_pos = self.df.at[frame_idx - 1, str(car_id)]
            if prev_pos == '' or prev_pos is None or (isinstance(prev_pos, float) and np.isnan(prev_pos)):
                continue
            
            distances = [np.sqrt((prev_pos[0]-cx)**2 + (prev_pos[1]-cy)**2) for cx, cy in zip(cxx, cyy)]
            if distances:
                min_idx = np.argmin(distances)
                if distances[min_idx] < self.max_radius and min_idx not in used_indices:
                    self.df.at[frame_idx, str(car_id)] = [cxx[min_idx], cyy[min_idx]]
                    used_indices.append(min_idx)

        for i in range(len(cxx)):
            if i not in used_indices:
                self._add_new_car(frame_idx, cxx[i], cyy[i])

    def _add_new_car(self, frame_idx, x, y):
        new_id = str(self.total_cars)
        self.df[new_id] = ""
        self.df.at[frame_idx, new_id] = [x, y]
        self.car_ids.append(self.total_cars)
        self.total_cars += 1
        if self.total_cars % 50 == 0: self.df = self.df.copy()

    def get_pos(self, frame_idx, car_id):
        if frame_idx < 0: return None
        if str(car_id) not in self.df.columns: return None
        return self.df.at[frame_idx, str(car_id)]