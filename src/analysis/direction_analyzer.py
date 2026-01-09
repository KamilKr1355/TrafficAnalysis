import pandas as pd

class DirectionAnalyzer:
    def __init__(self, lines_config):
        self.lines = lines_config
        self.counts = {d: {'OSOBOWY': 0, 'CIEZAROWY': 0, 'BUS': 0, 'JEDNOSLAD': 0} for d in ['UP', 'DOWN', 'LEFT', 'RIGHT']}
        self.car_states = {} 
        self.car_labels = {} 
        self.best_crops = {} 
        self.counted_ids = set()
        self.events = []

    def check_crossing(self, car_id, p_pos, c_pos, bbox, frame, detector, frame_idx, fps):
        if car_id in self.counted_ids or p_pos is None: return None
        if car_id not in self.car_states:
            self.car_states[car_id] = {d: 'none' for d in self.counts.keys()}

        px, py, cx, cy = p_pos[0], p_pos[1], c_pos[0], c_pos[1]
        s = self.car_states[car_id]

        for d in self.counts.keys():
            if s[d] == 'invalid': continue
            val_p, val_c = (py, cy) if d in ['UP', 'DOWN'] else (px, cx)
            out, inn = self.lines[d.lower()]['outer'], self.lines[d.lower()]['inner']
            
            # 1. Strefa gotowości (inner)
            if s[d] == 'none':
                if (val_p >= inn > val_c if d in ['UP', 'LEFT'] else val_p <= inn < val_c):
                    s[d] = 'ready'

            # 2. Aktywne szukanie najlepszego zdjęcia
            if s[d] == 'ready':
                label, crop = detector.classify_and_crop(frame, bbox)
                if label in ['CIEZAROWY', 'BUS'] or car_id not in self.car_labels:
                    self.car_labels[car_id], self.best_crops[car_id] = label, crop

                # 3. Zliczenie (outer)
                if (val_p >= out > val_c if d in ['UP', 'LEFT'] else val_p <= out < val_c):
                    lbl = self.car_labels.get(car_id, "OSOBOWY")
                    crp = self.best_crops.get(car_id, crop)
                    self.counts[d][lbl] += 1
                    self.counted_ids.add(car_id)
                    self.events.append({'ID': car_id, 'Typ': lbl, 'Kierunek': d, 'Czas': round(frame_idx/fps, 2)})
                    return crp
        return None

    def save_results(self, path_sum, path_log):
        res = []
        for d, t in self.counts.items():
            row = {'Kierunek': d}; row.update(t); res.append(row)
        pd.DataFrame(res).to_csv(path_sum, index=False)
        if self.events: pd.DataFrame(self.events).to_csv(path_log, index=False)