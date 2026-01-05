import pandas as pd

class DirectionAnalyzer:
    def __init__(self, lines_config):
        self.lines = lines_config 
        self.counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        self.car_states = {} 
        self.counted_ids = set()
        self.events = [] # Lista do przechowywania sensownych danych (logów)

    def check_crossing(self, car_id, pos, frame_idx, fps):
        if car_id in self.counted_ids:
            return

        if car_id not in self.car_states:
            self.car_states[car_id] = {'up': 'none', 'down': 'none', 'left': 'none', 'right': 'none'}

        x, y = pos
        s = self.car_states[car_id]
        time_sec = round(frame_idx / fps, 2)

        # Logika sprawdzania kierunków z blokadą "invalid"
        for direction in ['up', 'down', 'left', 'right']:
            if s[direction] == 'invalid': continue

            # Parametry osiowe w zależności od kierunku
            val = y if direction in ['up', 'down'] else x
            outer = self.lines[direction]['outer']
            inner = self.lines[direction]['inner']
            
            # Warunek przekroczenia (rosnąco lub malejąco)
            is_beyond_outer = val < outer if direction in ['up', 'left'] else val > outer
            is_beyond_inner = val < inner if direction in ['up', 'left'] else val > inner

            if is_beyond_outer and s[direction] != 'ready':
                s[direction] = 'invalid'
            elif is_beyond_inner and s[direction] == 'none':
                s[direction] = 'ready'
            elif is_beyond_outer and s[direction] == 'ready':
                self.counts[direction] += 1
                self.counted_ids.add(car_id)
                # ZAPISUJEMY SENZOWNE DANE:
                self.events.append({
                    'ID': car_id,
                    'Kierunek': direction.upper(),
                    'Sekunda': time_sec,
                    'Klatka': frame_idx
                })

    def save_results(self, output_path_summary, output_path_log):
        # 1. Zapisujemy podsumowanie (ile aut gdzie)
        summary_data = {
            'Kierunek': [k.upper() for k in self.counts.keys()],
            'Suma_Pojazdow': list(self.counts.values())
        }
        pd.DataFrame(summary_data).to_csv(output_path_summary, index=False)
        
        # 2. Zapisujemy szczegółowy log zdarzeń (co i kiedy)
        if self.events:
            pd.DataFrame(self.events).to_csv(output_path_log, index=False)