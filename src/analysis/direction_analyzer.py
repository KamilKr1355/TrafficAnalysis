class DirectionAnalyzer:
    def __init__(self):
        self.counted_ids = set()
        self.counters = {
            "left": 0,
            "right": 0,
            "up": 0,
            "down": 0
        }

    def analyze(self, tracked_objects):
        for obj_id, obj in tracked_objects.items():
            history = obj["history"]
            if len(history) < 2 or obj_id in self.counted_ids:
                continue

            dx = history[-1][0] - history[0][0]
            dy = history[-1][1] - history[0][1]

            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"

            self.counters[direction] += 1
            self.counted_ids.add(obj_id)

        return self.counters
