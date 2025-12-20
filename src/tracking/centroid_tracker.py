import math

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def _distance(self, c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, detections):
        updated_objects = {}

        for (x, y, w, h) in detections:
            cx = x + w // 2
            cy = y + h // 2
            matched_id = None

            for obj_id, obj in self.objects.items():
                if self._distance((cx, cy), obj["centroid"]) < self.max_distance:
                    matched_id = obj_id
                    break

            if matched_id is None:
                updated_objects[self.next_id] = {
                    "centroid": (cx, cy),
                    "history": [(cx, cy)]
                }
                self.next_id += 1
            else:
                updated_objects[matched_id] = {
                    "centroid": (cx, cy),
                    "history": self.objects[matched_id]["history"] + [(cx, cy)]
                }

        self.objects = updated_objects
        return self.objects
