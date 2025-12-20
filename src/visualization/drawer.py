import cv2

def draw(frame, tracked_objects, counters):
    for obj_id, obj in tracked_objects.items():
        cx, cy = obj["centroid"]
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            frame, f"ID {obj_id}", (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    y = 30
    for k, v in counters.items():
        cv2.putText(
            frame, f"{k}: {v}", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        y += 30

    return frame
