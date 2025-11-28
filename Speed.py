from ultralytics import YOLO
import cv2
from math import sqrt

model = YOLO('yolov8n.pt')
video = cv2.VideoCapture('video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

ret, first = video.read()
if not ret:
    raise Exception("Cannot read video")

h, w = first.shape[:2]
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

previous = {}
limit = 50
ppm = 0.04

while True:
    ret, frame = video.read()
    if not ret:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml")
    boxes = result[0].boxes

    if len(boxes) == 0:
        cv2.imshow('Vehicles', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    for box in boxes:
        cls = int(box.cls[0])
        if cls not in [2, 3, 5, 7]:
            continue

        vid = int(box.id[0])
        x1, y1, x2, y2 = box.xyxy[0]

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if vid in previous:
            px, py = previous[vid]
            pixel_dist = sqrt((cx - px)**2 + (cy - py)**2)
            real_dist = pixel_dist * ppm
            time = 1 / fps
            speed_mps = real_dist / time
            speedkmph = speed_mps * 3.6
        else:
            speedkmph = 0

        previous[vid] = (cx, cy)
        color = (0, 0, 255) if speedkmph > limit else (0, 255, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame, f"{speedkmph:.1f} km/h",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    out.write(frame)

    cv2.imshow('Speed Detection', frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
