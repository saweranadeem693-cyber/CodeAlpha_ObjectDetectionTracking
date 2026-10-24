"""CodeAlpha AI Internship - Task 4: Object Detection and Tracking
YOLOv8 (pre-trained, COCO) for detection + Deep SORT for tracking, real time via OpenCV.

Usage:
  python main.py                      # webcam
  python main.py --source video.mp4   # video file
  python main.py --source video.mp4 --save output.mp4
Press 'q' to quit.
"""
import argparse
import random
import time

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0 for webcam, or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (auto-downloaded)")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--save", default=None, help="optional output video path")
    return p.parse_args()


def color_for(track_id: int):
    random.seed(track_id)  # stable colour per ID
    return tuple(random.randint(60, 255) for _ in range(3))


def main():
    args = parse_args()
    model = YOLO(args.model)
    names = model.names
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detection
        result = model(frame, conf=args.conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0])))

        # 2) Tracking (Deep SORT)
        tracks = tracker.update_tracks(detections, frame=frame)

        # 3) Draw boxes, labels and track IDs
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            label = f"{names[t.det_class]} ID:{t.track_id}" if t.det_class is not None else f"ID:{t.track_id}"
            col = color_for(int(t.track_id))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        now = time.time()
        cv2.putText(frame, f"FPS: {1 / (now - prev + 1e-9):.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        prev = now

        if writer:
            writer.write(frame)
        cv2.imshow("Object Detection & Tracking (YOLOv8 + Deep SORT)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
