# CodeAlpha_ObjectDetectionTracking
 
CodeAlpha AI Internship – **Task 4: Object Detection and Tracking**

Pipeline (per frame): **OpenCV** capture → **YOLOv8** (pre-trained on COCO) detection → **Deep SORT** tracking → draw boxes, class labels and tracking IDs in real time.

## Run
```bash
pip install -r requirements.txt
python main.py                          # webcam
python main.py --source video.mp4       # video file
python main.py --source video.mp4 --save output.mp4
```
Press **q** to quit. YOLO weights (`yolov8n.pt`) download automatically on first run.
Use `--model yolov8s.pt` for better accuracy, `--conf 0.5` to be stricter.
 
Developed an Object Detection project during my CodeAlpha internship to identify and locate objects in images using computer vision and deep learning techniques. The project detects multiple objects, classifies them accurately, and highlights their locations with bounding boxes, demonstrating practical applications of AI and image processing.
 
