import os
from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# 1. Configuration (Relative paths for Render)
model = YOLO("best.pt")
video_path = "tomato_video.mp4" # ENSURE FILE IS RENAMED IN GITHUB

# 2. Global State (Physics: State Integration over Time)
total_registry = set()
class_registry = {k: set() for k in ["bloom flowers", "green", "turning", "red", "damaged tomatoes"]}
seen_ids = {}
conf_threshold = 0.25
target_fps = 10

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FAILED TO OPEN: {video_path}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        try:
            # Physics: imgsz=160 minimizes FLOPs to stay under Render CPU limits
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                  imgsz=160, conf=conf_threshold, verbose=False)
            
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()
                names = results[0].names
                
                for obj_id, cls_idx in zip(ids, classes):
                    # Filtering for persistence
                    seen_ids[obj_id] = seen_ids.get(obj_id, 0) + 1
                    if seen_ids[obj_id] >= 3:
                        total_registry.add(obj_id)
                        label = names[cls_idx].lower()
                        if label in class_registry:
                            class_registry[label].add(obj_id)

            annotated_frame = results[0].plot(labels=True, conf=False)
        except Exception as e:
            print(f"AI INFERENCE ERROR: {e}")
            annotated_frame = frame

        # JPEG Compression (Quality 40 minimizes packet size for mobile viewing)
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    # Syncing dictionary keys with your index.html IDs
    return jsonify({
        'bloom': len(class_registry['bloom flowers']),
        'green': len(class_registry['green']),
        'turning': len(class_registry['turning']),
        'red': len(class_registry['red']),
        'damaged': len(class_registry['damaged tomatoes']),
        'total': len(total_registry)
    })

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global conf_threshold, target_fps
    data = request.json
    conf_threshold = float(data.get('conf', 0.25))
    target_fps = int(data.get('fps', 10))
    return jsonify(success=True)

if __name__ == "__main__":
    # Physics: Dynamic port allocation for Cloud Hosting
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)