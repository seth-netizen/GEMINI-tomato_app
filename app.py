from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

model = YOLO("best.pt")
video_path = "VID_20260312_153546.mp4"

# Global System State
total_registry = set()
class_registry = {k: set() for k in ["bloom flowers", "green", "turning", "red", "damaged tomatoes"]}
seen_ids = {}
conf_threshold = 0.25
target_fps = 10

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # AI with dynamic Confidence
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
                              imgsz=160, conf=conf_threshold, verbose=False)
        
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names
            for obj_id, cls_idx in zip(ids, classes):
                seen_ids[obj_id] = seen_ids.get(obj_id, 0) + 1
                if seen_ids[obj_id] >= 3:
                    total_registry.add(obj_id)
                    label = names[cls_idx].lower()
                    if label in class_registry:
                        class_registry[label].add(obj_id)

        annotated_frame = results[0].plot(labels=True, conf=False)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        
        # Control playback speed (Wait logic)
        process_time = time.time() - start_time
        wait_time = max(0, (1.0 / target_fps) - process_time)
        time.sleep(wait_time)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    # Make sure keys here match the IDs in the HTML (bloom, green, turning, red, damaged)
    stats = {
        'bloom flowers': len(class_registry.get('bloom flowers', set())), # Match your specific YOLO name
        'green': len(class_registry.get('green', set())),
        'turning': len(class_registry.get('turning', set())),
        'red': len(class_registry.get('red', set())),
        'damaged': len(class_registry.get('damaged tomatoes', set())), # Match your specific YOLO name
        'total': len(total_registry)
    }
    return jsonify(stats)
# Route to update settings from the web sliders
@app.route('/update_settings', methods=['POST'])
def update_settings():
    global conf_threshold, target_fps
    data = request.json
    conf_threshold = float(data.get('conf', 0.25))
    target_fps = int(data.get('fps', 10))
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)