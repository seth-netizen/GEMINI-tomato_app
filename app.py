import os, tempfile, time, cv2
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")

# Global variables for real-time stats
total_registry = set()
class_registry = {k: set() for k in ["bloom flowers", "green", "turning", "red", "damaged tomatoes"]}
seen_ids = {}
conf_threshold = 0.25
current_video_path = None # Starts empty

def generate_frames():
    global current_video_path
    if not current_video_path: return

    cap = cv2.VideoCapture(current_video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break # End of uploaded video
        
        results = model.track(frame, persist=True, imgsz=160, conf=conf_threshold, verbose=False)
        
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names
            for obj_id, cls_idx in zip(ids, classes):
                seen_ids[obj_id] = seen_ids.get(obj_id, 0) + 1
                if seen_ids[obj_id] >= 3:
                    total_registry.add(obj_id)
                    label = names[cls_idx].lower()
                    if label in class_registry: class_registry[label].add(obj_id)

        ret, buffer = cv2.imencode('.jpg', results[0].plot(labels=True, conf=False), [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_path, total_registry, seen_ids, class_registry
    if 'file' not in request.files: return "No file", 400
    
    file = request.files['file']
    # Reset stats for new video
    total_registry.clear()
    seen_ids.clear()
    for k in class_registry: class_registry[k].clear()
    
    # Save to temp path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(tfile.name)
    current_video_path = tfile.name
    return jsonify(success=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'bloom': len(class_registry['bloom flowers']),
        'green': len(class_registry['green']),
        'turning': len(class_registry['turning']),
        'red': len(class_registry['red']),
        'damaged': len(class_registry['damaged tomatoes']),
        'total': len(total_registry)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)