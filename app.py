from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

model = YOLO("best.pt")
video_path = "tomato_video.mp4"

# Global System State
total_registry = set()
class_registry = {k: set() for k in ["bloom flowers", "green", "turning", "red", "damaged tomatoes"]}
seen_ids = {}
conf_threshold = 0.25
target_fps = 10

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Restart video
            continue
        
        try:
            # Physics: Downsampling resolution to 160 dramatically reduces FLOPs
            # (Floating Point Operations), preventing server timeout.
            results = model.track(frame, persist=True, imgsz=160, conf=conf_threshold, verbose=False)
            
            # Extract counting logic...
            if results[0].boxes.id is not None:
                # ... (keep your existing counting code here) ...
                pass

            annotated_frame = results[0].plot(labels=True, conf=False)
        except Exception as e:
            # Fallback: If AI fails/lags, just show the raw frame so the video doesn't break
            print(f"AI Error: {e}")
            annotated_frame = frame

        # Compress to JPEG to reduce network bandwidth
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
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