import cv2
from ultralytics import YOLO
import numpy as np

# =========================
# 1. LOAD MODEL & VIDEO
# =========================
# Using raw strings (r"") to avoid Windows path errors
model = YOLO(r"C:\Users\ADMIN\Desktop\GEMINI tomato_app\best.pt")
video_path = r"C:\Users\ADMIN\Downloads\video tomatoes\VID_20260312_153546.mp4"
cap = cv2.VideoCapture(video_path)

# =========================
# 2. PARAMETERS & REGISTRIES
# =========================
confidence_threshold = 0.25
selected_fps = 10
paused = False

total_registry = set()
class_registry = {k: set() for k in ["bloom", "green", "turning", "red", "damaged"]}
# Stability Registry: Stores how many frames each ID has been seen
seen_ids = {} 

# UI Dimensions
VIDEO_W, VIDEO_H, INFO_W = 960, 540, 360
win_name = "Tomato Yield Monitor - Meru University"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# =========================
# 3. INTERACTIVE CLICK HANDLER
# =========================
def handle_click(event, x, y, flags, param):
    global selected_fps, confidence_threshold
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate X relative to the start of the sidebar
        sx = x - VIDEO_W 
        
        # Only process if the click is actually inside the sidebar area
        if 0 <= sx <= INFO_W:
            # --- FPS ADJUSTMENTS (Row Y=440) ---
            # Increase (+)
            if 210 <= sx <= 250 and 420 <= y <= 460:
                selected_fps += 1
                print(f"FPS Increased to: {selected_fps}")
            
            # Decrease (-)
            if 250 < sx <= 290 and 420 <= y <= 460:
                selected_fps = max(1, selected_fps - 1)
                print(f"FPS Decreased to: {selected_fps}")

            # --- CONFIDENCE ADJUSTMENTS (Row Y=480) ---
            # Increase (+)
            if 210 <= sx <= 250 and 460 <= y <= 500:
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Confidence Increased to: {confidence_threshold:.2f}")
            
            # Decrease (-)
            if 250 < sx <= 290 and 460 <= y <= 500:
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"Confidence Decreased to: {confidence_threshold:.2f}")

cv2.setMouseCallback(win_name, handle_click)

# =========================
# 4. MAIN ANALYTICS LOOP
# =========================
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret: break

        # AI Inference with ByteTrack (Best for movement)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", 
                              conf=confidence_threshold, imgsz=640, verbose=False)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names

            for obj_id, cls_idx in zip(ids, classes):
                # STABILITY LOGIC: Count if seen for AT LEAST 3 frames
                seen_ids[obj_id] = seen_ids.get(obj_id, 0) + 1
                
                if seen_ids[obj_id] >= 3:
                    total_registry.add(obj_id)
                    label = names[cls_idx].lower()
                    if label in class_registry:
                        class_registry[label].add(obj_id)

        # Rendering
        annotated = results[0].plot(labels=True, conf=False) # Keep IDs visible for proof
        video_resized = cv2.resize(annotated, (VIDEO_W, VIDEO_H))

        # --- Dashboard Sidebar ---
        sidebar = np.zeros((VIDEO_H, INFO_W, 3), dtype=np.uint8)
        sidebar[:] = (30, 31, 34) # Professional Dark Gray

        # Title & Branding
        cv2.putText(sidebar, "YIELD ANALYTICS", (40,50), 1, 1.8, (255,255,255), 2)
        cv2.line(sidebar, (40, 70), (INFO_W-40, 70), (100, 100, 100), 1)

        # Main Yield Metric
        cv2.putText(sidebar, f"TOTAL: {len(total_registry)}", (40,120), 1, 2.0, (0,255,0), 2)

        # Class Breakdown
        y_offset = 180
        for k, v in class_registry.items():
            color = (0, 0, 255) if k == "red" else (200, 200, 200)
            cv2.putText(sidebar, f"{k.upper()}: {len(v)}", (40, y_offset), 1, 1.2, color, 1)
            y_offset += 35

        # Controls UI
        cv2.putText(sidebar, "SYSTEM CONTROLS", (40,400), 1, 1.2, (255,255,255), 1)
        # FPS Row
        cv2.putText(sidebar, f"FPS: {selected_fps}", (40,440), 1, 1.2, (255,255,255), 1)
        cv2.putText(sidebar, "+", (220, 440), 1, 1.5, (0, 255, 255), 2)
        cv2.putText(sidebar, "-", (260, 440), 1, 1.5, (0, 255, 255), 2)
        # Conf Row
        cv2.putText(sidebar, f"CONF: {confidence_threshold:.2f}", (40,480), 1, 1.2, (255,255,255), 1)
        cv2.putText(sidebar, "+", (220, 480), 1, 1.5, (0, 255, 255), 2)
        cv2.putText(sidebar, "-", (260, 480), 1, 1.5, (0, 255, 255), 2)

        # Combine Panels
        dashboard = np.hstack((video_resized, sidebar))
        cv2.imshow(win_name, dashboard)

    # Physics: Temporal Control
    delay = max(1, int(1000 / selected_fps))
    key = cv2.waitKey(delay) & 0xFF
    if key == ord(' '): paused = not paused
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()