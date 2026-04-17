import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ── Models ──────────────────────────────────────────────
model = YOLO("yolov8n.pt")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# ── Supervision Tracker ──────────────────────────────────
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

# ── Video ────────────────────────────────────────────────
# Hide main tkinter window
Tk().withdraw()

video_path = askopenfilename(title="Select Video File")

if not video_path:
    print("No file selected")
    exit()

cap = cv2.VideoCapture(video_path)
cv2.namedWindow("Supervision Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Supervision Tracking", 1280, 720)

# ── Class Filter & Styles ────────────────────────────────
class_styles = {
    "person":     {"color": (0, 255, 120)},
    "car":        {"color": (60, 120, 255)},
    "truck":      {"color": (255, 100, 50)},
    "bus":        {"color": (255, 220, 0)},
    "motorcycle": {"color": (180, 0, 255)},
}

ALLOWED_CLASSES = list(class_styles.keys())

# ── Supervision Annotators ───────────────────────────────
box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=4,
)
trace_annotator = sv.TraceAnnotator(
    thickness=2,
    trace_length=30,
)

# ── Helper: Count Display ────────────────────────────────
def draw_counts(frame, counts):
    y_offset = 20
    for label, count in counts.items():
        color = class_styles[label]["color"]
        text  = f"{label}: {count}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (10, y_offset - th - 6),
                      (10 + tw + 4, y_offset), color, -1)
        cv2.putText(frame, text, (12, y_offset - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        y_offset += 25

# ── Main Loop ────────────────────────────────────────────
frame_count = 0
detections  = sv.Detections.empty()   # safe empty default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = cv2.resize(frame, (1280, 720))
    small_frame   = cv2.resize(frame, (640, 640))

    sx = 1280 / 640
    sy = 720  / 640

    # ── Run YOLOv8 every 2nd frame ───────────────────────
    if frame_count % 2 == 0:
        results = model(small_frame, conf=0.4, verbose=False)[0]

        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter allowed classes
        if detections.class_id is not None and len(detections) > 0:
            class_mask = np.array([
                model.names[cls] in ALLOWED_CLASSES
                for cls in detections.class_id
            ])
            detections = detections[class_mask]

        # Scale coords 640x640 → 1280x720
        if len(detections) > 0:
            detections.xyxy[:, [0, 2]] *= sx # type: ignore
            detections.xyxy[:, [1, 3]] *= sy # type: ignore

        # Update ByteTrack — now detections have tracker_id
        detections = tracker.update_with_detections(detections) # type: ignore

    # ── Build labels ──────────────────────────────────────
    labels = []
    if detections.tracker_id is not None:
        for class_id, tracker_id, conf in zip(
            detections.class_id, # type: ignore
            detections.tracker_id,
            detections.confidence # type: ignore
        ):
            label_name = model.names[class_id]
            labels.append(f"ID:{tracker_id} {label_name} {conf:.0%}")

    # ── Annotate — only when tracker_id exists ────────────
    if detections.tracker_id is not None and len(detections) > 0:
        display_frame = trace_annotator.annotate(display_frame, detections)
        display_frame = box_annotator.annotate(display_frame, detections)
        display_frame = label_annotator.annotate(
            display_frame, detections, labels
        )

    # ── Count per class ───────────────────────────────────
    counts = {}
    if detections.class_id is not None:
        for class_id in detections.class_id:
            label = model.names[class_id]
            counts[label] = counts.get(label, 0) + 1

    draw_counts(display_frame, counts)

    cv2.imshow("Supervision Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()