"""
Streamlit app: RTSP face data collection with detection and tracking.
- Input RTSP URL as-is (no %23/%24 conversion).
- Live stream with face detection (bestempfacereal.pt) and tracking (DeepSORT).
- Persistent sorted IDs per face; capture 50 images per face to autofacedata.
- CSV: person_name, file_path (comma-separated paths).

Glitch-free stream:
- Dedicated reader thread only reads RTSP (no processing = no delay).
- FFmpeg options: TCP, nobuffer, low_delay to reduce HEVC "Could not find ref with POC" errors.
- Every frame and every saved crop is validated; glitched frames are skipped (not shown, not saved).
"""

import os
import csv
import cv2
import time
import threading
from collections import defaultdict

import streamlit as st
import numpy as np

# Paths – use as-is
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOFACEDATA_BASE_DIR = r"D:\ardiyan\ardiyan\uploads\autofacedata"
AUTOFACEDATA_CSV = r"D:\ardiyan\ardiyan\uploads\autofacedata.csv"
FACE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "bestempfacereal.pt")
NUM_IMAGES_PER_FACE = 50
FACE_CLASS_ID = 0
CONF_THRESHOLD = 0.05
# NMS IOU: lower = stronger suppression of overlapping boxes (one face = one detection)
NMS_IOU = 0.40
# Expand face box by this ratio (e.g. 1.2 = 20% larger) to capture full face + extra context
FACE_BOX_EXPAND_RATIO = 1.2
# Save every frame we have a track = quick capture, multiple poses
SAVE_INTERVAL_FRAMES = 1  # Changed to 1 to achieve 3 images per second (assuming ~30fps)
# Run YOLO every N frames so display stays smooth (1 = every frame, 2 = every 2nd)
DETECT_EVERY_N_FRAMES = 2
# Display loop: iterations between reruns (fewer reruns = less flicker)
DISPLAY_LOOP_ITERATIONS = 90
DISPLAY_LOOP_SLEEP_SEC = 0.033
SHARPNESS_THRESHOLD = 100.0

# Lazy imports
YOLO_AVAILABLE = False
DEEPSORT_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    pass


class _SimpleTrack:
    """Minimal track-like object for drawing when we skip detection frame."""
    __slots__ = ("track_id", "_bbox")
    def __init__(self, tid, bbox):
        self.track_id = tid
        self._bbox = bbox
    def to_ltrb(self):
        return self._bbox
    def is_confirmed(self):
        return True


def ensure_dirs(person_name=None):
    """Create base directory and person-specific folder if person_name provided."""
    os.makedirs(AUTOFACEDATA_BASE_DIR, exist_ok=True)
    if person_name:
        person_dir = os.path.join(AUTOFACEDATA_BASE_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir
    return AUTOFACEDATA_BASE_DIR


def get_or_create_csv():
    """Return (rows, fieldnames). Creates file with header if missing."""
    fieldnames = ["person_name", "file_path"]
    if os.path.exists(AUTOFACEDATA_CSV):
        rows = []
        with open(AUTOFACEDATA_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fn = reader.fieldnames or fieldnames
            for row in reader:
                rows.append(row)
        return rows, fn
    # Create CSV with header
    ensure_dirs()
    with open(AUTOFACEDATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return [], fieldnames


def save_csv(rows, fieldnames):
    try:
        with open(AUTOFACEDATA_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return True
    except Exception:
        return False


def update_csv_for_person(person_name, paths_list):
    """Update or append row for this person with comma-separated file_path."""
    rows, fieldnames = get_or_create_csv()
    path_str = ",".join(paths_list)
    found = False
    for row in rows:
        if str(row.get("person_name", "")).strip() == str(person_name).strip():
            row["file_path"] = path_str
            found = True
            break
    if not found:
        rows.append({"person_name": person_name, "file_path": path_str})
    return save_csv(rows, fieldnames)


def expand_bbox(bbox, frame_shape, ratio=None):
    """Expand bbox by ratio from center; clamp to frame. For full face + extra features."""
    if ratio is None:
        ratio = FACE_BOX_EXPAND_RATIO
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_shape[:2]
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return (x1, y1, x2, y2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    nw, nh = cw * ratio, ch * ratio
    nx1 = int(cx - nw / 2)
    ny1 = int(cy - nh / 2)
    nx2 = int(cx + nw / 2)
    ny2 = int(cy + nh / 2)
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)
    return (nx1, ny1, nx2, ny2)


def extract_face_crop(frame, bbox, expanded=True):
    """Extract face region; use expanded bbox if expanded=True for full face capture."""
    if expanded:
        bbox = expand_bbox(bbox, frame.shape)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def get_fixed_person_crop(frame, bbox, size=640):
    """
    Create fixed-size square crop centered on face.
    Pads if crop goes outside frame.
    """
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    half = size // 2

    nx1 = cx - half
    ny1 = cy - half
    nx2 = cx + half
    ny2 = cy + half

    crop = np.zeros((size, size, 3), dtype=np.uint8)

    src_x1 = max(0, nx1)
    src_y1 = max(0, ny1)
    src_x2 = min(w, nx2)
    src_y2 = min(h, ny2)

    dst_x1 = src_x1 - nx1
    dst_y1 = src_y1 - ny1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return crop


def is_frame_usable(frame):
    """
    Reject HEVC/decoder glitched frames (e.g. 'Could not find ref with POC').
    Glitched frames often have: very low variance, washed out, or broken structure.
    """
    if frame is None or frame.size == 0:
        return False
    try:
        mean = np.mean(frame)
        std = np.std(frame)
        if std < 5.0 or mean < 3 or mean > 252:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 15:  # heavily corrupted / blocky HEVC glitch
            return False
        return True
    except Exception:
        return False

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < SHARPNESS_THRESHOLD

def is_crop_usable(crop):
    """Reject face crops that are corrupted, all black, or HEVC glitch artifacts."""
    if crop is None or crop.size == 0:
        return False
    try:
        h, w = crop.shape[:2]
        if w < 20 or h < 20:
            return False
        # 👉 NEW: Blur check
        if is_blurry(crop):
            return False
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 8 or mean < 8 or mean > 247:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
            return False
        return True
    except Exception:
        return False


def _reader_thread(rtsp_url, state):
    """
    Only job: read from RTSP as fast as possible. No processing = no delay, no blocking.
    HEVC/low-latency options set so decoder doesn't buffer. Glitches still possible; processor will skip bad frames.
    """
    # FFmpeg: TCP (reliable), no buffer, low delay – reduces HEVC ref-frame errors
    prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            state["error"] = "Could not open RTSP stream. Check URL and network."
            return
        while not state.get("stop", False):
            ret, frame = cap.read()
            if ret and frame is not None:
                state["latest_frame"] = frame
                state["latest_ok"] = True
            else:
                state["latest_ok"] = False
            time.sleep(0)
        cap.release()
    except Exception as e:
        state["error"] = str(e)
    finally:
        if prev is not None:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


def _processor_thread(state, person_name):
    """
    Grabs latest frame from reader; never blocks the reader. Validates frame (skip HEVC glitches),
    runs YOLO every Nth frame, only saves crops that pass validation = no glitched snapshots.
    """
    ensure_dirs(person_name)
    if not YOLO_AVAILABLE:
        state["error"] = "ultralytics not installed. pip install ultralytics"
        return
    if not DEEPSORT_AVAILABLE:
        state["error"] = "deep_sort_realtime not installed. pip install deep-sort-realtime"
        return
    if not os.path.exists(FACE_MODEL_PATH):
        state["error"] = f"Face model not found: {FACE_MODEL_PATH}"
        return

    model = YOLO(FACE_MODEL_PATH)
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=0.7,
        max_cosine_distance=0.3,
        nn_budget=100,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
    )

    next_persistent_id = 1
    track_to_persistent = {}
    persistent_counts = defaultdict(int)
    persistent_paths = defaultdict(list)
    last_bbox_by_track = {}
    frame_index = 0
    last_good_display = None  # show this when current frame is glitched
    last_save_time = time.time()
    save_interval = 1.0 / 3.0  # 3 images per second

    try:
        while not state.get("stop", False):
            frame = state.get("latest_frame")
            if frame is None:
                time.sleep(0.01)
                continue
            frame = frame.copy()
            frame_index += 1

            # Skip glitched frames (HEVC POC / decoder errors) – don't display, don't save
            if not is_frame_usable(frame) or is_blurry(frame):
                if last_good_display is not None:
                    state["frame"] = last_good_display
                time.sleep(0.001)
                continue

            display_frame = frame.copy()
            run_detection = (frame_index % DETECT_EVERY_N_FRAMES == 1)

            if run_detection:
                results = model(
                    frame,
                    conf=CONF_THRESHOLD,
                    iou=NMS_IOU,
                    classes=[FACE_CLASS_ID],
                    verbose=False,
                )
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) != FACE_CLASS_ID:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, "face"))
                tracked = tracker.update_tracks(detections, frame=frame)
            else:
                tracked = []
                for track_id, bbox in list(last_bbox_by_track.items()):
                    if track_id in track_to_persistent:
                        tracked.append(_SimpleTrack(track_id, bbox))

            for track in tracked:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                raw_bbox = (x1, y1, x2, y2)
                # Use expanded bbox for display and crop (full face + extra features)
                bbox = expand_bbox(raw_bbox, frame.shape)
                x1, y1, x2, y2 = bbox
                last_bbox_by_track[track_id] = bbox

                if track_id not in track_to_persistent:
                    track_to_persistent[track_id] = next_persistent_id
                    next_persistent_id += 1
                persistent_id = track_to_persistent[track_id]
                pid_str = f"{persistent_id:03d}"

                color = (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame, f"ID {persistent_id}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                # Save only when crop is valid (no glitched images); crop uses expanded bbox
                # Rate limit to 3 images per second
                current_time = time.time()
                if (persistent_counts[persistent_id] < NUM_IMAGES_PER_FACE and 
                    current_time - last_save_time >= save_interval):
                    
                    crop = extract_face_crop(frame, bbox, expanded=True)
                    if crop is not None and is_crop_usable(crop):
                        crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LANCZOS4)

                        # Save to person-specific folder
                        person_dir = os.path.join(AUTOFACEDATA_BASE_DIR, person_name)
                        fname = f"face_{person_name}_{persistent_counts[persistent_id]:03d}.jpg"
                        out_path = os.path.join(person_dir, fname)
                        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        rel_path = os.path.join("autofacedata", person_name, fname)
                        persistent_paths[persistent_id].append(rel_path)
                        persistent_counts[persistent_id] += 1
                        last_save_time = current_time
                        
                        if persistent_counts[persistent_id] == NUM_IMAGES_PER_FACE:
                            # Update CSV for the person (no ID column)
                            update_csv_for_person(person_name, persistent_paths[persistent_id])
                            state["completed_ids"] = state.get("completed_ids", set()) | {persistent_id}

                cv2.putText(
                    display_frame, f"{persistent_counts[persistent_id]}/{NUM_IMAGES_PER_FACE}",
                    (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )

            if run_detection:
                current_ids = {t.track_id for t in tracked if t.is_confirmed()}
                for tid in list(last_bbox_by_track.keys()):
                    if tid not in current_ids:
                        last_bbox_by_track.pop(tid, None)

            out_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            last_good_display = out_rgb
            state["frame"] = out_rgb
            state["counts"] = dict(persistent_counts)
            state["completed"] = list(state.get("completed_ids", set()))
    except Exception as e:
        state["error"] = str(e)


def main():
    st.set_page_config(page_title="RTSP Face Data Collection", layout="wide")
    st.title("RTSP Face Data Collection")
    st.markdown("Enter RTSP URL **as-is** (no conversion of %23/%24). Connect to show live stream, detect faces with **bestempfacereal.pt**, track with DeepSORT, and capture 50 images per face to `autofacedata` with CSV.")

    if "worker_started" not in st.session_state:
        st.session_state.worker_started = False
    if "stop" not in st.session_state:
        st.session_state.stop = False
    if "stream_state" not in st.session_state:
        st.session_state.stream_state = {}
    if "current_person_name" not in st.session_state:
        st.session_state.current_person_name = ""

    rtsp_url = st.text_input(
        "RTSP URL",
        value="rtsp://Kartikkumar:xyz456%23%24@172.30.121.41:554/Streaming/Channels/801",
        help="Use the URL exactly as provided; do not decode %23 or %24.",
    )
    
    person_name = st.text_input(
        "Enter Person Name",
        placeholder="e.g., John_Doe, Jane_Smith, etc.",
        help="Enter the name of the person. Images will be saved in a folder with this name."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("Start Stream & Collect Images")
    with col2:
        stop_clicked = st.button("Stop Stream")

    if stop_clicked:
        st.session_state.stop = True
        st.session_state.worker_started = False
        if "stream_state" in st.session_state and isinstance(st.session_state.stream_state, dict):
            st.session_state.stream_state["stop"] = True
        st.rerun()

    if start_clicked and rtsp_url.strip():
        if not person_name.strip():
            st.error("Please enter a person name before starting the stream.")
            return
        st.session_state.current_person_name = person_name.strip()
        st.session_state.stop = False
        st.session_state.worker_started = True
        st.session_state.stream_state = {
            "stop": False, "frame": None, "error": None,
            "latest_frame": None, "latest_ok": False,
            "counts": {}, "completed": [], "completed_ids": set(),
        }

    if st.session_state.worker_started and rtsp_url.strip() and not st.session_state.stop:
        state = st.session_state.stream_state
        state["stop"] = st.session_state.stop

        if not state.get("thread_started"):
            state["thread_started"] = True
            # Reader: only reads RTSP (0 blocking on processing). Processor: validates + detects + saves.
            threading.Thread(target=_reader_thread, args=(rtsp_url.strip(), state), daemon=True).start()
            threading.Thread(target=_processor_thread, args=(state, st.session_state.current_person_name), daemon=True).start()

        if state.get("error"):
            st.error(state["error"])
            st.session_state.worker_started = False
            return

        # Single placeholder: update in place for many iterations to avoid flicker (fewer reruns)
        place = st.empty()
        status_place = st.empty()
        for _ in range(DISPLAY_LOOP_ITERATIONS):
            if st.session_state.stop:
                break
            if state.get("frame") is not None:
                place.image(state["frame"], channels="RGB", use_container_width=True)
            counts = state.get("counts", {})
            completed = state.get("completed", [])
            if counts or completed:
                status_place.markdown(
                    f"**Person:** {st.session_state.current_person_name}  |  "
                    f"**Per-face progress:** {dict(counts)}  |  "
                    f"**Completed (50 images):** {sorted(completed)}  |  "
                    f"**CSV:** `{AUTOFACEDATA_CSV}`"
                )
            time.sleep(DISPLAY_LOOP_SLEEP_SEC)
        st.rerun()
    if st.session_state.stop:
        st.session_state.worker_started = False
        st.session_state.stop = False
        st.info(f"Stream stopped. Images for {st.session_state.current_person_name} saved in {os.path.join(AUTOFACEDATA_BASE_DIR, st.session_state.current_person_name)}")

    st.info("Enter RTSP URL and person name, then click **Start Stream & Collect Images** to begin. Faces will get persistent IDs; 50 images per face are saved to person-specific folders and recorded in CSV.")


if __name__ == "__main__":
    main()