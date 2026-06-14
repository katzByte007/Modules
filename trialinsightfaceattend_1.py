import os
import sys
import time
import pickle
import hashlib
import contextlib
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import streamlit as st
from PIL import Image
import tempfile
from collections import deque
import queue
import asyncio

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    import onnxruntime as ort
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    st.error("Warning: insightface not installed. Run: pip install insightface onnxruntime")

# Check if OpenCV has GUI support - define at module level
def check_opencv_gui():
    """Check if OpenCV has GUI support and provide fallback."""
    try:
        cv2.namedWindow('test', cv2.WINDOW_GUI_NORMAL)
        cv2.destroyWindow('test')
        return True
    except cv2.error:
        return False

OPENCV_HAS_GUI = check_opencv_gui()

# ---------------------------------------------------------------------------
# Production: InsightFace with AntelopeV2 for robust angle-invariant recognition
# ---------------------------------------------------------------------------
MODEL_NAME = "antelopev2"
DETECTOR_BACKEND = "insightface"
MIN_SIDE_FOR_EMBEDDING = 300
RECOGNITION_THRESHOLD = 0.30
CACHE_FILENAME = "insightface_embeddings_antelopev2.pkl"

# RTSP stream configuration
RTSP_URL = r"rtsp://Kartikkumar:xyz456%23%24@172.30.121.41:554/Streaming/Channels/801"
RTSP_USE_TCP = True
LIVE_DETECTION_MAX_WIDTH = 1920
NMS_IOU = 0.40

# InsightFace detection settings
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_DET_THRESH = 0.25

# Face tracking settings
TRACK_IOU_THRESHOLD = 0.25
TRACK_MAX_AGE_FRAMES = 5
TRACK_REFRESH_NAME_EVERY = 1

# Performance optimization settings - OPTIMIZED FOR SMOOTH PLAYBACK
FRAME_SKIP = 4  # Process every 2nd frame for better performance
DISPLAY_RESIZE_WIDTH = 1920  # Keep larger for text visibility
DETECTION_RESIZE_WIDTH = 640  # Increased slightly for better detection
ENABLE_FACE_TRACKING = True
USE_THREADING = True
MAX_FPS = 50  # Reduced to 25 for stable playback

# Stream buffer control for low latency
RTSP_BUFFER_SIZE = 1  # Minimize buffer to reduce latency
FRAME_PROCESSING_TIMEOUT = 0.001  # Faster timeout

# Aesthetic drawing settings - INCREASED FONT SIZES FOR BETTER VISIBILITY
FACE_PALETTE = [
    (72, 187, 99),    # green
    (255, 165, 0),    # orange
    (203, 192, 255),  # lavender
    (147, 255, 255),  # yellow
    (255, 144, 238),  # pink
    (128, 255, 203),  # mint
    (185, 218, 255),  # peach
]
UNKNOWN_COLOR = (80, 80, 255)    # soft red
BOX_THICKNESS = 3  # Increased from 2
CORNER_RADIUS = 10  # Increased from 8
LABEL_PADDING = 10  # Increased from 6
FONT_SCALE_LABEL = 0.8  # Increased from 0.55
FONT_SCALE_SUB = 0.65  # Increased from 0.45
FONT_THICKNESS = 2  # Increased from 1

# Confidence threshold for displaying bounding boxes (only show if >= 45%)
DISPLAY_CONFIDENCE_THRESHOLD = 0.55

# AntelopeV2 model paths
ANTELOPEV2_PATH = r"D:\Antelope\antelopev2\antelopev2"
DETECTION_MODEL = os.path.join(ANTELOPEV2_PATH, "scrfd_10g_bnkps.onnx")
RECOGNITION_MODEL = os.path.join(ANTELOPEV2_PATH, "glintr100.onnx")
GENDER_AGE_MODEL = os.path.join(ANTELOPEV2_PATH, "genderage.onnx")

# Debug directory
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_face_detection")
os.makedirs(DEBUG_DIR, exist_ok=True)


def _resize_for_detection(frame, max_width=None):
    """Shrink frame for fast detection only."""
    if max_width is None:
        max_width = DETECTION_RESIZE_WIDTH
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / w
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, w / new_w, h / new_h


def _scale_bbox(bbox, sx, sy):
    """Scale bbox from detection coords to full-res coords."""
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _iou_bbox(b1, b2):
    """Intersection-over-union of two bboxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match_tracks_to_detections(tracks, detections, frame_w, frame_h):
    """Match detections to tracks by IoU."""
    for t in tracks:
        t["frames_since_match"] = t.get("frames_since_match", 0) + 1

    display_list = []
    used_track_idx = set()
    for det in detections:
        # Skip if confidence is below display threshold
        if det.get("confidence", 0.0) < DISPLAY_CONFIDENCE_THRESHOLD:
            continue
            
        best_iou = TRACK_IOU_THRESHOLD
        best_ti = -1
        for ti, t in enumerate(tracks):
            if ti in used_track_idx:
                continue
            iou = _iou_bbox(det["bbox"], t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_ti >= 0:
            t = tracks[best_ti]
            used_track_idx.add(best_ti)
            t["bbox"] = det["bbox"]
            t["frames_since_match"] = 0
            t["match_count"] = t.get("match_count", 0) + 1
            if t["match_count"] % TRACK_REFRESH_NAME_EVERY == 1:
                t["name"] = det["name"]
                t["confidence"] = det["confidence"]
            # Only add to display if confidence meets threshold
            if t["confidence"] >= DISPLAY_CONFIDENCE_THRESHOLD:
                display_list.append({"bbox": t["bbox"], "name": t["name"], "confidence": t["confidence"]})
        else:
            next_id = max([t.get("id", 0) for t in tracks], default=0) + 1
            tracks.append({
                "id": next_id, "bbox": det["bbox"], "name": det["name"], "confidence": det["confidence"],
                "frames_since_match": 0, "match_count": 1
            })
            # Only add to display if confidence meets threshold
            if det["confidence"] >= DISPLAY_CONFIDENCE_THRESHOLD:
                display_list.append({"bbox": det["bbox"], "name": det["name"], "confidence": det["confidence"]})

    tracks[:] = [t for t in tracks if t.get("frames_since_match", 0) <= TRACK_MAX_AGE_FRAMES]
    return display_list


_CACHE_VERSION = "v8_antelopev2_final"

def _cache_path(csv_path, image_base_path):
    """Cache file path based on config."""
    key = f"{_CACHE_VERSION}|{csv_path}|{image_base_path}|{MODEL_NAME}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), f"{CACHE_FILENAME}.{h}")


def _color_for_identity(name):
    """Stable color for a given name."""
    if name is None:
        return UNKNOWN_COLOR
    idx = hash(name) % len(FACE_PALETTE)
    return FACE_PALETTE[idx]


def _draw_rounded_rect(img, x1, y1, x2, y2, color, thickness, radius=None):
    """Draw a rounded-rectangle bounding box."""
    if radius is None:
        radius = CORNER_RADIUS
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    radius = min(radius, w // 2, h // 2, 12)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


def _draw_bbox_with_label(img, x1, y1, x2, y2, label, sublabel, color, rounded=True):
    """Draw professional bbox + filled label bar above it with black text for visibility."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, FONT_THICKNESS)
    tw2, th2 = 0, 0
    if sublabel:
        (tw2, th2), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SUB, FONT_THICKNESS)
    label_w = max(tw, tw2) + LABEL_PADDING * 2
    label_h = th + (th2 + LABEL_PADDING if sublabel else 0) + LABEL_PADDING
    ly1 = max(0, y1 - label_h - 4)
    ly2 = y1 - 4
    lx1 = max(0, min(x1, img.shape[1] - label_w))
    lx2 = min(img.shape[1], lx1 + label_w)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, BOX_THICKNESS)
    # Use white text with black outline for better visibility on any background
    # Draw black outline first
    cv2.putText(img, label, (lx1 + LABEL_PADDING, ly1 + th + LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
    # Draw white text on top
    cv2.putText(img, label, (lx1 + LABEL_PADDING, ly1 + th + LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_LABEL, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    if sublabel:
        # Draw black outline first
        cv2.putText(img, sublabel, (lx1 + LABEL_PADDING, ly1 + th + LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SUB, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
        # Draw white text on top
        cv2.putText(img, sublabel, (lx1 + LABEL_PADDING, ly1 + th + LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_SUB, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    if rounded:
        _draw_rounded_rect(img, x1, y1, x2, y2, color, BOX_THICKNESS)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)


class AttendanceSystem:
    """Production attendance system using InsightFace AntelopeV2."""
    
    def __init__(self,
                 csv_path="D:\\ardiyan\\ardiyan\\uploads\\autofacedata.csv",
                 image_base_path="D:\\ardiyan\\ardiyan"):
        
        self.csv_path = csv_path
        self.image_base_path = image_base_path
        self.device_temp_dir = os.path.join(image_base_path, 'uploads', 'device-temp-image')
        self.model_name = "antelopev2"
        
        # InsightFace components
        self.face_app = None
        self.known_embeddings = []
        self.known_names = []
        self.known_rollnos = []
        
        # Cache for embeddings matrix (for faster matching)
        self.known_embeddings_matrix = None
        
        # Threading components - MODIFIED: Simplified for better performance
        self.latest_frame = None  # Track latest frame from reader
        self.latest_results = []  # Track latest results
        self.frame_lock = threading.Lock()  # Add lock for thread safety
        self.reader_thread = None
        self.processor_thread = None
        self.stop_processing = False
        self.rtsp_url = None
        
        self._load_employee_data()
        self._init_antelopev2_direct()
        self._ensure_embeddings()
    
    def _init_antelopev2_direct(self):
        """Initialize AntelopeV2 directly using downloaded models."""
        if not INSIGHTFACE_AVAILABLE:
            st.error("ERROR: insightface not installed. Run: pip install insightface onnxruntime")
            sys.exit(1)
        
        with st.spinner("Initializing AntelopeV2 model..."):
            # Check if model files exist
            if not os.path.exists(DETECTION_MODEL):
                st.error(f"Detection model not found: {DETECTION_MODEL}")
                sys.exit(1)
            
            if not os.path.exists(RECOGNITION_MODEL):
                st.error(f"Recognition model not found: {RECOGNITION_MODEL}")
                sys.exit(1)
            
            # Set environment variable
            os.environ['INSIGHTFACE_MODELS_DIR'] = ANTELOPEV2_PATH
            
            # Manual loading (direct approach)
            try:
                # Load detection model
                det_model = get_model(DETECTION_MODEL)
                det_model.prepare(ctx_id=0, input_size=INSIGHTFACE_DET_SIZE)
                if hasattr(det_model, 'threshold'):
                    det_model.threshold = INSIGHTFACE_DET_THRESH
                
                # Load recognition model
                rec_model = get_model(RECOGNITION_MODEL)
                rec_model.prepare(ctx_id=0)
                
                genderage_model = None
                if os.path.exists(GENDER_AGE_MODEL):
                    genderage_model = get_model(GENDER_AGE_MODEL)
                    genderage_model.prepare(ctx_id=0)
                
                # Create wrapper
                class SimpleInsightFace:
                    def __init__(self, det_model, rec_model, genderage_model=None, det_thresh=0.3):
                        self.det_model = det_model
                        self.rec_model = rec_model
                        self.genderage_model = genderage_model
                        self.det_thresh = det_thresh
                        
                    def get(self, img):
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            if img.dtype != np.uint8:
                                img = img.astype(np.uint8)
                        
                        try:
                            bboxes, kpss = self.det_model.detect(img)
                        except:
                            bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh)
                        
                        if bboxes is None or bboxes.shape[0] == 0:
                            return []
                        
                        if bboxes.shape[1] >= 5:
                            mask = bboxes[:, 4] >= self.det_thresh
                            bboxes = bboxes[mask]
                            if kpss is not None:
                                kpss = kpss[mask]
                        
                        if bboxes.shape[0] == 0:
                            return []
                        
                        faces = []
                        for i in range(bboxes.shape[0]):
                            bbox = bboxes[i][:4]
                            kps = kpss[i] if kpss is not None else None
                            
                            from insightface.app.common import Face
                            face = Face(bbox=bbox, kps=kps)
                            face.det_score = bboxes[i][4] if bboxes.shape[1] >= 5 else 1.0
                            
                            # Get embedding
                            self.rec_model.get(img, face)
                            
                            faces.append(face)
                        
                        return faces
                
                self.face_app = SimpleInsightFace(det_model, rec_model, genderage_model, INSIGHTFACE_DET_THRESH)
                st.success("✓ AntelopeV2 initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize AntelopeV2: {e}")
                sys.exit(1)
    
    def _load_employee_data(self):
        """Load employee data from CSV."""
        if not os.path.exists(self.csv_path):
            st.error(f"CSV not found: {self.csv_path}")
            sys.exit(1)
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            self.df = pd.read_csv(self.csv_path, encoding="utf-8", encoding_errors="replace")
        st.info(f"✓ Loaded {len(self.df)} employees from {os.path.basename(self.csv_path)}")
        
        # Print detailed employee list for debugging
        st.write("### DEBUG: Employee List from CSV")
        employee_list = []
        for idx, row in self.df.iterrows():
            employee_list.append(f"Name: {row['person_name']}, Path: {row['file_path']}")
        st.write("\n".join(employee_list))
    
    def get_full_image_path(self, file_path):
        """Path used for live device uploads."""
        filename = os.path.basename(file_path)
        return os.path.normpath(os.path.join(self.device_temp_dir, filename))
    
    def _parse_file_paths(self, file_path):
        """Parse comma-separated file paths."""
        if not file_path or (isinstance(file_path, float) and np.isnan(file_path)):
            return []
        s = str(file_path).strip()
        if not s:
            return []
        return [p.strip() for p in s.split(",") if p.strip()]
    
    def resolve_image_path(self, file_path):
        """Resolve a single employee image path."""
        if not file_path or (isinstance(file_path, float) and np.isnan(file_path)):
            return None
        
        file_path = str(file_path).strip()
        if not file_path:
            return None
        
        direct_path = os.path.join(self.image_base_path, file_path)
        if os.path.exists(direct_path):
            return os.path.normpath(direct_path)
        
        filename = os.path.basename(file_path)
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        shortnew_dir = os.path.join(self.image_base_path, 'uploads', 'shortnew')
        
        candidates = [
            os.path.normpath(os.path.join(self.device_temp_dir, filename)),
            os.path.normpath(os.path.join(shortnew_dir, filename)),
            os.path.normpath(os.path.join(csv_dir, file_path)),
            os.path.normpath(os.path.join(csv_dir, filename)),
            os.path.normpath(os.path.join(self.image_base_path, filename)),
            os.path.normpath(file_path),
        ]
        
        for p in candidates:
            if p and os.path.exists(p) and os.path.isfile(p):
                return p
        
        return None
    
    def resolve_all_image_paths(self, file_path):
        """Parse comma-separated file_path and resolve each to full path."""
        paths = self._parse_file_paths(file_path)
        resolved = []
        for p in paths:
            full = self.resolve_image_path(p)
            if full and full not in resolved:
                resolved.append(full)
        return resolved
    
    def upscale_if_small(self, img, min_side=None):
        """Upscale image if too small."""
        if min_side is None:
            min_side = MIN_SIDE_FOR_EMBEDDING
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) >= min_side:
            return img
        scale = min_side / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def _get_embedding_insightface(self, img):
        """Get face embedding using InsightFace AntelopeV2."""
        if self.face_app is None:
            return None
        
        try:
            # Convert BGR to RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img
            
            # Detect and get faces
            faces = self.face_app.get(rgb_img)
            
            if not faces or len(faces) == 0:
                return None
            
            # Get the best face (largest area)
            best_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            embedding = best_face.normed_embedding
            
            if embedding is None or len(embedding) == 0:
                return None
            
            emb = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            return emb
            
        except Exception as e:
            return None
    
    def _generate_embeddings_insightface(self):
        """Build InsightFace embeddings for ALL employee images with detailed logging."""
        embeddings = []
        names = []
        rollnos = []
        
        total_processed = 0
        total_faces_detected = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in self.df.iterrows():
            name = row['person_name']
            file_path = row['file_path']
            full_paths = self.resolve_all_image_paths(file_path)
            
            status_text.text(f"Processing: {name} - {len(full_paths)} images")
            st.write(f"DEBUG: Processing {name} with {len(full_paths)} images")
            
            if not full_paths:
                st.write(f"DEBUG: No image paths found for {name}")
                continue
            
            n_ok = 0
            n_fail = 0
            
            # Process ALL images
            for i, full_path in enumerate(full_paths):
                st.write(f"DEBUG: Reading image: {full_path}")
                img = cv2.imread(full_path)
                if img is None:
                    st.write(f"DEBUG: Failed to read image: {full_path}")
                    n_fail += 1
                    continue
                
                img = self.upscale_if_small(img)
                emb = self._get_embedding_insightface(img)
                
                if emb is not None:
                    embeddings.append(emb)
                    names.append(name)
                    rollnos.append(name)  # Use name as rollno since there's no rollno column
                    n_ok += 1
                    total_faces_detected += 1
                    st.write(f"DEBUG: Successfully generated embedding for {name} (image {i+1})")
                else:
                    n_fail += 1
                    st.write(f"DEBUG: Failed to generate embedding for {name} (image {i+1})")
                
                total_processed += 1
                progress_bar.progress(min(1.0, (idx + i/len(full_paths)) / len(self.df)))
            
            status_text.text(f"✓ {name}: {n_ok} successful, {n_fail} failed")
            st.write(f"DEBUG: {name} completed - {n_ok} embeddings generated")
        
        progress_bar.empty()
        status_text.empty()
        
        st.write(f"DEBUG: Total embeddings generated: {len(embeddings)}")
        return embeddings, names, rollnos
    
    def _ensure_embeddings(self):
        """Load embeddings from cache or generate new ones, appending for new employees."""
        cache_file = _cache_path(self.csv_path, self.image_base_path)
        
        st.write("### DEBUG: Starting _ensure_embeddings")
        st.write(f"DEBUG: Cache file path: {cache_file}")
        
        # Get current employees from CSV
        current_employees = set()
        current_employee_names = {}
        for idx, row in self.df.iterrows():
            name = row['person_name']
            current_employees.add(name)
            current_employee_names[name] = name
        
        st.write(f"DEBUG: Current employees in CSV: {len(current_employees)}")
        st.write(f"DEBUG: Employee names: {current_employees}")
        
        # Try to load existing cache
        cached_embeddings = []
        cached_names = []
        cached_rollnos = []
        cache_exists = False
        
        if os.path.exists(cache_file):
            st.write("DEBUG: Cache file exists, loading...")
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if (data.get('model') == self.model_name and
                        len(data.get('names', [])) > 0):
                    cached_embeddings = data['embeddings']
                    cached_names = data['names']
                    cached_rollnos = data['rollnos']
                    cache_exists = True
                    st.write(f"DEBUG: Loaded {len(cached_embeddings)} cached embeddings")
                    st.write(f"DEBUG: Cached names: {set(cached_names)}")
                else:
                    st.write("DEBUG: Cache model mismatch or empty")
            except Exception as e:
                st.warning(f"Cache load error: {e}")
                st.write(f"DEBUG: Cache load exception: {str(e)}")
        else:
            st.write("DEBUG: Cache file does not exist")
        
        # Find which employees are already in cache
        cached_employees = set(cached_names) if cached_names else set()
        st.write(f"DEBUG: Employees in cache: {cached_employees}")
        
        new_employees = current_employees - cached_employees
        st.write(f"DEBUG: New employees to add: {new_employees}")
        
        # Also check for employees that might have been renamed or have different embeddings
        # Force regeneration for any employee in CSV but not properly embedded
        missing_embeddings = []
        for name in current_employees:
            if name not in cached_employees:
                missing_embeddings.append(name)
            else:
                # Check if the cached embeddings for this employee actually exist
                emp_indices = [i for i, n in enumerate(cached_names) if n == name]
                if not emp_indices:
                    missing_embeddings.append(name)
        
        if missing_embeddings:
            st.write(f"DEBUG: Employees missing embeddings: {missing_embeddings}")
            new_employees = set(missing_embeddings)
        
        if not new_employees and cache_exists:
            # No new employees, just use cached data
            st.write("DEBUG: No new employees found, using cached data")
            self.known_embeddings = cached_embeddings
            self.known_names = cached_names
            self.known_rollnos = cached_rollnos
            self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)
            unique_names = set(self.known_names)
            st.info(f"✓ Loaded {len(self.known_embeddings)} cached embeddings for {len(unique_names)} employees")
            st.write(f"DEBUG: Final - {len(unique_names)} unique employees")
            return
        
        # Generate embeddings for new employees only
        if new_employees:
            st.info(f"Found {len(new_employees)} new employee(s). Generating embeddings...")
            st.write(f"DEBUG: New employees to process: {new_employees}")
            
            new_embeddings = []
            new_names = []
            new_rollnos = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process only new employees
            new_employee_rows = []
            for idx, row in self.df.iterrows():
                name = row['person_name']
                if name in new_employees:
                    new_employee_rows.append((idx, row))
                    st.write(f"DEBUG: Added {name} for processing")
            
            st.write(f"DEBUG: Total new employee rows to process: {len(new_employee_rows)}")
            
            for progress_idx, (idx, row) in enumerate(new_employee_rows):
                name = row['person_name']
                file_path = row['file_path']
                full_paths = self.resolve_all_image_paths(file_path)
                
                status_text.text(f"Processing new employee: {name} - {len(full_paths)} images")
                st.write(f"DEBUG: Processing {name} with {len(full_paths)} images from path: {file_path}")
                
                if not full_paths:
                    st.write(f"DEBUG: No valid image paths found for {name}")
                    continue
                
                n_ok = 0
                n_fail = 0
                
                for i, full_path in enumerate(full_paths):
                    st.write(f"DEBUG: Reading image {i+1}/{len(full_paths)}: {full_path}")
                    img = cv2.imread(full_path)
                    if img is None:
                        st.write(f"DEBUG: Failed to read image: {full_path}")
                        n_fail += 1
                        continue
                    
                    st.write(f"DEBUG: Image read successfully, shape: {img.shape}")
                    img = self.upscale_if_small(img)
                    emb = self._get_embedding_insightface(img)
                    
                    if emb is not None:
                        new_embeddings.append(emb)
                        new_names.append(name)
                        new_rollnos.append(name)
                        n_ok += 1
                        st.write(f"DEBUG: Successfully generated embedding for {name} (image {i+1})")
                    else:
                        n_fail += 1
                        st.write(f"DEBUG: Failed to generate embedding for {name} (image {i+1})")
                    
                    progress_bar.progress((progress_idx + i/len(full_paths)) / len(new_employee_rows))
                
                status_text.text(f"✓ {name}: {n_ok} successful, {n_fail} failed")
                st.write(f"DEBUG: {name} completed - {n_ok} embeddings generated")
            
            progress_bar.empty()
            status_text.empty()
            
            st.write(f"DEBUG: New embeddings generated: {len(new_embeddings)}")
            
            # Combine cached and new embeddings
            if cache_exists and len(cached_embeddings) > 0:
                st.write(f"DEBUG: Combining {len(cached_embeddings)} cached embeddings with {len(new_embeddings)} new embeddings")
                self.known_embeddings = cached_embeddings + new_embeddings
                self.known_names = cached_names + new_names
                self.known_rollnos = cached_rollnos + new_rollnos
            else:
                st.write(f"DEBUG: No cache exists, using only new embeddings ({len(new_embeddings)})")
                self.known_embeddings = new_embeddings
                self.known_names = new_names
                self.known_rollnos = new_rollnos
            
            # Pre-compute matrix for faster matching
            if len(self.known_embeddings) > 0:
                self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)
            else:
                self.known_embeddings_matrix = None
            
            unique_names = set(self.known_names)
            st.success(f"✓ Generated/Updated {len(self.known_embeddings)} total embeddings for {len(unique_names)} employees")
            st.write(f"DEBUG: Final - {len(unique_names)} unique employees")
            st.write(f"DEBUG: Employee names in final set: {unique_names}")
            
            # Save updated cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'model': self.model_name,
                        'embeddings': self.known_embeddings,
                        'names': self.known_names,
                        'rollnos': self.known_rollnos,
                    }, f)
                st.success(f"✓ Embeddings cached to {os.path.basename(cache_file)}")
                st.write(f"DEBUG: Cache saved with {len(self.known_embeddings)} embeddings")
            except Exception as e:
                st.warning(f"Cache save error: {e}")
                st.write(f"DEBUG: Cache save exception: {str(e)}")
        else:
            # Fallback: generate all embeddings if no cache exists
            st.info("Building embeddings from ALL images...")
            st.write("DEBUG: No cache and no new employees identified, generating all embeddings")
            self.known_embeddings, self.known_names, self.known_rollnos = self._generate_embeddings_insightface()
            if len(self.known_embeddings) > 0:
                self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)
            else:
                self.known_embeddings_matrix = None
            unique_names = set(self.known_names)
            st.success(f"✓ Generated {len(self.known_embeddings)} embeddings for {len(unique_names)} employees")
            st.write(f"DEBUG: Generated {len(self.known_embeddings)} embeddings for {len(unique_names)} employees")
            
            if len(self.known_embeddings) == 0:
                st.warning("WARNING: No embeddings were generated!")
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'model': self.model_name,
                        'embeddings': self.known_embeddings,
                        'names': self.known_names,
                        'rollnos': self.known_rollnos,
                    }, f)
                st.success(f"✓ Embeddings cached to {os.path.basename(cache_file)}")
                st.write(f"DEBUG: Cache saved with {len(self.known_embeddings)} embeddings")
            except Exception as e:
                st.warning(f"Cache save error: {e}")
                st.write(f"DEBUG: Cache save exception: {str(e)}")
    
    def _match_embedding_fast(self, query_emb, threshold):
        """Fast matching using pre-computed matrix."""
        if query_emb is None or self.known_embeddings_matrix is None or len(self.known_embeddings) == 0:
            return None, 0.0
        
        sims = self.known_embeddings_matrix @ query_emb
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        
        if best_score > threshold:
            return self.known_names[best_idx], best_score
        return None, best_score
    
    def _match_embedding(self, query_emb):
        """Match query embedding against known embeddings (legacy method)."""
        if query_emb is None or len(self.known_embeddings) == 0:
            return None, 0.0
        
        known_matrix = np.array(self.known_embeddings, dtype=np.float32)
        sims = known_matrix @ query_emb
        
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        
        if best_score > RECOGNITION_THRESHOLD:
            return self.known_names[best_idx], best_score
        return None, best_score
    
    def recognize_face_from_image(self, image):
        """Recognize face from uploaded image."""
        if isinstance(image, np.ndarray):
            img = image
        else:
            # Convert PIL to numpy
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        emb = self._get_embedding_insightface(img)
        if emb is not None:
            name, score = self._match_embedding(emb)
            return name, score
        return None, 0.0
    
    def _reader_thread_func(self, rtsp_url, threshold):
        """Dedicated reader thread: only reads RTSP frames, no processing."""
        # Set FFmpeg options for low latency
        prev = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
            
            if not cap.isOpened():
                st.error(f"Could not open RTSP stream: {rtsp_url}")
                return
            
            while not self.stop_processing:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Update latest frame (thread-safe)
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                else:
                    time.sleep(0.001)
            
            cap.release()
        except Exception as e:
            st.error(f"Reader thread error: {e}")
        finally:
            if prev is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    
    def _processor_thread_func(self, threshold):
        """Processor thread: takes latest frame and processes it."""
        frame_index = 0
        
        while not self.stop_processing:
            # Get the latest frame (thread-safe)
            with self.frame_lock:
                frame = self.latest_frame
                if frame is not None:
                    frame = frame.copy()
            
            if frame is None:
                time.sleep(0.005)
                continue
            
            frame_index += 1
            
            # Skip frames based on FRAME_SKIP
            if frame_index % FRAME_SKIP != 0:
                continue
            
            # Process the frame
            processed_frame, results = self.process_frame(frame, threshold)
            
            # Update results (thread-safe)
            with self.frame_lock:
                self.latest_results = results
    
    def process_frame(self, frame, threshold):
        """Process a single frame for face detection and recognition (optimized)."""
        if frame is None or self.face_app is None:
            return frame, []
        
        # Make a copy for drawing
        display = frame.copy()
        
        # Resize for faster detection (much smaller)
        small, sx, sy = _resize_for_detection(frame, max_width=DETECTION_RESIZE_WIDTH)
        
        try:
            # Convert to RGB for InsightFace
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_small)
            
            results = []
            for face in faces:
                bbox_small = face.bbox.astype(int).tolist()
                bbox_full = _scale_bbox(bbox_small, sx, sy)
                embedding = face.normed_embedding
                
                if embedding is not None:
                    emb = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    # Use fast matching with pre-computed matrix
                    name, confidence = self._match_embedding_fast(emb, threshold)
                else:
                    name, confidence = None, 0.0
                
                results.append({
                    "bbox": bbox_full,
                    "name": name,
                    "confidence": float(confidence),
                })
                
                # Only draw bounding box if confidence is above 45%
                if confidence >= DISPLAY_CONFIDENCE_THRESHOLD:
                    # Draw on display frame
                    x1, y1, x2, y2 = bbox_full
                    color = _color_for_identity(name)
                    label = name if name else "Unknown"
                    sublabel = f"{confidence:.0%}"
                    _draw_bbox_with_label(display, x1, y1, x2, y2, label, sublabel, color, rounded=True)
            
            return display, results
            
        except Exception as e:
            return display, []
    
    def get_latest_results(self):
        """Get the latest processed results."""
        with self.frame_lock:
            return self.latest_results
    
    def get_latest_frame_for_display(self):
        """Get the latest frame for display (with processing overlay)."""
        with self.frame_lock:
            # We need to return the processed frame - but since processor doesn't store it,
            # we need to process on the fly for display
            if self.latest_frame is not None and self.latest_results:
                # Re-draw results on latest frame
                frame_copy = self.latest_frame.copy()
                for result in self.latest_results:
                    if result.get('confidence', 0.0) >= DISPLAY_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = result['bbox']
                        name = result['name']
                        confidence = result['confidence']
                        color = _color_for_identity(name)
                        label = name if name else "Unknown"
                        sublabel = f"{confidence:.0%}"
                        _draw_bbox_with_label(frame_copy, x1, y1, x2, y2, label, sublabel, color, rounded=True)
                return frame_copy
            return self.latest_frame
    
    def start_processing_thread(self, threshold, rtsp_url):
        """Start separate threads for reading and processing."""
        self.stop_processing = False
        self.rtsp_url = rtsp_url
        self.latest_frame = None
        self.latest_results = []
        
        # Start reader thread (only reads RTSP)
        self.reader_thread = threading.Thread(
            target=self._reader_thread_func, 
            args=(rtsp_url, threshold), 
            daemon=True
        )
        self.reader_thread.start()
        
        # Start processor thread (processes frames)
        self.processor_thread = threading.Thread(
            target=self._processor_thread_func, 
            args=(threshold,), 
            daemon=True
        )
        self.processor_thread.start()
    
    def stop_processing_thread(self):
        """Stop both reader and processor threads."""
        self.stop_processing = True
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)


def main():
    st.set_page_config(
        page_title="Face Attendance System",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Recognition System")
    st.markdown("---")
    
    # Initialize the attendance system
    @st.cache_resource
    def init_system():
        CSV_PATH = "D:\\ardiyan\\ardiyan\\uploads\\autofacedata.csv"
        IMAGE_BASE_PATH = "D:\\ardiyan\\ardiyan"
        return AttendanceSystem(csv_path=CSV_PATH, image_base_path=IMAGE_BASE_PATH)
     
    try:
        system = init_system()
        st.success("✅ System initialized successfully!")
        
        # Display system info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Employees Loaded", len(set(system.known_names)))
        with col2:
            st.metric("Total Embeddings", len(system.known_embeddings))
        with col3:
            st.metric("Recognition Threshold", f"{RECOGNITION_THRESHOLD}")
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["📸 Single Image Recognition", "🎥 Live RTSP Stream"],
            horizontal=True
        )
        
        if mode == "📸 Single Image Recognition":
            st.subheader("Upload an image for face recognition")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width="stretch")
                
                if st.button("🔍 Recognize Face", type="primary"):
                    with st.spinner("Processing image..."):
                        name, confidence = system.recognize_face_from_image(image)
                        
                        if name:
                            st.success(f"### ✅ Recognized: **{name}**")
                            st.metric("Confidence", f"{confidence:.2%}")
                        else:
                            st.error(f"### ❌ Not Recognized")
                            st.metric("Best Match Score", f"{confidence:.2%}")
        
        else:  # Live RTSP Stream
            st.subheader("Live RTSP Stream Face Recognition")
            
            # RTSP URL input
            rtsp_url = st.text_input(
                "RTSP URL",
                value=RTSP_URL,
                help="Enter the RTSP stream URL"
            )
            
            # REMOVED: Recognition Threshold, Face Detection Threshold, and Performance Mode sliders
            
            start_button = st.button("🎥 Start Stream", type="primary")
            stop_button = st.button("⏹️ Stop Stream")
            
            # Video display placeholder
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            fps_placeholder = st.empty()
            resolution_placeholder = st.empty()
            
            if start_button:
                # Build URL with TCP if specified
                url = rtsp_url
                if RTSP_USE_TCP:
                    if '?' in url:
                        url = url + "&rtsp_transport=tcp"
                    else:
                        url = url + "?rtsp_transport=tcp"
                
                status_placeholder.info("🟢 Stream is running... Press 'Stop Stream' to end.")
                
                # Start processing threads with default threshold
                system.start_processing_thread(RECOGNITION_THRESHOLD, url)
                
                # Performance tracking
                frame_count = 0
                last_time = time.time()
                display_fps = 0
                last_display_time = 0
                frame_interval = 1.0 / MAX_FPS
                
                # Stream processing loop - OPTIMIZED FOR SMOOTH PLAYBACK
                while not stop_button:
                    loop_start = time.time()
                    
                    # Get latest processed results
                    results = system.get_latest_results()
                    
                    # Get frame for display (with overlay)
                    display_frame = system.get_latest_frame_for_display()
                    
                    if display_frame is not None:
                        frame_count += 1
                        
                        # Resize for display if needed
                        if display_frame.shape[1] > DISPLAY_RESIZE_WIDTH:
                            scale = DISPLAY_RESIZE_WIDTH / display_frame.shape[1]
                            new_w = int(display_frame.shape[1] * scale)
                            new_h = int(display_frame.shape[0] * scale)
                            display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Convert BGR to RGB for Streamlit
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        
                        # Throttle display updates to avoid overwhelming Streamlit
                        current_time = time.time()
                        if current_time - last_display_time >= 0.033:  # ~30 FPS max display
                            frame_placeholder.image(display_frame_rgb, channels="RGB", width="stretch")
                            last_display_time = current_time
                        
                        # Calculate FPS
                        if current_time - last_time >= 1.0:
                            display_fps = frame_count
                            frame_count = 0
                            last_time = current_time
                            fps_placeholder.metric("Display FPS", f"{display_fps} fps")
                        
                        # Update status every ~2 seconds to reduce UI updates
                        if int(current_time) % 2 == 0:
                            # Only count results that meet display threshold for status
                            if results:
                                valid_results = [r for r in results if r.get('confidence', 0.0) >= DISPLAY_CONFIDENCE_THRESHOLD]
                                recognized = [r for r in valid_results if r['name'] is not None]
                                status_placeholder.info(f"👥 Faces: {len(valid_results)} | ✅ Recognized: {len(recognized)}")
                            else:
                                status_placeholder.info("🔍 No faces detected")
                    
                    # Precise frame rate limiting for smooth playback
                    elapsed = time.time() - loop_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0.002:  # Sleep only if significant
                        time.sleep(sleep_time)
                
                # Cleanup
                system.stop_processing_thread()
                frame_placeholder.empty()
                status_placeholder.empty()
                fps_placeholder.empty()
                resolution_placeholder.empty()
                st.success("🟢 Stream stopped.")
    
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.info("Please check your CSV file path and AntelopeV2 model paths.")


if __name__ == "__main__":
    main()