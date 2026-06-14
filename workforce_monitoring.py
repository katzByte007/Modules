"""Workforce Monitoring module — person count, PPE, pose consciousness, no-manpower alerts."""

import os
import json
import time
import threading
import logging
from datetime import datetime
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ffmpeg_env_lock = threading.Lock()

WORKFORCE_MACHINE_IDS = [f'MACHINE-{i:02d}' for i in range(1, 7)]
NO_MANPOWER_THRESHOLD_SEC = 300  # 5 minutes
WORKFORCE_STREAM_MAX_WIDTH = 854   # ~480p for smooth streaming / less memory
WORKFORCE_DETECT_MAX_WIDTH = 512   # smaller YOLO input for faster inference
INFERENCE_INTERVAL = 1.0    # detection ~1 fps — keeps video smooth
DISPLAY_INTERVAL = 0.04     # ~25 fps video refresh
_wf_infer_semaphore = threading.Semaphore(2)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    mp = None
    _MP_AVAILABLE = False
    logger.warning('mediapipe not installed — consciousness detection disabled. pip install mediapipe')

# Global state (managed by this module)
workforce_video_readers = {}   # machine_id -> VideoFileReader
workforce_procs = {}           # machine_id -> WorkforceMonitoringProcessor
_workforce_pose = None
_workforce_pose_lock = threading.Lock()


def _resize_frame_max(frame, max_w):
    """Downscale frames for streaming and lighter processing."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    nh = max(1, int(h * max_w / w))
    return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)


def _resize_for_detect(frame, max_w=WORKFORCE_DETECT_MAX_WIDTH):
    """Return a smaller copy for YOLO plus scale factor (full / small width)."""
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    scale = max_w / float(w)
    nh = max(1, int(h * scale))
    small = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_LINEAR)
    return small, scale


def _scale_boxes(boxes, scale):
    """Map detection boxes from resized inference frame back to display coordinates."""
    if scale == 1.0 or not boxes:
        return boxes
    inv = 1.0 / scale
    return [[int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)]
            for x1, y1, x2, y2 in boxes]


def _get_pose_detector():
    global _workforce_pose
    if not _MP_AVAILABLE:
        return None
    with _workforce_pose_lock:
        if _workforce_pose is None:
            _workforce_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return _workforce_pose


class VideoFileReader:
    """Loops an MP4 file as a virtual camera stream."""

    def __init__(self, machine_id, video_path):
        self.machine_id = machine_id
        self.video_path = video_path
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.status = 'initializing'
        self._thread = None
        self._cap = None
        self._loop_cap = None
        self._cap_lock = threading.Lock()
        self._frame_dt = 0.033

    def _release_caps(self):
        with self._cap_lock:
            caps = []
            if self._cap is not None:
                caps.append(self._cap)
                self._cap = None
            if self._loop_cap is not None:
                caps.append(self._loop_cap)
                self._loop_cap = None
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass

    def _rewind_or_reopen(self, cap):
        """Loop short MP4 files — rewind or reopen when EOF reached."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, frame
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, frame
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        new_cap = self._open_local_capture()
        if new_cap is not None:
            ret, frame = new_cap.read()
            if ret and frame is not None:
                return new_cap, frame
            try:
                new_cap.release()
            except Exception:
                pass
        return None, None

    def start(self, preopen=True):
        if self.running:
            return
        self.running = True
        if preopen:
            cap = self._open_local_capture()
            if cap is not None:
                self._cap = cap
                self.status = 'active'
            elif not self.video_path or not os.path.isfile(self.video_path):
                self.status = 'no_video'
            else:
                self.status = 'error'
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f'WF-Video-{self.machine_id}')
        self._thread.start()

    def _open_local_capture(self):
        """Open local MP4 without RTSP ffmpeg options (avoids 30s RTSP timeouts blocking file read)."""
        if not self.video_path or not os.path.isfile(self.video_path):
            return None
        backends = [cv2.CAP_FFMPEG]
        if hasattr(cv2, 'CAP_ANY'):
            backends.append(cv2.CAP_ANY)
        prev = None
        with _ffmpeg_env_lock:
            prev = os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
        try:
            for backend in backends:
                cap = cv2.VideoCapture(self.video_path, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.frame = _resize_frame_max(frame, WORKFORCE_STREAM_MAX_WIDTH)
                        try:
                            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                            if fps > 1:
                                self._frame_dt = min(1.0 / fps, 0.033)
                        except Exception:
                            self._frame_dt = 0.033
                        logger.info(f'Video {self.machine_id}: opened {self.video_path} (backend={backend})')
                        return cap
                try:
                    cap.release()
                except Exception:
                    pass
        finally:
            with _ffmpeg_env_lock:
                if prev is not None:
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = prev
        logger.error(f'Video {self.machine_id}: cannot open {self.video_path}')
        return None

    def _loop(self):
        cap = self._cap
        self._cap = None
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    if not self.video_path or not os.path.isfile(self.video_path):
                        self.status = 'no_video'
                        time.sleep(1.0)
                        continue
                    cap = self._open_local_capture()
                    if cap is None:
                        self.status = 'error'
                        time.sleep(2.0)
                        continue
                    self.status = 'active'

                with self._cap_lock:
                    self._loop_cap = cap

                ret, frame = cap.read()
                if not ret or frame is None:
                    cap, frame = self._rewind_or_reopen(cap)
                    if cap is None or frame is None:
                        self.status = 'error'
                        time.sleep(0.5)
                        continue
                    with self._cap_lock:
                        self._loop_cap = cap

                with self.lock:
                    self.frame = _resize_frame_max(frame, WORKFORCE_STREAM_MAX_WIDTH)
                time.sleep(self._frame_dt)
            except Exception as e:
                logger.error(f'Workforce video {self.machine_id}: {e}')
                with self._cap_lock:
                    self._loop_cap = None
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                time.sleep(1.0)

        with self._cap_lock:
            self._loop_cap = None
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        self.status = 'stopped'

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self._release_caps()
        if self._thread:
            self._thread.join(timeout=5.0)


def _analyze_consciousness(frame, bbox):
    """MediaPipe pose: conscious / sleep / unconscious."""
    if not _MP_AVAILABLE:
        return 'unknown'
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 30 or y2 - y1 < 30:
        return 'unknown'
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 'unknown'
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pose = _get_pose_detector()
    if pose is None:
        return 'unknown'
    try:
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return 'unknown'
        lm = results.pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
        nose = lm[PL.NOSE]
        l_sh = lm[PL.LEFT_SHOULDER]
        r_sh = lm[PL.RIGHT_SHOULDER]
        l_hip = lm[PL.LEFT_HIP]
        r_hip = lm[PL.RIGHT_HIP]
        shoulder_y = (l_sh.y + r_sh.y) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        nose_y = nose.y
        torso_vert = hip_y - shoulder_y
        if torso_vert < 0.06:
            return 'unconscious'
        if torso_vert < 0.12 or nose_y > shoulder_y + 0.08:
            return 'sleep'
        return 'conscious'
    except Exception:
        return 'unknown'


PPE_VIOLATION_KEYWORDS = (
    'no-helmet', 'no-hardhat', 'no_hard_hat', 'no_helmet', 'no hardhat', 'no hard hat',
)


class WorkforceMonitoringProcessor:
    """Per-machine: person count, PPE helmet, pose consciousness, no-manpower timer."""

    def __init__(self, machine_id, video_reader, person_model, ppe_model, conf=0.35,
                 save_alert_fn=None, get_db_fn=None, alerts_dir=None):
        self.machine_id = machine_id
        self.video_reader = video_reader
        self.person_model = person_model
        self.ppe_model = ppe_model
        self.conf = conf
        self.save_alert_fn = save_alert_fn
        self.get_db_fn = get_db_fn
        self.alerts_dir = alerts_dir

        self.running = False
        self._display_thread = None
        self._infer_thread = None
        self.lock = threading.Lock()
        self.annotated_frame = None

        self.worker_count = 0
        self.helmet_ok_count = 0
        self.helmet_viol_count = 0
        self.conscious_states = []
        self.no_manpower_since = None
        self.no_manpower_elapsed = 0
        self.person_present_since = None
        self.person_present_elapsed = 0
        self.status = 'manpower'
        self.badge = 'manpower'
        self.alert_timer_str = None
        self.recent_alerts = deque(maxlen=20)
        self.utilization_history = deque(maxlen=96)

        self.stats = {
            'machine_id': machine_id,
            'worker_count': 0,
            'helmet_ok': 0,
            'helmet_violations': 0,
            'status': 'no-manpower',
            'badge': 'no-manpower',
            'conscious_states': [],
            'no_manpower_elapsed': 0,
            'person_present_elapsed': 0,
            'alert_timer': None,
            'video_status': 'initializing',
        }

        self._last_db_save = time.time()
        self._db_interval = 60
        self._last_alert_no_man = 0
        self._last_alert_ppe = 0
        self._last_alert_conscious = 0

        self._cache_lock = threading.Lock()
        self._person_boxes = []
        self._helmet_violations = []
        self._helmet_ok = 0
        self._conscious_states = []
        self._cached_worker_count = 0
        self._cached_status = 'no-manpower'
        self._cached_badge = 'no-manpower'
        self._cached_alert_timer = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True, name=f'WF-Display-{self.machine_id}')
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name=f'WF-Infer-{self.machine_id}')
        self._display_thread.start()
        self._infer_thread.start()

    def stop(self):
        self.running = False
        for t in (self._display_thread, self._infer_thread):
            if t:
                t.join(timeout=5.0)

    def _is_ppe_violation(self, class_name):
        nl = (class_name or '').lower()
        return any(kw in nl for kw in PPE_VIOLATION_KEYWORDS)

    def _run_inference(self, frame, now):
        person_boxes = []
        helmet_violations = []
        helmet_ok = 0
        conscious_states = []

        det_frame, det_scale = _resize_for_detect(frame)

        with _wf_infer_semaphore:
            try:
                pres = self.person_model(det_frame, conf=self.conf, verbose=False)
                if pres and len(pres) > 0 and pres[0].boxes is not None:
                    for box in pres[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            except Exception as e:
                logger.error(f'WF person detect {self.machine_id}: {e}')

            try:
                ppe_res = self.ppe_model(det_frame, conf=0.3, verbose=False)
                if ppe_res and len(ppe_res) > 0 and ppe_res[0].boxes is not None:
                    names = ppe_res[0].names or {}
                    for box in ppe_res[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        if self._is_ppe_violation(cls_name):
                            helmet_violations.append([x1, y1, x2, y2])
                        elif 'helmet' in cls_name.lower() or 'hardhat' in cls_name.lower():
                            helmet_ok += 1
            except Exception as e:
                logger.error(f'WF PPE detect {self.machine_id}: {e}')

        person_boxes = _scale_boxes(person_boxes, det_scale)
        helmet_violations = _scale_boxes(helmet_violations, det_scale)

        for pb in person_boxes[:4]:
            conscious_states.append(_analyze_consciousness(frame, pb))

        worker_count = len(person_boxes)

        if worker_count == 0:
            if self.no_manpower_since is None:
                self.no_manpower_since = now
            self.no_manpower_elapsed = int(now - self.no_manpower_since)
            self.person_present_since = None
            self.person_present_elapsed = 0
        else:
            self.no_manpower_since = None
            self.no_manpower_elapsed = 0
            if self.person_present_since is None:
                self.person_present_since = now
            self.person_present_elapsed = int(now - self.person_present_since)

        has_helmet_viol = len(helmet_violations) > 0 and worker_count > 0
        has_unconscious = any(s in ('unconscious', 'sleep') for s in conscious_states)

        if worker_count == 0:
            status, badge = 'no-manpower', 'no-manpower'
            alert_timer = self._fmt_timer(self.no_manpower_elapsed)
        elif has_helmet_viol:
            status, badge = 'helmet-warn', 'helmet-viol'
            alert_timer = '00:03'
        elif has_unconscious:
            status, badge = 'helmet-warn', 'helmet-viol'
            alert_timer = conscious_states[0] if conscious_states else None
        elif helmet_ok >= worker_count and worker_count > 0:
            status, badge = 'manpower', 'helmet-ok'
            alert_timer = '00:01'
        else:
            status, badge = 'manpower', 'manpower'
            alert_timer = None

        with self._cache_lock:
            self._person_boxes = person_boxes
            self._helmet_violations = helmet_violations
            self._helmet_ok = helmet_ok
            self._conscious_states = conscious_states
            self._cached_worker_count = worker_count
            self._cached_status = status
            self._cached_badge = badge
            self._cached_alert_timer = alert_timer

        self.worker_count = worker_count
        self.helmet_ok_count = helmet_ok
        self.helmet_viol_count = len(helmet_violations)
        self.conscious_states = conscious_states
        self.status = status
        self.badge = badge
        self.alert_timer_str = alert_timer
        self.stats = {
            'machine_id': self.machine_id,
            'worker_count': worker_count,
            'helmet_ok': helmet_ok,
            'helmet_violations': len(helmet_violations),
            'status': status,
            'badge': badge,
            'conscious_states': conscious_states,
            'no_manpower_elapsed': self.no_manpower_elapsed,
            'person_present_elapsed': self.person_present_elapsed,
            'alert_timer': alert_timer,
            'video_status': self.video_reader.status,
        }

        hour_slot = datetime.now().strftime('%H')
        self.utilization_history.append({
            'hour': hour_slot,
            'manned': 1 if worker_count > 0 else 0,
            'unmanned': 1 if worker_count == 0 else 0,
            'workers': worker_count,
        })

        self._check_alerts(frame, worker_count, helmet_violations, conscious_states, now)
        if now - self._last_db_save >= self._db_interval:
            self._save_event()
            self._last_db_save = now

    def _draw_frame(self, frame):
        with self._cache_lock:
            person_boxes = list(self._person_boxes)
            helmet_violations = list(self._helmet_violations)
            conscious_states = list(self._conscious_states)
            worker_count = self._cached_worker_count

        vis = frame.copy()
        h, w = vis.shape[:2]

        for idx, pb in enumerate(person_boxes):
            x1, y1, x2, y2 = pb
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 118), 2)
            cs = conscious_states[idx] if idx < len(conscious_states) else 'unknown'
            cv2.putText(vis, f'PERSON · {cs.upper()}', (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 118), 1)

        for vb in helmet_violations:
            x1, y1, x2, y2 = vb
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, 'NO HELMET', (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        if worker_count == 0:
            cv2.rectangle(vis, (2, 2), (w - 2, h - 2), (0, 0, 255), 2)
            cv2.putText(vis, 'NO WORKER DETECTED', (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        overlay = (f'{self.machine_id} | Workers:{worker_count} | '
                   f'HelmViol:{len(helmet_violations)}')
        cv2.putText(vis, overlay, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1)
        return vis

    def _display_loop(self):
        """Fast loop — video always plays; never waits on YOLO."""
        while self.running:
            try:
                frame = self.video_reader.get_frame()
                if frame is None:
                    self.stats['video_status'] = self.video_reader.status
                    time.sleep(0.05)
                    continue
                vis = self._draw_frame(frame)
                with self.lock:
                    self.annotated_frame = vis
                time.sleep(DISPLAY_INTERVAL)
            except Exception as e:
                logger.error(f'WF display {self.machine_id}: {e}')
                time.sleep(0.1)

    def _inference_loop(self):
        """Slow loop — YOLO runs separately so video is never blocked."""
        while self.running:
            try:
                frame = self.video_reader.get_frame()
                if frame is None:
                    time.sleep(0.4)
                    continue
                self._run_inference(frame, time.time())
                time.sleep(INFERENCE_INTERVAL)
            except Exception as e:
                logger.error(f'WF infer {self.machine_id}: {e}')
                time.sleep(0.5)

    def _fmt_timer(self, secs):
        m, s = divmod(max(0, secs), 60)
        return f'{m:02d}:{s:02d}'

    def _check_alerts(self, frame, worker_count, helmet_violations, conscious_states, now):
        if not self.save_alert_fn:
            return
        mid = self.machine_id

        if worker_count == 0 and self.no_manpower_elapsed >= NO_MANPOWER_THRESHOLD_SEC:
            if now - self._last_alert_no_man > 60:
                self._last_alert_no_man = now
                dets = [('bbox', f'No Manpower (>{NO_MANPOWER_THRESHOLD_SEC // 60} min)')]
                self.save_alert_fn(mid, 'workforce', frame, dets, severity='high',
                                   meta={'type': 'no_manpower', 'elapsed_sec': self.no_manpower_elapsed})
                self.recent_alerts.appendleft({
                    'type': 'noman', 'title': f'No Manpower (>{NO_MANPOWER_THRESHOLD_SEC // 60} min)',
                    'sub': f'{mid} · Unattended', 'time': datetime.now().strftime('%I:%M %p').lstrip('0'),
                })

        if helmet_violations and now - self._last_alert_ppe > 45:
            self._last_alert_ppe = now
            dets = [('bbox', 'PPE Violation · Helmet missing')]
            self.save_alert_fn(mid, 'workforce', frame, dets, severity='medium',
                               meta={'type': 'ppe_violation', 'count': len(helmet_violations)})
            self.recent_alerts.appendleft({
                'type': 'ppe', 'title': 'PPE Violation Detected',
                'sub': f'{mid} · Helmet missing', 'time': datetime.now().strftime('%I:%M %p').lstrip('0'),
            })

        bad_states = [s for s in conscious_states if s in ('unconscious', 'sleep')]
        if bad_states and now - self._last_alert_conscious > 60:
            self._last_alert_conscious = now
            state = bad_states[0]
            dets = [('bbox', f'Worker {state}')]
            self.save_alert_fn(mid, 'workforce', frame, dets, severity='high',
                               meta={'type': 'consciousness', 'state': state})
            self.recent_alerts.appendleft({
                'type': 'ppe', 'title': f'Worker {state.title()} Detected',
                'sub': f'{mid} · Pose alert', 'time': datetime.now().strftime('%I:%M %p').lstrip('0'),
            })

    def _save_event(self):
        if not self.get_db_fn:
            return
        try:
            conn = self.get_db_fn()
            conn.execute(
                '''INSERT INTO workforce_events
                   (machine_id, worker_count, helmet_ok, helmet_violations, status, conscious_json, person_present_sec)
                   VALUES (?,?,?,?,?,?,?)''',
                (self.machine_id, self.worker_count, self.helmet_ok_count, self.helmet_viol_count,
                 self.status, json.dumps(self.conscious_states), self.person_present_elapsed)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'WF DB save {self.machine_id}: {e}')

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None


def init_workforce_tables(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS workforce_machines (
        machine_id TEXT PRIMARY KEY,
        cam_label TEXT,
        video_path TEXT,
        enabled INTEGER DEFAULT 0,
        config_json TEXT DEFAULT '{}',
        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS workforce_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        worker_count INTEGER DEFAULT 0,
        helmet_ok INTEGER DEFAULT 0,
        helmet_violations INTEGER DEFAULT 0,
        status TEXT,
        conscious_json TEXT DEFAULT '[]',
        person_present_sec INTEGER DEFAULT 0
    )''')
    for mid in WORKFORCE_MACHINE_IDS:
        conn.execute(
            'INSERT OR IGNORE INTO workforce_machines (machine_id, cam_label, enabled) VALUES (?,?,0)',
            (mid, f'CAM-{mid.split("-")[1]}')
        )


def get_aggregate_status():
    """Dashboard KPIs and per-machine stats."""
    machines = []
    manned = unmanned = ppe_viol = 0
    all_alerts = []
    cameras_online = 0

    for mid in WORKFORCE_MACHINE_IDS:
        proc = workforce_procs.get(mid)
        reader = workforce_video_readers.get(mid)
        if proc:
            s = dict(proc.stats)
            machines.append(s)
            if s.get('worker_count', 0) > 0:
                manned += 1
            else:
                unmanned += 1
            if s.get('helmet_violations', 0) > 0:
                ppe_viol += 1
            all_alerts.extend(list(proc.recent_alerts))
        elif reader:
            machines.append({
                'machine_id': mid, 'worker_count': 0, 'status': 'no-manpower',
                'badge': 'no-manpower', 'video_status': reader.status,
            })
            unmanned += 1
        else:
            machines.append({
                'machine_id': mid, 'worker_count': 0, 'status': 'no-manpower',
                'badge': 'no-manpower', 'video_status': 'inactive',
            })
            unmanned += 1

        if reader and reader.status == 'active':
            cameras_online += 1

    total = len(WORKFORCE_MACHINE_IDS)
    ppe_compliance = 90
    if manned > 0:
        ok = sum(1 for m in machines if m.get('badge') in ('helmet-ok', 'manpower') and m.get('worker_count', 0) > 0)
        ppe_compliance = int((ok / max(manned, 1)) * 100)

    all_alerts.sort(key=lambda x: x.get('time', ''), reverse=True)

    return {
        'total_machines': total,
        'manned': manned,
        'unmanned': unmanned,
        'ppe_violations': ppe_viol,
        'cameras_online': cameras_online,
        'ppe_compliance_pct': ppe_compliance,
        'machines': machines,
        'alerts': all_alerts[:10],
    }


def get_utilization_chart(get_db_fn):
    """Last 16 hours utilization for chart."""
    try:
        conn = get_db_fn()
        rows = conn.execute(
            '''SELECT strftime('%H', timestamp) AS hr,
                      AVG(CASE WHEN worker_count > 0 THEN worker_count ELSE 0 END) AS manned,
                      AVG(CASE WHEN worker_count = 0 THEN 1 ELSE 0 END) AS unmanned
               FROM workforce_events
               WHERE timestamp >= datetime('now', '-16 hours')
               GROUP BY hr ORDER BY hr'''
        ).fetchall()
        conn.close()
        labels, data1, data2 = [], [], []
        for r in rows:
            labels.append(f'M{int(r["hr"]):02d}' if r['hr'] else 'M00')
            data1.append(round(float(r['manned'] or 0), 1))
            data2.append(round(float(r['unmanned'] or 0), 1))
        if not labels:
            labels = [f'M{i:02d}' for i in range(0, 11)]
            data1 = [3, 4, 3, 3, 4, 5, 3, 4, 2, 3, 6]
            data2 = [1, 0, 0, 1, 0, 0, 1, 2, 1, 0, 1]
        return {'labels': labels, 'manned': data1, 'unmanned': data2}
    except Exception as e:
        logger.error(f'WF chart: {e}')
        return {
            'labels': [f'M{i:02d}' for i in range(0, 11)],
            'manned': [3, 4, 3, 3, 4, 5, 3, 4, 2, 3, 6],
            'unmanned': [1, 0, 0, 1, 0, 0, 1, 2, 1, 0, 1],
        }


def get_ppe_stats(get_db_fn):
    try:
        conn = get_db_fn()
        shift_viol = conn.execute(
            '''SELECT COUNT(*) AS c FROM alert_events
               WHERE detection_type='workforce' AND alert_label LIKE '%PPE%'
               AND created_time >= datetime('now', 'start of day', '+6 hours')'''
        ).fetchone()
        today_viol = conn.execute(
            '''SELECT COUNT(*) AS c FROM alert_events
               WHERE detection_type='workforce' AND alert_label LIKE '%PPE%'
               AND created_time >= datetime('now', 'start of day')'''
        ).fetchone()
        conn.close()
        return {'shift': shift_viol['c'] if shift_viol else 0, 'today': today_viol['c'] if today_viol else 0}
    except Exception:
        return {'shift': 0, 'today': 0}


def stop_machine(machine_id):
    proc = workforce_procs.pop(machine_id, None)
    if proc:
        proc.stop()
    reader = workforce_video_readers.pop(machine_id, None)
    if reader:
        reader.stop()
    time.sleep(0.3)


def start_playback_only(machine_id, video_path):
    """Display-only loop via OpenCV — no YOLO/detection (works with mp4v and other codecs)."""
    if not video_path or not os.path.isfile(video_path):
        return False
    reader = workforce_video_readers.get(machine_id)
    if (
        reader and reader.running and reader.video_path == video_path
        and machine_id not in workforce_procs
    ):
        return True
    stop_machine(machine_id)
    reader = VideoFileReader(machine_id, video_path)
    workforce_video_readers[machine_id] = reader
    reader.start(preopen=True)
    for _ in range(30):
        if reader.get_frame() is not None and reader.status == 'active':
            logger.info(f'Workforce playback started: {machine_id} ({os.path.basename(video_path)})')
            return True
        time.sleep(0.1)
    logger.warning(f'Workforce playback slow start: {machine_id} status={reader.status}')
    return reader.status in ('active', 'initializing')


def ensure_workforce_playback(videos_dir):
    """Start lightweight file readers for all bundled workforce videos."""
    for mid in WORKFORCE_MACHINE_IDS:
        path = discover_workforce_video(mid, videos_dir)
        if path:
            try:
                start_playback_only(mid, path)
            except Exception as e:
                logger.error(f'Workforce playback {mid}: {e}')


def start_machine(machine_id, video_path, person_model, ppe_model, conf, save_alert_fn, get_db_fn, alerts_dir):
    stop_machine(machine_id)
    if not video_path or not os.path.isfile(video_path):
        logger.error(f'WF start {machine_id}: video missing at {video_path!r}')
        return False, 'No video file configured'
    reader = VideoFileReader(machine_id, video_path)
    workforce_video_readers[machine_id] = reader
    reader.start(preopen=True)
    for _ in range(25):
        if reader.get_frame() is not None and reader.status == 'active':
            break
        time.sleep(0.1)
    if reader.status not in ('active', 'initializing'):
        logger.error(f'WF start {machine_id}: video not ready (status={reader.status})')
        stop_machine(machine_id)
        return False, f'Could not open video (status={reader.status})'
    proc = WorkforceMonitoringProcessor(
        machine_id, reader, person_model, ppe_model, conf=conf,
        save_alert_fn=save_alert_fn, get_db_fn=get_db_fn, alerts_dir=alerts_dir,
    )
    workforce_procs[machine_id] = proc
    proc.start()
    return True, 'started'


def restore_workforce(get_db_fn, person_model, ppe_model, save_alert_fn, alerts_dir, videos_dir):
    try:
        for mid in list(set(list(workforce_procs.keys()) + list(workforce_video_readers.keys()))):
            if mid not in WORKFORCE_MACHINE_IDS:
                stop_machine(mid)
        conn = get_db_fn()
        rows = conn.execute('SELECT * FROM workforce_machines WHERE enabled=1').fetchall()
        conn.close()
        for row in rows:
            mid = row['machine_id']
            if mid not in WORKFORCE_MACHINE_IDS:
                continue
            vp = row['video_path']
            if vp and not os.path.isabs(vp):
                vp = os.path.join(videos_dir, vp)
            cfg = json.loads(row['config_json'] or '{}')
            conf = cfg.get('confidence', 0.35)
            if vp and os.path.isfile(vp):
                start_machine(mid, vp, person_model, ppe_model, conf, save_alert_fn, get_db_fn, alerts_dir)
                logger.info(f'Restored workforce monitoring: {mid}')
    except Exception as e:
        logger.error(f'WF restore: {e}')


def discover_workforce_video(machine_id, videos_dir):
    """Find bundled MP4 for a machine under data/workforce_videos (newest match wins)."""
    if not videos_dir or not os.path.isdir(videos_dir):
        return None
    prefix = machine_id.upper()
    matches = []
    for fname in os.listdir(videos_dir):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        if fname.upper().startswith(prefix):
            full = os.path.join(videos_dir, fname)
            if os.path.isfile(full):
                matches.append(full)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def workforce_video_version(path):
    """Cache-bust token from file modification time."""
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


def bind_workforce_videos_from_folder(get_db_fn, videos_dir):
    """Register bundled videos in DB without starting detection."""
    try:
        conn = get_db_fn()
        for mid in WORKFORCE_MACHINE_IDS:
            path = discover_workforce_video(mid, videos_dir)
            if path and os.path.isfile(path):
                rel = os.path.basename(path)
                conn.execute(
                    '''UPDATE workforce_machines SET video_path=?, updated_time=CURRENT_TIMESTAMP
                       WHERE machine_id=?''',
                    (rel, mid),
                )
            else:
                conn.execute(
                    'UPDATE workforce_machines SET video_path=NULL WHERE machine_id=?',
                    (mid,),
                )
        conn.commit()
        conn.close()
        logger.info('Workforce videos bound from %s', videos_dir)
    except Exception as e:
        logger.error(f'WF bind videos: {e}')


_DEMO_MACHINE_STATES = [
    {'machine_id': 'MACHINE-01', 'worker_count': 2, 'status': 'active', 'badge': 'helmet-ok',
     'helmet_ok': 2, 'helmet_violations': 0, 'alert_timer': ''},
    {'machine_id': 'MACHINE-02', 'worker_count': 1, 'status': 'active', 'badge': 'manpower',
     'helmet_ok': 1, 'helmet_violations': 0, 'alert_timer': ''},
    {'machine_id': 'MACHINE-03', 'worker_count': 0, 'status': 'no-manpower', 'badge': 'no-manpower',
     'helmet_violations': 0, 'alert_timer': '⏱ 4:12'},
    {'machine_id': 'MACHINE-04', 'worker_count': 1, 'status': 'helmet-warn', 'badge': 'helmet-viol',
     'helmet_ok': 0, 'helmet_violations': 1, 'alert_timer': ''},
    {'machine_id': 'MACHINE-05', 'worker_count': 2, 'status': 'active', 'badge': 'helmet-ok',
     'helmet_ok': 2, 'helmet_violations': 0, 'alert_timer': ''},
    {'machine_id': 'MACHINE-06', 'worker_count': 1, 'status': 'active', 'badge': 'manpower',
     'helmet_ok': 1, 'helmet_violations': 0, 'alert_timer': ''},
]

_DEMO_ALERTS = [
    {'type': 'noman', 'title': 'NO MANPOWER', 'sub': 'MACHINE-03 · 5 min threshold', 'time': '2m ago'},
    {'type': 'ppe', 'title': 'PPE VIOLATION', 'sub': 'MACHINE-04 · Helmet missing', 'time': '5m ago'},
    {'type': 'ppe', 'title': 'PPE VIOLATION', 'sub': 'MACHINE-01 · Helmet missing', 'time': '12m ago'},
    {'type': 'noman', 'title': 'NO MANPOWER', 'sub': 'MACHINE-06 · Cleared', 'time': '18m ago'},
]


def get_workforce_demo_status(video_map=None):
    """Autofill dashboard KPIs while live view plays raw video only."""
    video_map = video_map or {}
    machines = []
    for base in _DEMO_MACHINE_STATES:
        m = dict(base)
        m['video_status'] = 'active' if video_map.get(m['machine_id']) else 'inactive'
        m['running'] = bool(video_map.get(m['machine_id']))
        m['video_name'] = os.path.basename(video_map.get(m['machine_id']) or '') or ''
        vp = video_map.get(m['machine_id'])
        if vp:
            ver = workforce_video_version(vp)
            m['video_url'] = f"/video_feed/workforce/{m['machine_id']}?v={ver}"
        else:
            m['video_url'] = ''
        machines.append(m)

    manned = sum(1 for m in machines if m.get('worker_count', 0) > 0)
    unmanned = len(machines) - manned
    ppe_viol = sum(1 for m in machines if m.get('helmet_violations', 0) > 0)
    cameras_online = sum(1 for m in machines if m.get('video_status') == 'active')

    return {
        'total_machines': len(WORKFORCE_MACHINE_IDS),
        'manned': manned,
        'unmanned': unmanned,
        'ppe_violations': ppe_viol,
        'cameras_online': cameras_online,
        'ppe_compliance_pct': 87,
        'machines': machines,
        'alerts': list(_DEMO_ALERTS),
        'playback_only': True,
    }


def get_workforce_video_map(get_db_fn, videos_dir):
    """machine_id -> absolute path — folder scan is source of truth."""
    out = {}
    try:
        for mid in WORKFORCE_MACHINE_IDS:
            found = discover_workforce_video(mid, videos_dir)
            if found:
                out[mid] = found
                continue
            conn = get_db_fn()
            row = conn.execute(
                'SELECT video_path FROM workforce_machines WHERE machine_id=?', (mid,)
            ).fetchone()
            conn.close()
            if row and row['video_path']:
                full = row['video_path'] if os.path.isabs(row['video_path']) else os.path.join(videos_dir, row['video_path'])
                if os.path.isfile(full):
                    out[mid] = full
    except Exception as e:
        logger.error(f'WF video map: {e}')
    return out
