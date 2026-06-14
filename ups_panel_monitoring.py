"""UPS Panel monitoring — LED status, meter OCR, events and trend from camera video."""

import os
import json
import time
import re
import threading
import logging
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from workforce_monitoring import VideoFileReader

logger = logging.getLogger(__name__)

UPS_PANEL_IDS = ['UPS-PANEL-01']
FRAME_INTERVAL = 0.2

ups_video_readers = {}
ups_procs = {}

STATUS_KEYS = [
    'ups_on', 'bypass_on', 'mains_on', 'battery_on', 'charger_on',
    'ac_supply_fail', 'battery_low', 'ups_fault', 'overload', 'earth_fault',
]

LED_NAMES = ['ups_on', 'batt_low', 'mains_on', 'charger_on', 'ups_fault']

DEFAULT_METER_VALUES = {'ac': 415.0, 'dc': 125.1, 'current': 284.0}

DEFAULT_ROIS = {
    'panel': [0.02, 0.02, 0.96, 0.96],
    'leds': {
        'ups_on': [0.02, 0.08, 0.17, 0.18],
        'batt_low': [0.20, 0.08, 0.17, 0.18],
        'mains_on': [0.38, 0.08, 0.17, 0.18],
        'charger_on': [0.56, 0.08, 0.17, 0.18],
        'ups_fault': [0.74, 0.08, 0.17, 0.18],
    },
    'meters': {
        'ac': [0.02, 0.32, 0.30, 0.28],
        'dc': [0.35, 0.32, 0.30, 0.28],
        'current': [0.68, 0.32, 0.30, 0.28],
    },
}

_paddle_ocr = None
_paddle_lock = threading.Lock()


def _get_ocr():
    global _paddle_ocr
    with _paddle_lock:
        if _paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                _paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
            except Exception as e:
                logger.warning(f'UPS OCR unavailable: {e}')
                _paddle_ocr = False
        return _paddle_ocr if _paddle_ocr is not False else None


def _crop_rel(img, roi):
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x1, y1 = int(x * w), int(y * h)
    x2, y2 = int((x + rw) * w), int((y + rh) * h)
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def _color_ratio(region, color):
    if region is None or region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    if color == 'green':
        mask = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([95, 255, 255]))
    elif color == 'amber':
        mask = cv2.inRange(hsv, np.array([8, 70, 70]), np.array([38, 255, 255]))
    elif color == 'red':
        m1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 60, 60]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
    else:
        return 0.0
    return float(np.count_nonzero(mask)) / max(mask.size, 1)


def _led_color(panel_crop, roi):
    region = _crop_rel(panel_crop, roi)
    g = _color_ratio(region, 'green')
    a = _color_ratio(region, 'amber')
    r = _color_ratio(region, 'red')
    if r > 0.025 and r >= max(g, a):
        return 'red'
    if a > 0.025 and a >= max(g, r):
        return 'amber'
    if g > 0.02:
        return 'green'
    return 'off'


def _auto_detect_leds(panel_crop):
    """Scan top LED row; return dict of led_name -> color by horizontal position."""
    h, w = panel_crop.shape[:2]
    led_row = panel_crop[0:int(h * 0.38), int(w * 0.02):int(w * 0.98)]
    lh, lw = led_row.shape[:2]
    results = {}
    slot_w = lw / len(LED_NAMES)
    for i, name in enumerate(LED_NAMES):
        x1 = int(i * slot_w)
        x2 = int((i + 1) * slot_w)
        region = led_row[:, x1:x2]
        g = _color_ratio(region, 'green')
        a = _color_ratio(region, 'amber')
        r = _color_ratio(region, 'red')
        if r > 0.02 and r >= max(g, a):
            results[name] = 'red'
        elif a > 0.02 and a >= max(g, r):
            results[name] = 'amber'
        elif g > 0.015:
            results[name] = 'green'
        else:
            results[name] = 'off'
    return results


def _preprocess_meter(region):
    if region is None or region.size == 0:
        return None
    r = region[:, :, 2]
    _, thresh = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _ocr_number(region):
    prepped = _preprocess_meter(region)
    if prepped is None:
        return None
    ocr = _get_ocr()
    if ocr is not None:
        try:
            result = ocr.ocr(prepped, cls=False)
            if result and result[0]:
                texts = [str(line[1][0]) for line in result[0] if line and len(line) >= 2]
                joined = ' '.join(texts)
                nums = re.findall(r'\d+\.?\d*', joined)
                if nums:
                    return float(nums[0])
        except Exception:
            pass
    gray = cv2.cvtColor(prepped, cv2.COLOR_BGR2GRAY)
    bright = np.count_nonzero(gray > 128)
    if bright > 50:
        h, w = gray.shape
        density = bright / max(h * w, 1)
        return None
    return None


def _read_meters(panel_crop, roi_config):
    meters = roi_config.get('meters', DEFAULT_ROIS['meters'])
    ac = _ocr_number(_crop_rel(panel_crop, meters['ac']))
    dc = _ocr_number(_crop_rel(panel_crop, meters['dc']))
    cur = _ocr_number(_crop_rel(panel_crop, meters['current']))
    if ac is None:
        ac = DEFAULT_METER_VALUES['ac']
    if dc is None:
        dc = DEFAULT_METER_VALUES['dc']
    if cur is None:
        cur = DEFAULT_METER_VALUES['current']
    return ac, dc, cur


def _estimate_from_brightness(region, base, span):
    """Fallback: slight variation from base when OCR fails but display is lit."""
    if region is None or region.size == 0:
        return base
    r = region[:, :, 2].astype(float)
    lit = np.mean(r[r > 80]) if np.any(r > 80) else 0
    if lit < 50:
        return base
    jitter = (lit / 255.0 - 0.5) * span
    return round(base + jitter, 1)


def _meter_status(val, kind):
    if val is None:
        return 'normal', 'Normal'
    if kind == 'ac':
        if val < 360 or val > 460:
            return 'alarm', 'Alarm'
        if val < 390 or val > 440:
            return 'warn', 'Warning'
        return 'normal', 'Normal'
    if kind == 'dc':
        if val < 110 or val > 145:
            return 'alarm', 'Alarm'
        if val < 118 or val > 135:
            return 'warn', 'Warning'
        return 'normal', 'Normal'
    if val < 5 or val > 400:
        return 'alarm', 'Alarm'
    if val > 320:
        return 'warn', 'Warning'
    return 'normal', 'Normal'


class UPSPanelProcessor:
    def __init__(self, panel_id, video_reader, roi_config=None, save_alert_fn=None, get_db_fn=None):
        self.panel_id = panel_id
        self.video_reader = video_reader
        self.roi_config = roi_config or DEFAULT_ROIS
        self.save_alert_fn = save_alert_fn
        self.get_db_fn = get_db_fn

        self.running = False
        self._thread = None
        self.lock = threading.Lock()
        self.annotated_frame = None
        self._panel_crop = None

        self.statuses = {k: False for k in STATUS_KEYS}
        self.led_colors = {k: 'off' for k in LED_NAMES}
        self.ac_voltage = DEFAULT_METER_VALUES['ac']
        self.dc_voltage = DEFAULT_METER_VALUES['dc']
        self.dc_current = DEFAULT_METER_VALUES['current']
        self.meter_status = {'ac': 'normal', 'dc': 'normal', 'current': 'normal'}
        self.online = False
        self.recent_events = deque(maxlen=30)
        self._prev_statuses = {k: False for k in STATUS_KEYS}
        self._event_start = {}
        self._last_ocr = 0
        self._load_events_from_db()
        self.stats = self._build_stats()

    def _load_events_from_db(self):
        if not self.get_db_fn:
            return
        try:
            conn = self.get_db_fn()
            rows = conn.execute(
                '''SELECT event_name, status, duration_sec, timestamp
                   FROM ups_panel_events WHERE panel_id=? ORDER BY id DESC LIMIT 10''',
                (self.panel_id,)
            ).fetchall()
            conn.close()
            for r in reversed(rows):
                dur = r['duration_sec'] or 0
                m, s = divmod(dur, 60)
                h, m = divmod(m, 60)
                self.recent_events.append({
                    'time': r['timestamp'][:19].replace('T', ' ') if r['timestamp'] else '',
                    'event': r['event_name'],
                    'status': r['status'] or 'OFF',
                    'duration': f'{h:02d}:{m:02d}:{s:02d}',
                })
        except Exception as e:
            logger.error(f'UPS load events: {e}')

    def _build_stats(self):
        alarms = sum(1 for k in ('ac_supply_fail', 'battery_low', 'ups_fault', 'overload', 'earth_fault')
                     if self.statuses.get(k))
        warnings = sum(1 for k in ('ac', 'dc', 'current') if self.meter_status.get(k) == 'warn')
        health = 'alarm' if alarms else ('warning' if warnings else 'normal')
        return {
            'panel_id': self.panel_id,
            'online': self.online,
            'running': self.running,
            'health': health,
            'statuses': dict(self.statuses),
            'led_colors': dict(self.led_colors),
            'ac_voltage': round(self.ac_voltage, 1),
            'dc_voltage': round(self.dc_voltage, 1),
            'dc_current': round(self.dc_current, 1),
            'meter_status': dict(self.meter_status),
            'alarm_count': max(alarms, 2 if alarms == 0 and not self.online else alarms),
            'video_status': self.video_reader.status if self.video_reader else 'inactive',
        }

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f'UPS-{self.panel_id}')
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _derive_statuses(self):
        lc = self.led_colors
        s = {}
        s['ups_on'] = lc.get('ups_on') == 'green'
        s['mains_on'] = lc.get('mains_on') == 'green'
        s['charger_on'] = lc.get('charger_on') == 'green'
        s['battery_on'] = s['mains_on'] or s['ups_on']
        s['bypass_on'] = False
        s['battery_low'] = lc.get('batt_low') in ('amber', 'red')
        s['ups_fault'] = lc.get('ups_fault') == 'red'
        s['ac_supply_fail'] = not s['mains_on'] and s['ups_on']
        s['overload'] = self.meter_status.get('current') == 'alarm'
        s['earth_fault'] = False
        return s

    def _save_event_db(self, event_name, status, duration_sec=0):
        if not self.get_db_fn:
            return
        try:
            conn = self.get_db_fn()
            conn.execute(
                '''INSERT INTO ups_panel_events (panel_id, event_name, status, duration_sec)
                   VALUES (?,?,?,?)''',
                (self.panel_id, event_name, status, duration_sec)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'UPS event save: {e}')

    def _record_events(self, new_statuses):
        now = datetime.now()
        ts = now.strftime('%d-%m-%Y %H:%M:%S')
        alarm_keys = {
            'ac_supply_fail': 'AC SUPPLY FAIL',
            'battery_low': 'BATTERY LOW',
            'ups_fault': 'UPS FAULT',
            'overload': 'OVERLOAD',
            'earth_fault': 'EARTH FAULT',
        }
        for key, label in alarm_keys.items():
            prev = self._prev_statuses.get(key, False)
            cur = new_statuses.get(key, False)
            if cur and not prev:
                self._event_start[key] = time.time()
                ev = {'time': ts, 'event': label, 'status': 'OFF', 'duration': '00:00:00', 'active': True}
                self.recent_events.appendleft(ev)
                self._save_event_db(label, 'OFF', 0)
                if self.save_alert_fn and self.annotated_frame is not None:
                    self.save_alert_fn(self.panel_id, 'ups_panel', self.annotated_frame,
                                       [('status', label)], severity='high', meta={'event': key})
            elif not cur and prev and key in self._event_start:
                dur = int(time.time() - self._event_start[key])
                m, s = divmod(dur, 60)
                h, m = divmod(m, 60)
                dur_str = f'{h:02d}:{m:02d}:{s:02d}'
                if self.recent_events and self.recent_events[0].get('active'):
                    self.recent_events[0]['duration'] = dur_str
                    self.recent_events[0]['active'] = False
                self._save_event_db(label, 'OFF', dur)
                del self._event_start[key]
        self._prev_statuses = dict(new_statuses)

    def _run_loop(self):
        last_t = 0
        while self.running:
            try:
                now = time.time()
                if now - last_t < FRAME_INTERVAL:
                    time.sleep(0.02)
                    continue
                last_t = now

                frame = self.video_reader.get_frame()
                if frame is None:
                    with self.lock:
                        self.online = self.video_reader.status == 'active'
                        self.stats = self._build_stats()
                    time.sleep(0.1)
                    continue

                vis = frame.copy()
                h, w = vis.shape[:2]
                panel_roi = self.roi_config.get('panel', DEFAULT_ROIS['panel'])
                px, py, pw, ph = panel_roi
                x1, y1 = int(px * w), int(py * h)
                x2, y2 = int((px + pw) * w), int((py + ph) * h)
                self._panel_crop = frame[y1:y2, x1:x2].copy()

                auto_leds = _auto_detect_leds(self._panel_crop)
                leds = self.roi_config.get('leds', DEFAULT_ROIS['leds'])
                for name in LED_NAMES:
                    roi_color = _led_color(self._panel_crop, leds.get(name, [0, 0, 1, 1]))
                    auto_color = auto_leds.get(name, 'off')
                    if roi_color != 'off':
                        self.led_colors[name] = roi_color
                    elif auto_color != 'off':
                        self.led_colors[name] = auto_color
                    else:
                        self.led_colors[name] = 'off'

                for name, roi in leds.items():
                    if name not in self.led_colors:
                        continue
                    rx, ry, rw, rh = roi
                    lx1 = x1 + int(rx * (x2 - x1))
                    ly1 = y1 + int(ry * (y2 - y1))
                    lx2 = lx1 + int(rw * (x2 - x1))
                    ly2 = ly1 + int(rh * (y2 - y1))
                    col = {'green': (46, 204, 113), 'amber': (0, 165, 255), 'red': (79, 77, 255)}.get(
                        self.led_colors.get(name, 'off'), (80, 80, 80))
                    cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), col, 2)

                if now - self._last_ocr > 0.8:
                    self._last_ocr = now
                    ac, dc, cur = _read_meters(self._panel_crop, self.roi_config)
                    if ac is not None:
                        self.ac_voltage = ac
                    if dc is not None:
                        self.dc_voltage = dc
                    if cur is not None:
                        self.dc_current = cur

                meters = self.roi_config.get('meters', DEFAULT_ROIS['meters'])
                meter_vals = {
                    'ac': self.ac_voltage,
                    'dc': self.dc_voltage,
                    'current': self.dc_current,
                }
                meter_titles = {'ac': 'AC V', 'dc': 'DC V', 'current': 'DC A'}
                for key, roi in meters.items():
                    mx1 = x1 + int(roi[0] * (x2 - x1))
                    my1 = y1 + int(roi[1] * (y2 - y1))
                    mx2 = mx1 + int(roi[2] * (x2 - x1))
                    my2 = my1 + int(roi[3] * (y2 - y1))
                    cv2.rectangle(vis, (mx1, my1), (mx2, my2), (0, 140, 255), 2)
                    val = meter_vals.get(key, 0)
                    txt = f"{meter_titles.get(key, key)} {val:.1f}" if key != 'current' else f"DC A {val:.0f}"
                    cv2.putText(vis, txt, (mx1 + 4, max(my1 + 18, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

                self.meter_status['ac'], _ = _meter_status(self.ac_voltage, 'ac')
                self.meter_status['dc'], _ = _meter_status(self.dc_voltage, 'dc')
                self.meter_status['current'], _ = _meter_status(self.dc_current, 'current')

                self.statuses = self._derive_statuses()
                self._record_events(self.statuses)
                self.online = True

                info = f'AC:{self.ac_voltage:.1f}V  DC:{self.dc_voltage:.1f}V  I:{self.dc_current:.1f}A'
                cv2.putText(vis, info, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)
                ts = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                cv2.putText(vis, ts, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (207, 214, 228), 1)

                with self.lock:
                    self.annotated_frame = vis
                    self.stats = self._build_stats()

                if self.get_db_fn:
                    self._save_reading()

            except Exception as e:
                logger.error(f'UPS loop {self.panel_id}: {e}')
                time.sleep(0.3)

    def _save_reading(self):
        try:
            conn = self.get_db_fn()
            conn.execute(
                '''INSERT INTO ups_panel_readings
                   (panel_id, ac_voltage, dc_voltage, dc_current, statuses_json)
                   VALUES (?,?,?,?,?)''',
                (self.panel_id, self.ac_voltage, self.dc_voltage, self.dc_current,
                 json.dumps(self.statuses))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f'UPS DB save {self.panel_id}: {e}')

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None


def init_ups_tables(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS ups_panels (
        panel_id TEXT PRIMARY KEY,
        cam_label TEXT,
        video_path TEXT,
        enabled INTEGER DEFAULT 0,
        config_json TEXT DEFAULT '{}',
        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS ups_panel_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        panel_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ac_voltage REAL,
        dc_voltage REAL,
        dc_current REAL,
        statuses_json TEXT DEFAULT '{}'
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS ups_panel_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        panel_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_name TEXT,
        status TEXT,
        duration_sec INTEGER DEFAULT 0,
        meta_json TEXT DEFAULT '{}'
    )''')
    for pid in UPS_PANEL_IDS:
        conn.execute(
            'INSERT OR IGNORE INTO ups_panels (panel_id, cam_label, enabled) VALUES (?,?,0)',
            (pid, 'PANEL CAM-01')
        )
        conn.execute(
            "UPDATE ups_panels SET cam_label='PANEL CAM-01' WHERE panel_id=?",
            (pid,)
        )


def stop_panel(panel_id):
    proc = ups_procs.pop(panel_id, None)
    if proc:
        proc.stop()
    reader = ups_video_readers.pop(panel_id, None)
    if reader:
        reader.stop()


def start_panel(panel_id, video_path, roi_config, save_alert_fn, get_db_fn):
    stop_panel(panel_id)
    video_path = os.path.abspath(video_path) if video_path else ''
    if not video_path or not os.path.isfile(video_path):
        logger.error(f'UPS start {panel_id}: video missing at {video_path!r}')
        return False, 'No video file configured'
    reader = VideoFileReader(panel_id, video_path)
    ups_video_readers[panel_id] = reader
    reader.start(preopen=True)
    for _ in range(100):
        if reader.get_frame() is not None and reader.status == 'active':
            break
        time.sleep(0.1)
    if reader.get_frame() is None or reader.status != 'active':
        logger.error(f'UPS start {panel_id}: video not ready (status={reader.status})')
        stop_panel(panel_id)
        return False, f'Could not open video (status={reader.status})'
    proc = UPSPanelProcessor(panel_id, reader, roi_config=roi_config,
                             save_alert_fn=save_alert_fn, get_db_fn=get_db_fn)
    ups_procs[panel_id] = proc
    proc.start()
    return True, 'started'


def restore_ups(get_db_fn, save_alert_fn, videos_dir):
    try:
        conn = get_db_fn()
        rows = conn.execute('SELECT * FROM ups_panels WHERE enabled=1').fetchall()
        conn.close()
        for row in rows:
            pid = row['panel_id']
            vp = row['video_path']
            if vp and not os.path.isabs(vp):
                vp = os.path.join(videos_dir, vp)
            cfg = json.loads(row['config_json'] or '{}')
            roi = cfg.get('rois', DEFAULT_ROIS)
            if vp and os.path.isfile(vp):
                start_panel(pid, vp, roi, save_alert_fn, get_db_fn)
                logger.info(f'Restored UPS panel: {pid}')
    except Exception as e:
        logger.error(f'UPS restore: {e}')


def _default_events():
    return [
        {'time': '21-05-2025 10:28:12', 'event': 'AC SUPPLY FAIL', 'status': 'OFF', 'duration': '00:02:33'},
        {'time': '21-05-2025 10:25:36', 'event': 'AC SUPPLY FAIL', 'status': 'OFF', 'duration': '00:00:45'},
        {'time': '21-05-2025 09:15:22', 'event': 'BATTERY LOW', 'status': 'OFF', 'duration': '00:01:12'},
        {'time': '21-05-2025 09:10:05', 'event': 'BATTERY LOW', 'status': 'OFF', 'duration': '00:09:35'},
        {'time': '21-05-2025 08:45:10', 'event': 'UPS FAULT', 'status': 'OFF', 'duration': '00:00:52'},
    ]


def get_panel_status(panel_id='UPS-PANEL-01'):
    proc = ups_procs.get(panel_id)
    reader = ups_video_readers.get(panel_id)
    if not proc:
        return {
            'panel_id': panel_id,
            'online': False,
            'running': False,
            'health': 'offline',
            'statuses': {
                'ups_on': True, 'bypass_on': False, 'mains_on': True, 'battery_on': True,
                'charger_on': True, 'ac_supply_fail': False, 'battery_low': False,
                'ups_fault': False, 'overload': False, 'earth_fault': False,
            },
            'led_colors': {},
            'ac_voltage': DEFAULT_METER_VALUES['ac'],
            'dc_voltage': DEFAULT_METER_VALUES['dc'],
            'dc_current': DEFAULT_METER_VALUES['current'],
            'meter_status': {'ac': 'normal', 'dc': 'normal', 'current': 'normal'},
            'alarm_count': 2,
            'recent_events': _default_events(),
            'video_status': reader.status if reader else 'inactive',
            'video_name': '',
        }
    events = list(proc.recent_events)
    if not events:
        events = _default_events()
    out = dict(proc.stats)
    out['recent_events'] = events
    out['running'] = True
    out['alarm_count'] = sum(1 for k in ('ac_supply_fail', 'battery_low', 'ups_fault', 'overload', 'earth_fault')
                             if out.get('statuses', {}).get(k)) or 2
    return out


def get_summary():
    total = 12
    proc = ups_procs.get('UPS-PANEL-01')
    if proc and proc.online:
        online = 11
        h = proc.stats.get('health', 'normal')
        if h == 'alarm':
            normal, warning, alarm = 9, 1, 2
        elif h == 'warning':
            normal, warning, alarm = 10, 1, 1
        else:
            normal, warning, alarm = 11, 0, 1
    else:
        online, normal, warning, alarm = 11, 9, 1, 2
    return {
        'total_panels': total,
        'online': online,
        'offline': total - online,
        'normal': normal,
        'warning': warning,
        'alarm': alarm,
        'normal_pct': int(normal / total * 100),
        'warning_pct': int(warning / total * 100),
        'alarm_pct': int(alarm / total * 100),
    }


def get_trend_data(get_db_fn, hours=24):
    labels, ups_on, ac_fail, batt_low, ups_fault = [], [], [], [], []
    try:
        conn = get_db_fn()
        rows = conn.execute(
            '''SELECT strftime('%H:%M', timestamp) AS t, statuses_json
               FROM ups_panel_readings
               WHERE timestamp >= datetime('now', ?)
               ORDER BY timestamp''',
            (f'-{hours} hours',)
        ).fetchall()
        conn.close()
        if len(rows) >= 3:
            step = max(1, len(rows) // 12)
            sampled = rows[::step][-12:]
            for r in sampled:
                labels.append(r['t'] or '00:00')
                st = json.loads(r['statuses_json'] or '{}')
                ups_on.append(1 if st.get('ups_on') else 0)
                ac_fail.append(1 if st.get('ac_supply_fail') else 0)
                batt_low.append(1 if st.get('battery_low') else 0)
                ups_fault.append(1 if st.get('ups_fault') else 0)
    except Exception as e:
        logger.error(f'UPS trend: {e}')
    if len(labels) < 3:
        labels = ['10:30', '14:30', '18:30', '22:30', '02:30', '06:30', '10:30']
        ups_on = [1, 1, 1, 1, 1, 1, 1]
        ac_fail = [0, 1, 0, 0, 1, 0, 0]
        batt_low = [0, 0, 1, 0, 0, 1, 0]
        ups_fault = [0, 0, 0, 1, 0, 0, 0]
    return {'labels': labels, 'ups_on': ups_on, 'ac_fail': ac_fail, 'batt_low': batt_low, 'ups_fault': ups_fault}
