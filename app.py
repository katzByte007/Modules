import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import logging
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, g, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO
import threading
import socket
from contextlib import closing
import warnings
import math
import colorsys
import sqlite3
import os
import json
import secrets
from datetime import datetime
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import subprocess
import platform
import inspect
import gc
import asyncio
import re
from urllib.parse import urlsplit
import smtplib
from email.mime.text import MIMEText

# Optional: system/GPU stats (app works without these)
try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False
try:
    import torch
    _torch_cuda_available = torch.cuda.is_available()
except Exception:
    torch = None
    _torch_cuda_available = False

try:
    from libreyolo import ByteTracker as _LibreByteTracker, LibreYOLO as _LibreYOLO, TrackConfig as _LibreTrackConfig
    _LIBREYOLO_AVAILABLE = True
except Exception:
    _LibreByteTracker = None
    _LibreYOLO = None
    _LibreTrackConfig = None
    _LIBREYOLO_AVAILABLE = False

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except Exception:
    _PaddleOCR = None
    _PADDLEOCR_AVAILABLE = False

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================
# CONFIGURATION
# ==============================================================
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#DATA_DIR = os.path.join(BASE_DIR, 'data')
#DB_PATH = os.path.join(DATA_DIR, 'vision_ai.db')
#   HEADCOUNT_MODEL_PATH = os.path.join(BASE_DIR, 'best10xemphead.pt')
#ENTRYEXIT_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8m-seg.pt')
#FLAPGATE_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8m-seg.pt')
BASE_DIR = os.path.abspath(os.getcwd())
# Persist DB, alerts, and incident logs here. Override on servers, e.g. bind-mount host folder to /app/data.
_DATA_ENV = os.environ.get('VISION_DATA_DIR', '').strip()
DATA_DIR = _DATA_ENV if _DATA_ENV else os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'vision_ai.db')
HEADCOUNT_MODEL_PATH = os.path.join(BASE_DIR, 'best10xemphead.pt')
ENTRYEXIT_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8m.pt')
FLAPGATE_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8m.pt')
PPE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'safetypremium.pt')
FIRE_SMOKE_MODEL_PATH = os.environ.get('VISION_FIRE_SMOKE_MODEL_PATH', os.path.join(BASE_DIR, 'firensmoke.pt')).strip()
ANPR_PLATE_MODEL_PATH = os.environ.get('VISION_ANPR_PLATE_MODEL_PATH', os.path.join(BASE_DIR, 'ANPRlib.pt')).strip()
ANPR_VEHICLE_MODEL_PATH = os.environ.get('VISION_ANPR_VEHICLE_MODEL_PATH', os.path.join(BASE_DIR, 'LibreYOLO9s.pt')).strip()
OWLV2_MODEL_ID = os.environ.get('VISION_OWLV2_MODEL', 'google/owlv2-base-patch16-ensemble')
OWLV2_LOCAL_PATH = os.environ.get('VISION_OWLV2_LOCAL_PATH', '').strip()
FLORENCE2_LOCAL_PATH = os.environ.get('VISION_FLORENCE2_LOCAL_PATH', '').strip()
QWEN25VL_LOCAL_PATH = os.environ.get('VISION_QWEN25VL_LOCAL_PATH', '').strip()
# Subnet for IP camera discovery (camera VLAN). Empty = infer from configured RTSP URLs, else server subnet.
VISION_CAMERA_SCAN_SUBNET = os.environ.get('VISION_CAMERA_SCAN_SUBNET', '').strip()
ALERTS_DIR = os.path.join(DATA_DIR, 'alerts')
WORKFORCE_VIDEOS_DIR = os.path.join(DATA_DIR, 'workforce_videos')
UPS_VIDEOS_DIR = os.path.join(DATA_DIR, 'ups_videos')
ALERT_COOLDOWN_SEC = float(os.environ.get('VISION_ALERT_COOLDOWN_SEC', '18'))
INCIDENT_AI_LOG_PATH = os.path.join(DATA_DIR, 'incident_ai_outputs.jsonl')
ODD_UPLOADS_DIR = os.path.join(DATA_DIR, 'odd_uploads')

# Microsoft Edge TTS neural voices (used when browser speech lacks Indic/local voices).
EDGE_TTS_VOICES = {
    'English': 'en-US-JennyNeural',
    'Hindi': 'hi-IN-SwaraNeural',
    'Tamil': 'ta-IN-PallaviNeural',
    'Telugu': 'te-IN-ShrutiNeural',
    'Kannada': 'kn-IN-GaganNeural',
    'Malayalam': 'ml-IN-SobhanaNeural',
    'Marathi': 'mr-IN-AarohiNeural',
    'Bengali': 'bn-IN-TanishaaNeural',
    'Gujarati': 'gu-IN-DhwaniNeural',
    'Spanish': 'es-ES-ElviraNeural',
    'French': 'fr-FR-DeniseNeural',
    'German': 'de-DE-KatjaNeural',
    'Arabic': 'ar-SA-ZariyahNeural',
    'Japanese': 'ja-JP-NanamiNeural',
    'Korean': 'ko-KR-SunHiNeural',
    'Chinese': 'zh-CN-XiaoxiaoNeural',
}

# VLM resource management: OWLv2 / Florence (beta) and Qwen2.5-VL (incident) share one GPU with YOLO.
# Default: beta VLMs on CUDA; Qwen on CUDA (fast response) and unload on explicit Exit.
VISION_VLM_DEVICE = os.environ.get('VISION_VLM_DEVICE', 'cuda').strip().lower()
VISION_QWEN_DEVICE = os.environ.get('VISION_QWEN_DEVICE', 'cuda').strip().lower()
VISION_QWEN_UNLOAD_AFTER_USE = os.environ.get('VISION_QWEN_UNLOAD_AFTER_USE', '0').strip().lower() in ('1', 'true', 'yes')
VISION_VLM_EVICT_OTHERS = os.environ.get('VISION_VLM_EVICT_OTHERS', '1').strip().lower() in ('1', 'true', 'yes')
VISION_QWEN_ALLOW_GPU_WITH_BETA = os.environ.get('VISION_QWEN_ALLOW_GPU_WITH_BETA', '1').strip().lower() in ('1', 'true', 'yes')
VISION_QWEN_MIN_PIXELS = int(os.environ.get('VISION_QWEN_MIN_PIXELS', str(256 * 28 * 28)))
VISION_QWEN_MAX_PIXELS = int(os.environ.get('VISION_QWEN_MAX_PIXELS', str(1280 * 28 * 28)))
VISION_QWEN_IMAGE_MAX_SIDE = int(os.environ.get('VISION_QWEN_IMAGE_MAX_SIDE', '1024'))
VISION_QWEN_USE_4BIT = os.environ.get('VISION_QWEN_USE_4BIT', '0').strip().lower() in ('1', 'true', 'yes')
VISION_QWEN_GPU_MIN_FREE_GB = float(os.environ.get('VISION_QWEN_GPU_MIN_FREE_GB', '20'))
VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC = float(os.environ.get('VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC', '6'))

# ---------- Resource tuning: aim ~50% system use, 100% accuracy, 24/7 stable ----------
# Achieved so far: GPU ~17% -> ~13%, VRAM ~36% -> ~28% (imgsz 640->480, fps 12->10, fp16, fewer workers).
# For 1080p+ streams (e.g. 1920x1080, 2688x1520 @ 25fps HEVC): set VISION_INFERENCE_IMGSZ=640 (or 720)
# for better model accuracy on small/distant objects (PPE, heads); 480 is faster but loses detail.
INFERENCE_IMGSZ = int(os.environ.get('VISION_INFERENCE_IMGSZ', '720'))
# Inference FPS per stream; 8-10 is enough for counting/entry-exit without losing events.
TARGET_FPS = float(os.environ.get('VISION_TARGET_FPS', '25'))
# Frames per YOLO batch; 6-8 balances latency vs GPU utilization.
MICRO_BATCH = int(os.environ.get('VISION_MICRO_BATCH', '8'))
# Threads for post-inference callbacks; avoid oversubscribing CPU.
CALLBACK_POOL_WORKERS = int(os.environ.get('VISION_CALLBACK_POOL_WORKERS', '12'))
# FP16 on GPU saves VRAM and speeds inference (RTX 6000 supports it well).
USE_FP16 = os.environ.get('VISION_USE_FP16', '1').strip().lower() in ('1', 'true', 'yes')
# TensorRT: 1 = use .engine when present or auto-export from .pt (same accuracy, 2-5x less GPU). 0 = PyTorch only.
USE_TENSORRT_IF_AVAILABLE = os.environ.get('VISION_USE_TENSORRT', '1').strip().lower() in ('1', 'true', 'yes')
# PPE: 1 = use SAHI (sliced inference). 0 = use pharmappenew.pt via shared InferenceEngine (same as other models).
USE_SAHI_PPE = os.environ.get('VISION_PPE_USE_SAHI', '0').strip().lower() in ('1', 'true', 'yes')
ANPR_TARGET_FPS = float(os.environ.get('VISION_ANPR_TARGET_FPS', '8'))
ANPR_PLATE_CONF = float(os.environ.get('VISION_ANPR_PLATE_CONF', '0.1'))
ANPR_VEHICLE_CONF = float(os.environ.get('VISION_ANPR_VEHICLE_CONF', '0.25'))
ANPR_IOU = float(os.environ.get('VISION_ANPR_IOU', '0.45'))
ANPR_IMGSZ = int(os.environ.get('VISION_ANPR_IMGSZ', '640'))
ANPR_MAX_DET = int(os.environ.get('VISION_ANPR_MAX_DET', '300'))
ANPR_METERS_PER_PIXEL = float(os.environ.get('VISION_ANPR_METERS_PER_PIXEL', '0.15'))
ANPR_SPEED_SAMPLE_FRAMES = int(os.environ.get('VISION_ANPR_SPEED_SAMPLE_FRAMES', '5'))
ANPR_PLATE_DETECT_INTERVAL = int(os.environ.get('VISION_ANPR_PLATE_DETECT_INTERVAL', '2'))
ANPR_OCR_REFRESH_FRAMES = int(os.environ.get('VISION_ANPR_OCR_REFRESH_FRAMES', '45'))
ANPR_MAX_OCR_PLATES_PER_FRAME = int(os.environ.get('VISION_ANPR_MAX_OCR_PLATES_PER_FRAME', '3'))
ANPR_EVENT_SAVE_INTERVAL_SEC = float(os.environ.get('VISION_ANPR_EVENT_SAVE_INTERVAL_SEC', '15'))
try:
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
    _SAHI_AVAILABLE = True
except Exception:
    get_sliced_prediction = None

# Optional: OWLv2 VLM for Beta (open-vocabulary) detection
_OWLV2_AVAILABLE = False
try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    _OWLV2_AVAILABLE = True
except Exception:
    Owlv2Processor = None
    Owlv2ForObjectDetection = None
    Image = None
    AutoDetectionModel = None
    _SAHI_AVAILABLE = False

# Optional: Florence-2 for Beta (phrase grounding with user-provided label list)
_FLORENCE2_AVAILABLE = False
_FlorenceProcessorClass = None
_FlorenceModelClass = None
try:
    from transformers import AutoProcessor as _FlorenceProcessorClass
    try:
        from transformers import Florence2ForConditionalGeneration as _FlorenceModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as _FlorenceModelClass
    _FLORENCE2_AVAILABLE = True
except Exception:
    _FlorenceProcessorClass = None
    _FlorenceModelClass = None

# Optional: Qwen vision-language model for incident investigation
_QWEN25VL_AVAILABLE = False
_QwenProcessorClass = None
_QwenModelClass = None
try:
    from transformers import AutoProcessor as _QwenProcessorClass
    try:
        from transformers import AutoModelForImageTextToText as _QwenModelClass
    except Exception:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _QwenModelClass
        except Exception:
            from transformers import AutoModelForVision2Seq as _QwenModelClass
    _QWEN25VL_AVAILABLE = True
except Exception:
    _QwenProcessorClass = None
    _QwenModelClass = None

try:
    import edge_tts as _edge_tts_mod
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _edge_tts_mod = None
    _EDGE_TTS_AVAILABLE = False

try:
    from pypdf import PdfReader as _PdfReaderClass
    _PYPDF_AVAILABLE = True
except ImportError:
    _PdfReaderClass = None
    _PYPDF_AVAILABLE = False

if Image is None:
    try:
        from PIL import Image as _PILImage
        Image = _PILImage
    except ImportError:
        Image = None

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)
os.makedirs(ODD_UPLOADS_DIR, exist_ok=True)
os.makedirs(WORKFORCE_VIDEOS_DIR, exist_ok=True)
os.makedirs(UPS_VIDEOS_DIR, exist_ok=True)

from workforce_monitoring import (
    WORKFORCE_MACHINE_IDS,
    workforce_video_readers,
    workforce_procs,
    init_workforce_tables,
    get_aggregate_status,
    get_utilization_chart,
    get_ppe_stats,
    stop_machine as wf_stop_machine,
    start_machine as wf_start_machine,
    restore_workforce,
    bind_workforce_videos_from_folder,
    get_workforce_demo_status,
    get_workforce_video_map,
    discover_workforce_video,
    workforce_video_version,
    ensure_workforce_playback,
)

from module_dashboard_store import (
    init_module_dashboard_tables,
    log_chat_message,
    seed_workforce_demo_alerts,
    seed_panel_demo_data,
    start_dashboard_persistence,
    chat_answer_from_snapshots,
    persist_workforce_dashboard,
    persist_panel_dashboard,
)

from ups_panel_monitoring import (
    UPS_PANEL_IDS,
    ups_video_readers,
    ups_procs,
    init_ups_tables,
    stop_panel as ups_stop_panel,
    start_panel as ups_start_panel,
    restore_ups,
    get_panel_status,
    get_summary as ups_get_summary,
    get_trend_data as ups_get_trend_data,
)

# ==============================================================
# DATABASE
# ==============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS cameras (
        camera_id TEXT PRIMARY KEY,
        name TEXT,
        rtsp_url TEXT NOT NULL,
        added_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'inactive'
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS detection_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        detection_type TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        config_json TEXT DEFAULT '{}',
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(camera_id, detection_type),
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS headcounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        current_count INTEGER DEFAULT 0,
        total_entries INTEGER DEFAULT 0,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS entryexit_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        entries INTEGER DEFAULT 0,
        exits INTEGER DEFAULT 0,
        current_inside INTEGER DEFAULT 0,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS flapgate_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        trespassing_count INTEGER DEFAULT 0,
        persons_in_frame INTEGER DEFAULT 0,
        gate1_status TEXT DEFAULT 'red',
        gate2_status TEXT DEFAULT 'red',
        gate3_status TEXT DEFAULT 'red',
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS ppe_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        compliant_count INTEGER DEFAULT 0,
        violation_count INTEGER DEFAULT 0,
        total_persons INTEGER DEFAULT 0,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS fire_smoke_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        fire_count INTEGER DEFAULT 0,
        smoke_count INTEGER DEFAULT 0,
        total_detections INTEGER DEFAULT 0,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS anpr_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        track_id INTEGER,
        vehicle_type TEXT,
        license_plate TEXT,
        speed_kmh REAL,
        speed_threshold_kmh REAL,
        overspeeding INTEGER DEFAULT 0,
        confidence REAL DEFAULT 0,
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS beta_detection_prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_text TEXT NOT NULL,
        camera_ids TEXT NOT NULL,
        confidence REAL DEFAULT 0.2,
        enabled INTEGER DEFAULT 1,
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS beta_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        model TEXT NOT NULL DEFAULT 'owlv2',
        confidence REAL DEFAULT 0.2
    )''')
    c.execute("INSERT OR IGNORE INTO beta_settings (id, model, confidence) VALUES (1, 'owlv2', 0.2)")

    c.execute('''CREATE TABLE IF NOT EXISTS alert_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        detection_type TEXT NOT NULL,
        alert_label TEXT NOT NULL,
        severity TEXT DEFAULT 'medium',
        snapshot_path TEXT,
        meta_json TEXT DEFAULT '{}',
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS incident_ai_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id INTEGER NOT NULL,
        output_type TEXT NOT NULL,
        language TEXT,
        mode TEXT,
        prompt_text TEXT,
        response_text TEXT,
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS incident_workspaces (
        alert_id INTEGER PRIMARY KEY,
        analysis_language TEXT,
        analysis_mode TEXT,
        chat_language TEXT,
        chat_mode TEXT,
        odd_context TEXT,
        commentary TEXT,
        rootcause TEXT,
        chat_history_json TEXT DEFAULT '[]',
        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS app_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        level TEXT NOT NULL DEFAULT 'user',
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS user_active_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT UNIQUE NOT NULL,
        user_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        level TEXT NOT NULL,
        ip TEXT,
        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES app_users(id) ON DELETE CASCADE
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS system_config_kv (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    init_workforce_tables(conn)
    init_ups_tables(conn)
    init_module_dashboard_tables(conn)
    seed_workforce_demo_alerts(conn)
    seed_panel_demo_data(conn)

    _migrate_alert_events_schema(conn)
    _migrate_cameras_schema(conn)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def _migrate_alert_events_schema(conn):
    """Add status/assignment columns to alert_events on existing databases."""
    try:
        cols = {row[1] for row in conn.execute('PRAGMA table_info(alert_events)').fetchall()}
        if 'status' not in cols:
            conn.execute("ALTER TABLE alert_events ADD COLUMN status TEXT DEFAULT 'open'")
        if 'assigned_to' not in cols:
            conn.execute("ALTER TABLE alert_events ADD COLUMN assigned_to TEXT DEFAULT ''")
        if 'assigned_engineer_id' not in cols:
            conn.execute("ALTER TABLE alert_events ADD COLUMN assigned_engineer_id TEXT DEFAULT ''")
    except Exception as e:
        logger.warning(f"alert_events schema migration: {e}")


def _migrate_cameras_schema(conn):
    """Add geospatial columns to cameras on existing databases."""
    try:
        cols = {row[1] for row in conn.execute('PRAGMA table_info(cameras)').fetchall()}
        if 'latitude' not in cols:
            conn.execute('ALTER TABLE cameras ADD COLUMN latitude REAL')
        if 'longitude' not in cols:
            conn.execute('ALTER TABLE cameras ADD COLUMN longitude REAL')
    except Exception as e:
        logger.warning(f"cameras schema migration: {e}")


def _seed_default_admin():
    """Create a default administrator if no users exist (change password in production)."""
    conn = get_db()
    try:
        n = conn.execute('SELECT COUNT(*) AS c FROM app_users').fetchone()['c']
        if n == 0:
            u = (os.environ.get('VISION_DEFAULT_ADMIN_USER') or 'admin').strip() or 'admin'
            p = (os.environ.get('VISION_DEFAULT_ADMIN_PASSWORD') or 'admin').strip() or 'admin'
            h = generate_password_hash(p)
            conn.execute(
                'INSERT INTO app_users (username, password_hash, level) VALUES (?,?,?)',
                (u, h, 'admin'),
            )
            conn.commit()
            logger.warning(
                "No users in database: created default admin %r (set VISION_DEFAULT_ADMIN_USER/PASSWORD to override).",
                u,
            )
    except Exception as e:
        logger.error(f"Default admin seed error: {e}")
    finally:
        conn.close()


init_database()
_seed_default_admin()


def _save_incident_ai_output(alert_id, output_type, language, mode, prompt_text, response_text):
    try:
        conn = get_db()
        conn.execute(
            'INSERT INTO incident_ai_outputs (alert_id, output_type, language, mode, prompt_text, response_text) VALUES (?,?,?,?,?,?)',
            (int(alert_id), str(output_type), str(language or ''), str(mode or ''), str(prompt_text or ''), str(response_text or ''))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"incident_ai_outputs DB save error: {e}")
    try:
        rec = {
            'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alert_id': int(alert_id),
            'output_type': output_type,
            'language': language,
            'mode': mode,
            'prompt': prompt_text,
            'response': response_text,
        }
        with open(INCIDENT_AI_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _serialize_chat_history(items):
    clean = []
    if isinstance(items, list):
        for row in items:
            if not isinstance(row, dict):
                continue
            who = 'me' if str(row.get('who', '')).strip() == 'me' else 'ai'
            text = str(row.get('text') or '').strip()
            if not text:
                continue
            clean.append({'who': who, 'text': text[:4000]})
            if len(clean) >= 200:
                break
    return json.dumps(clean, ensure_ascii=False)


def _parse_chat_history(raw_json):
    try:
        data = json.loads(raw_json or '[]')
    except Exception:
        data = []
    out = []
    if isinstance(data, list):
        for row in data[:200]:
            if not isinstance(row, dict):
                continue
            who = 'me' if str(row.get('who', '')).strip() == 'me' else 'ai'
            text = str(row.get('text') or '').strip()
            if text:
                out.append({'who': who, 'text': text})
    return out

_alert_lock = threading.Lock()
_alert_last_ts = {}  # (camera_id, detection_type) -> unix time


_QWEN25VL_MODEL_TYPES = frozenset({'qwen2_5_vl', 'qwen2_vl'})


def _is_qwen25vl_model_dir(model_dir):
    """True only for Qwen2.x-VL checkpoints (excludes Qwen3.5 text/VLM variants)."""
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return False
    model_type = (cfg.get('model_type') or '').strip().lower()
    if model_type in _QWEN25VL_MODEL_TYPES:
        return True
    arch = cfg.get('architectures') or []
    arch_s = ' '.join(str(a) for a in arch).lower()
    return 'qwen2_5_vl' in arch_s or 'qwen2_vl' in arch_s


def _qwen25vl_local_dir():
    candidates = []
    if QWEN25VL_LOCAL_PATH:
        p = QWEN25VL_LOCAL_PATH
        if os.path.isfile(p):
            p = os.path.dirname(p)
        candidates.append(p)
    candidates.extend([
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache', 'Qwen2.5-VL-3B-Instruct'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache', 'Qwen2.5-VL-3B-Instruct'),
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache', 'Qwen2.5-VL'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache', 'Qwen2.5-VL'),
        os.path.join(BASE_DIR, '.hf_qwen25vl_cache'),
        os.path.join(BASE_DIR, 'hf_qwen25vl_cache'),
        os.path.join(
            BASE_DIR, 'hf_qwen2vl_3b_4bit', 'hf_qwen2vl_3b_4bit',
            'Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit',
        ),
    ])
    for d in candidates:
        if d and _is_qwen25vl_model_dir(d):
            return d
    return None


def _qwen25vl_ready():
    return bool(_QWEN25VL_AVAILABLE and _QwenProcessorClass and _QwenModelClass and _qwen25vl_local_dir() and Image is not None)


_vlm_global_lock = threading.Lock()
_qwen25vl_processor = None
_qwen25vl_model = None
_qwen25vl_lock = threading.Lock()
_qwen25vl_infer_lock = threading.Lock()
_qwen25vl_device = None  # 'cuda' | 'cpu' used for last load


def _torch_free_cuda_memory():
    try:
        gc.collect()
        if torch is not None and _torch_cuda_available:
            torch.cuda.empty_cache()
    except Exception:
        pass


def _beta_vlm_camera_count():
    """How many beta (OWLv2/Florence) streams are active — used to avoid GPU Qwen colliding with beta."""
    bp = globals().get('beta_procs')
    if not isinstance(bp, dict):
        return 0
    return len(bp)


def _vlm_beta_device():
    d = VISION_VLM_DEVICE if VISION_VLM_DEVICE in ('cuda', 'cpu') else 'cuda'
    if d == 'cuda' and not _torch_cuda_available:
        return 'cpu'
    return d


def _qwen_resolve_device():
    """Resolve Qwen device (default cuda). Can fall back to CPU if CUDA unavailable."""
    raw_cfg = os.environ.get('VISION_QWEN_DEVICE', VISION_QWEN_DEVICE).strip().lower()
    raw = raw_cfg if raw_cfg in ('cuda', 'cpu') else 'cpu'
    if raw == 'cuda' and not _torch_cuda_available:
        return 'cpu'
    if raw == 'cuda' and torch is not None and _torch_cuda_available:
        try:
            free_b, _total_b = torch.cuda.mem_get_info()
            free_gb = float(free_b) / (1024 ** 3)
            if free_gb < VISION_QWEN_GPU_MIN_FREE_GB:
                logger.warning(
                    f"Qwen vision: free VRAM {free_gb:.2f}GB < {VISION_QWEN_GPU_MIN_FREE_GB:.2f}GB; using CPU for safety."
                )
                return 'cpu'
        except Exception:
            pass
    if raw == 'cuda' and _beta_vlm_camera_count() > 0 and not VISION_QWEN_ALLOW_GPU_WITH_BETA:
        logger.warning(
            'Qwen2.5-VL: GPU requested but Beta VLM is active; using CPU. '
            'Set VISION_QWEN_ALLOW_GPU_WITH_BETA=1 to allow GPU (may OOM with YOLO).'
        )
        return 'cpu'
    return raw


def _unload_qwen25vl():
    global _qwen25vl_processor, _qwen25vl_model, _qwen25vl_device
    with _qwen25vl_infer_lock:
        with _qwen25vl_lock:
            try:
                if _qwen25vl_model is not None:
                    del _qwen25vl_model
            except Exception:
                pass
            _qwen25vl_model = None
            _qwen25vl_processor = None
            _qwen25vl_device = None
    _torch_free_cuda_memory()


def _unload_owlv2():
    global _owlv2_processor, _owlv2_model
    with _owlv2_infer_lock:
        with _owlv2_lock:
            try:
                if _owlv2_model is not None:
                    del _owlv2_model
            except Exception:
                pass
            _owlv2_model = None
            _owlv2_processor = None
    _torch_free_cuda_memory()


def _unload_florence2():
    global _florence2_processor, _florence2_model
    with _florence2_infer_lock:
        with _florence2_lock:
            try:
                if _florence2_model is not None:
                    del _florence2_model
            except Exception:
                pass
            _florence2_model = None
            _florence2_processor = None
    _torch_free_cuda_memory()


def _evict_vlms_except(keep):
    """Unload other VLMs to reduce VRAM before loading `keep` (owlv2 | florence2 | qwen)."""
    if not VISION_VLM_EVICT_OTHERS:
        return
    with _vlm_global_lock:
        if keep != 'owlv2':
            _unload_owlv2()
        if keep != 'florence2':
            _unload_florence2()
        if keep != 'qwen':
            _unload_qwen25vl()


def _get_qwen25vl():
    global _qwen25vl_processor, _qwen25vl_model, _qwen25vl_device
    if not _qwen25vl_ready():
        return None, None
    dev = _qwen_resolve_device()
    with _qwen25vl_lock:
        need_reload = _qwen25vl_processor is None or _qwen25vl_device != dev
    if need_reload and _qwen25vl_processor is not None:
        # Must not hold _qwen25vl_lock while unloading (unload takes the same locks).
        _unload_qwen25vl()
    # Evict other VLMs before taking Qwen locks (avoids lock-order inversion with _vlm_global_lock).
    if dev == 'cuda' and VISION_VLM_EVICT_OTHERS:
        _evict_vlms_except('qwen')
    with _qwen25vl_lock:
        if _qwen25vl_processor is None:
            local_dir = _qwen25vl_local_dir()
            if not local_dir:
                return None, None
            try:
                _qwen25vl_processor = _QwenProcessorClass.from_pretrained(
                    local_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    min_pixels=VISION_QWEN_MIN_PIXELS,
                    max_pixels=VISION_QWEN_MAX_PIXELS,
                )
                load_kw = {'local_files_only': True, 'trust_remote_code': True}
                if torch is not None:
                    if dev == 'cuda' and VISION_QWEN_USE_4BIT:
                        load_kw['load_in_4bit'] = True
                        load_kw['device_map'] = 'auto'
                    else:
                        load_kw['dtype'] = torch.float16 if dev == 'cuda' else torch.float32
                load_kw['low_cpu_mem_usage'] = True
                _qwen25vl_model = _QwenModelClass.from_pretrained(local_dir, **load_kw)
                if torch is not None and not (dev == 'cuda' and VISION_QWEN_USE_4BIT):
                    _qwen25vl_model = _qwen25vl_model.to(dev)
                try:
                    _qwen25vl_model.eval()
                except Exception:
                    pass
                _qwen25vl_device = dev
                logger.info(f"Qwen vision model loaded ({dev}) from {local_dir}")
            except Exception as e:
                logger.error(f"Qwen2.5-VL load failed: {e}")
                _qwen25vl_processor = None
                _qwen25vl_model = None
                _qwen25vl_device = None
        return _qwen25vl_processor, _qwen25vl_model


def maybe_unload_qwen_after_incident():
    """Free VRAM/RAM after incident investigation when VISION_QWEN_UNLOAD_AFTER_USE=1."""
    if VISION_QWEN_UNLOAD_AFTER_USE:
        _unload_qwen25vl()


def _local_settings_default():
    return {
        'live_view': {
            'protocol': 'tcp',
            'stream_type': 'main',
            'play_performance': 'balanced',
            'rules': 'disable',
            'pos_osd': 'disable',
            'image_size': 'autofill',
            'auto_start_live': 'no',
            'image_format': 'jpeg',
            'encryption_key': '',
            'fire_frame_point': False,
            'fire_display_point': False,
            'fire_display_highest': False,
            'fire_locate_highest': False,
            'temperature_info': 'disable',
        },
        'record_files': {
            'file_size': '512m',
            'record_path': '',
            'download_path': '',
        },
        'picture_clip': {
            'snapshot_live': '',
            'snapshot_playback': '',
            'clips': '',
        },
        'email_alerts': {
            'enabled': False,
            'sender_email': '',
            'app_password': '',
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
        },
        'engineers': [],
        'plant_map': {
            'center_lat': 28.6139,
            'center_lng': 77.2090,
            'default_zoom': 16,
        },
    }


def _normalize_engineers(engineers):
    out = []
    if not isinstance(engineers, list):
        return out
    for i, raw in enumerate(engineers):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get('name') or '').strip()
        if not name:
            continue
        eid = str(raw.get('id') or '').strip() or f"eng_{i + 1}_{int(time.time()) % 100000}"
        out.append({
            'id': eid[:80],
            'name': name[:120],
            'role': str(raw.get('role') or '').strip()[:120],
            'phone': str(raw.get('phone') or '').strip()[:40],
            'email': str(raw.get('email') or '').strip()[:200],
        })
    return out


def _get_engineers_list():
    local = _system_kv_get('local_settings', _local_settings_default())
    return _normalize_engineers(local.get('engineers') if isinstance(local, dict) else [])


def _engineer_by_id(engineer_id):
    eid = (engineer_id or '').strip()
    if not eid:
        return None
    for eng in _get_engineers_list():
        if eng.get('id') == eid:
            return eng
    return None


def _merge_local_settings(incoming, existing=None):
    """Merge posted local settings; preserve SMTP app password when field left blank."""
    base = _local_settings_default()
    prev = existing if isinstance(existing, dict) else {}
    merged = {}
    for section, defaults in base.items():
        prev_val = prev.get(section)
        new_val = incoming.get(section)
        if section == 'engineers':
            if isinstance(new_val, list):
                merged[section] = _normalize_engineers(new_val)
            elif isinstance(prev_val, list):
                merged[section] = _normalize_engineers(prev_val)
            else:
                merged[section] = []
            continue
        if section == 'plant_map':
            block = dict(defaults)
            if isinstance(prev_val, dict):
                block.update(prev_val)
            if isinstance(new_val, dict):
                block.update(new_val)
            try:
                block['center_lat'] = float(block.get('center_lat') or 28.6139)
                block['center_lng'] = float(block.get('center_lng') or 77.2090)
                block['default_zoom'] = int(block.get('default_zoom') or 16)
            except (TypeError, ValueError):
                block = dict(defaults)
            merged[section] = block
            continue
        block = dict(defaults)
        prev_block = prev_val if isinstance(prev_val, dict) else {}
        new_block = new_val if isinstance(new_val, dict) else {}
        block.update(prev_block)
        block.update(new_block)
        if section == 'email_alerts':
            new_pw = (new_block.get('app_password') or '').strip()
            if not new_pw and (prev_block.get('app_password') or '').strip():
                block['app_password'] = prev_block['app_password']
            try:
                block['smtp_port'] = int(block.get('smtp_port') or 587)
            except (TypeError, ValueError):
                block['smtp_port'] = 587
            block['enabled'] = bool(block.get('enabled'))
            block.pop('receiver_email', None)
            block.pop('subject', None)
        merged[section] = block
    return merged


def _local_settings_for_api(data):
    out = json.loads(json.dumps(data if isinstance(data, dict) else {}))
    ea = out.get('email_alerts')
    if isinstance(ea, dict) and (ea.get('app_password') or '').strip():
        ea['password_set'] = True
        ea['app_password'] = ''
    elif isinstance(ea, dict):
        ea['password_set'] = False
    return out


def _get_email_alert_settings():
    local = _system_kv_get('local_settings', _local_settings_default())
    ea = local.get('email_alerts') if isinstance(local, dict) else {}
    return ea if isinstance(ea, dict) else {}


def _alert_assign_email_subject(alert_row):
    det = (alert_row['detection_type'] or 'Alert').replace('_', ' ').title()
    sev = (alert_row['severity'] or 'medium').title()
    cam = alert_row['camera_name'] or alert_row['camera_id'] or 'Unknown'
    return f'Vision AI Alert — {det} / {sev} ({cam})'


def _alert_assign_email_body(alert_row, engineer_name, alert_id):
    cam = alert_row['camera_name'] or alert_row['camera_id'] or 'Unknown'
    return (
        f"You have been assigned a Vision AI alert.\n\n"
        f"Engineer: {engineer_name}\n"
        f"Alert ID: {alert_id}\n"
        f"Camera: {cam}\n"
        f"Detection type: {alert_row['detection_type']}\n"
        f"Alert: {alert_row['alert_label']}\n"
        f"Severity: {alert_row['severity']}\n"
        f"Time: {alert_row['created_time']}\n\n"
        f"Please review this incident in the Alerts section of the Vision AI platform."
    )


def _send_smtp_email(sender_email, receiver_email, subject, message, app_password,
                     smtp_host='smtp.gmail.com', smtp_port=587):
    sender_email = (sender_email or '').strip()
    receiver_email = (receiver_email or '').strip()
    subject = (subject or 'Vision AI Alert').strip()
    app_password = app_password or ''
    smtp_host = (smtp_host or 'smtp.gmail.com').strip()
    if not sender_email or not receiver_email or not app_password:
        raise ValueError('Sender email, receiver email, and app password are required')
    msg = MIMEText(message or '')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    server = smtplib.SMTP(smtp_host, int(smtp_port or 587), timeout=30)
    try:
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, [receiver_email], msg.as_string())
    finally:
        server.quit()


def _maybe_send_alert_email(camera_id, detection_type, label, severity):
    """Email is sent only when an alert is assigned to an engineer (see api_alerts_assign)."""
    return


def _maybe_unload_beta_vlms_if_idle():
    """If no beta processors are active, release OWLv2 and Florence models."""
    try:
        if _beta_vlm_camera_count() == 0:
            _unload_owlv2()
            _unload_florence2()
    except Exception:
        pass


def _save_alert_event(camera_id, detection_type, frame, detections, severity='medium', meta=None):
    """Persist alert snapshot + metadata. Additive and throttled per camera/type."""
    if frame is None:
        return
    if detections is None:
        detections = []
    now = time.time()
    key = (camera_id, detection_type)
    with _alert_lock:
        last_ts = _alert_last_ts.get(key, 0.0)
        if now - last_ts < ALERT_COOLDOWN_SEC:
            return
        _alert_last_ts[key] = now
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        fname = f"{camera_id}_{detection_type}_{ts}.jpg".replace(":", "_").replace("/", "_")
        path = os.path.join(ALERTS_DIR, fname)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            return
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        labels = [str(d[1]) for d in detections[:4] if isinstance(d, (list, tuple)) and len(d) >= 2]
        label = ", ".join(labels) if labels else f"{detection_type} alert"
        conn = get_db()
        conn.execute(
            'INSERT INTO alert_events (camera_id, detection_type, alert_label, severity, snapshot_path, meta_json) VALUES (?,?,?,?,?,?)',
            (camera_id, detection_type, label[:220], severity, path, json.dumps(meta or {}))
        )
        conn.commit()
        conn.close()
        _maybe_send_alert_email(camera_id, detection_type, label, severity)
    except Exception as e:
        logger.error(f"Alert save error {camera_id}/{detection_type}: {e}")

# ==============================================================
# TRACKING UTILITIES (reused from headcount.py)
# ==============================================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def nms_detections(detections, scores, iou_thresh=0.35):
    """Greedy NMS to suppress duplicate bounding boxes before tracking.
    Keeps the highest-confidence box when two boxes overlap above iou_thresh.
    detections: list of [x1,y1,x2,y2], scores: list of float confidences.
    Returns filtered list of [x1,y1,x2,y2]."""
    if not detections:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(detections[i])
        for j in order:
            if j != i and j not in suppressed:
                if compute_iou(detections[i], detections[j]) >= iou_thresh:
                    suppressed.add(j)
    return keep


class KalmanTracker:
    _id_counter = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / h
        self.kf.x[:4] = np.array([cx, cy, s, r]).reshape((4, 1))

        self.time_since_update = 0
        self.id = KalmanTracker._id_counter
        KalmanTracker._id_counter += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.last_valid_bbox = bbox
        self.position_history = deque(maxlen=30)
        self.position_history.append((cx, cy))

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        s = w * h
        r = w / h
        self.kf.update(np.array([cx, cy, s, r]).reshape((4, 1)))
        self.last_valid_bbox = bbox
        self.position_history.append((cx, cy))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x

    def get_state(self):
        try:
            x, y, s, r = self.kf.x[:4].flatten()
            if any(math.isnan(v) for v in [x, y, s, r]):
                return self.last_valid_bbox
            w = np.sqrt(abs(s) * r)
            h = abs(s) / w
            return [max(0, int(x - w / 2)), max(0, int(y - h / 2)), int(x + w / 2), int(y + h / 2)]
        except:
            return self.last_valid_bbox


class MultiObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1
        self.free_ids = set()
        self.active_ids = set()

    def _get_next_id(self):
        if self.free_ids:
            nid = min(self.free_ids)
            self.free_ids.remove(nid)
            return nid
        nid = self.next_id
        self.next_id += 1
        return nid

    def _release_id(self, tid):
        if tid < self.next_id:
            self.free_ids.add(tid)
        self.active_ids.discard(tid)

    def update(self, detections):
        self.frame_count += 1
        for t in self.trackers:
            t.predict()

        matches, unmatched_dets, unmatched_trks = self._associate(detections)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx])

        for d_idx in unmatched_dets:
            new_id = self._get_next_id()
            tracker = KalmanTracker(detections[d_idx])
            tracker.id = new_id
            self.trackers.append(tracker)
            self.active_ids.add(new_id)

        dead = [i for i, t in enumerate(self.trackers) if t.time_since_update > self.max_age]
        for i in sorted(dead, reverse=True):
            self._release_id(self.trackers[i].id)
            self.trackers.pop(i)

        self.active_ids.clear()
        active = []
        for t in self.trackers:
            if t.time_since_update < 2:
                self.active_ids.add(t.id)
                bbox = t.get_state()
                active.append({
                    'id': t.id,
                    'bbox': bbox,
                    'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    'foot': ((bbox[0] + bbox[2]) // 2, bbox[3])
                })
        return active, self.active_ids

    def _associate(self, detections):
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = compute_iou(det, trk.get_state())

        try:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matches, um_d, um_t = [], list(range(len(detections))), list(range(len(self.trackers)))
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    if r in um_d:
                        um_d.remove(r)
                    if c in um_t:
                        um_t.remove(c)
            return matches, um_d, um_t
        except:
            return [], list(range(len(detections))), list(range(len(self.trackers)))


# ==============================================================
# HEAD COUNT TRACKER — high-accuracy, persistent-ID tracker
# ==============================================================

class HeadCountTracker:
    """Specialized tracker for head-count accuracy in confined spaces.

    Key improvements over the generic MultiObjectTracker:
    - IDs are NEVER recycled: once assigned, an ID is permanent for the
      lifetime of the processor, so the same person always keeps the same ID.
    - Two-stage association: first match to recently updated tracks (strict IoU);
      then match remaining detections to "lost" tracks (recovery) by predicted
      position so bending/walking re-associates to the same ID instead of a new one.
    - When the track has high predicted velocity (person moving), stage-1 IoU
      threshold is relaxed so the same person keeps the same ID while moving.
    - Longer max_age (90 frames) keeps tracks alive through brief occlusions.
    """

    # Recovery: max centre distance (in multiples of mean box diagonal) to re-match
    # a detection to a track that was lost for a few frames (bending/walking).
    RECOVERY_DIST_MULT = 3.0
    # Velocity (pixels/frame) above which we relax IoU for stage-1 matching.
    # Keep conservative: too-loose matching causes ID switches in crowds.
    VELOCITY_RELAX_THRESH = 8.0
    IOU_RELAXED = 0.28   # used when track is moving

    def __init__(self, max_age=90, min_hits=2, iou_threshold=0.45):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self._next_id = 1          # monotonically increasing — never reused

    def _new_id(self):
        nid = self._next_id
        self._next_id += 1
        return nid

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2] (already NMS-filtered).
        Returns (active_list, active_id_set)."""
        self.frame_count += 1

        for t in self.trackers:
            t.predict()

        matches, unmatched_dets, _ = self._associate(detections)

        for d_idx, t_idx in matches:
            self.trackers[t_idx].update(detections[d_idx])

        for d_idx in unmatched_dets:
            tracker = KalmanTracker(detections[d_idx])
            tracker.id = self._new_id()
            self.trackers.append(tracker)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        active = []
        active_ids = set()
        recent_thresh = 2
        for t in self.trackers:
            if t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                if t.time_since_update < recent_thresh:
                    bbox = t.get_state()
                    active.append({
                        'id': t.id,
                        'bbox': bbox,
                        'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                        'foot': ((bbox[0] + bbox[2]) // 2, bbox[3])
                    })
                    active_ids.add(t.id)

        return active, active_ids

    def _associate(self, detections):
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        n_det = len(detections)
        n_trk = len(self.trackers)
        confirmed_idx = [i for i in range(n_trk) if self.trackers[i].time_since_update < 2]
        recovery_idx = [i for i in range(n_trk) if 2 <= self.trackers[i].time_since_update <= self.max_age]

        # ----- Stage 1: match detections to confirmed (recently updated) tracks -----
        # Use IoU + centre distance; relax IoU when track has high velocity (moving person).
        cost1 = np.full((n_det, n_trk), -1e9)
        iou_required = {}
        for t in range(n_trk):
            trk = self.trackers[t]
            vx = float(trk.kf.x[4])
            vy = float(trk.kf.x[5])
            vel = math.sqrt(vx * vx + vy * vy)
            iou_required[t] = self.IOU_RELAXED if vel > self.VELOCITY_RELAX_THRESH else self.iou_threshold

        for d, det in enumerate(detections):
            dcx = (det[0] + det[2]) / 2.0
            dcy = (det[1] + det[3]) / 2.0
            dw = max(det[2] - det[0], 1)
            dh = max(det[3] - det[1], 1)
            for t in confirmed_idx:
                trk = self.trackers[t]
                tb = trk.get_state()
                iou_val = compute_iou(det, tb)
                tcx = (tb[0] + tb[2]) / 2.0
                tcy = (tb[1] + tb[3]) / 2.0
                norm = (math.sqrt(dw * dw + dh * dh) + math.sqrt(max(tb[2] - tb[0], 1) ** 2 + max(tb[3] - tb[1], 1) ** 2)) / 2.0 + 1e-6
                dist_sim = max(0.0, 1.0 - math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2) / norm)
                cost1[d, t] = 0.7 * iou_val + 0.3 * dist_sim

        matched_d, matched_t = set(), set()
        matches = []
        if confirmed_idx:
            try:
                row_ind, col_ind = linear_sum_assignment(-cost1)
                for r, c in zip(row_ind, col_ind):
                    if c not in confirmed_idx:
                        continue
                    det = detections[r]
                    tb = self.trackers[c].get_state()
                    iou_val = compute_iou(det, tb)

                    # Anti-ID-switch gating: avoid an existing ID jumping to a different person.
                    da = max((det[2] - det[0]) * (det[3] - det[1]), 1)
                    ta = max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1)
                    area_ratio = max(da, ta) / max(1, min(da, ta))
                    if area_ratio > 2.5:
                        continue

                    dcx = (det[0] + det[2]) / 2.0
                    dcy = (det[1] + det[3]) / 2.0
                    tcx = (tb[0] + tb[2]) / 2.0
                    tcy = (tb[1] + tb[3]) / 2.0
                    diag = math.sqrt(max(det[2] - det[0], 1) ** 2 + max(det[3] - det[1], 1) ** 2) + 1e-6
                    if math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2) > 1.2 * diag:
                        continue

                    if iou_val >= iou_required[c] and cost1[r, c] >= max(0.35, iou_required[c] * 0.8):
                        matches.append((r, c))
                        matched_d.add(r)
                        matched_t.add(c)
            except Exception:
                pass

        unmatched_dets = [i for i in range(n_det) if i not in matched_d]
        unmatched_trks_recovery = [i for i in recovery_idx if i not in matched_t]

        # ----- Stage 2 (recovery): match unmatched detections to "lost" tracks by position -----
        # When person bends/walks we can lose them for a few frames; re-associate by predicted centre.
        if unmatched_dets and unmatched_trks_recovery:
            cost2 = np.zeros((len(unmatched_dets), len(unmatched_trks_recovery)))
            recovery_radius = []
            for i, t in enumerate(unmatched_trks_recovery):
                trk = self.trackers[t]
                tb = trk.get_state()
                tw = max(tb[2] - tb[0], 1)
                th = max(tb[3] - tb[1], 1)
                recovery_radius.append((i, t, math.sqrt(tw * tw + th * th)))
            for j, d in enumerate(unmatched_dets):
                det = detections[d]
                dcx = (det[0] + det[2]) / 2.0
                dcy = (det[1] + det[3]) / 2.0
                dw = max(det[2] - det[0], 1)
                dh = max(det[3] - det[1], 1)
                det_diag = math.sqrt(dw * dw + dh * dh)
                for i, (_, t, trk_diag) in enumerate(recovery_radius):
                    trk = self.trackers[t]
                    tb = trk.get_state()
                    tcx = (tb[0] + tb[2]) / 2.0
                    tcy = (tb[1] + tb[3]) / 2.0
                    dist = math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2)
                    mean_diag = (det_diag + trk_diag) / 2.0 + 1e-6
                    if dist <= self.RECOVERY_DIST_MULT * mean_diag:
                        cost2[j, i] = 1.0 / (1.0 + dist / mean_diag)
                    else:
                        cost2[j, i] = 0.0
            try:
                row_ind2, col_ind2 = linear_sum_assignment(-cost2)
                for r2, c2 in zip(row_ind2, col_ind2):
                    if cost2[r2, c2] <= 0:
                        continue
                    d_idx = unmatched_dets[r2]
                    _, t_idx, _ = recovery_radius[c2]
                    if t_idx in matched_t:
                        continue
                    matches.append((d_idx, t_idx))
                    matched_d.add(d_idx)
                    matched_t.add(t_idx)
            except Exception:
                pass

        unmatched_dets_final = [i for i in range(n_det) if i not in matched_d]
        unmatched_trks_final = [i for i in range(n_trk) if i not in matched_t]
        return matches, unmatched_dets_final, unmatched_trks_final


# ==============================================================
# GEOMETRY UTILITIES
# ==============================================================

def point_in_polygon(point, polygon):
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ==============================================================
# PRODUCTION INFERENCE INFRASTRUCTURE
# ==============================================================

def _export_pt_to_tensorrt_engine(pt_path, engine_path):
    """Export .pt to TensorRT .engine using app's INFERENCE_IMGSZ and USE_FP16. No accuracy change (FP16).
    Ultralytics writes .engine alongside .pt by default (same dir, same base name)."""
    try:
        logger.info(f"ModelRegistry: Exporting {pt_path} to TensorRT (imgsz={INFERENCE_IMGSZ}, half={USE_FP16}). This may take 2-5 min per model ...")
        m = YOLO(pt_path)
        kwargs = {"format": "engine", "device": 0, "half": USE_FP16, "imgsz": INFERENCE_IMGSZ, "verbose": False}
        # Force-disable dynamo to prevent Ultralytics from passing `dynamo` into torch.onnx.export
        # on stacks where torch.onnx.export doesn't support it.
        #
        # Strategy:
        # - First try with dynamo=False (works when Ultralytics supports/accepts it; prevents default dynamo=True).
        # - If Ultralytics doesn't accept the kwarg, retry without it.
        try:
            m.export(**{**kwargs, "dynamo": False})
        except TypeError as te:
            if "dynamo" in str(te):
                m.export(**kwargs)
            else:
                raise
        if os.path.isfile(engine_path):
            return True
        base = os.path.splitext(os.path.basename(pt_path))[0]
        for d in [os.path.dirname(os.path.abspath(pt_path)), os.getcwd()]:
            alt = os.path.join(d, base + ".engine")
            if os.path.isfile(alt):
                if os.path.abspath(alt) != os.path.abspath(engine_path):
                    import shutil
                    shutil.copy2(alt, engine_path)
                return True
        return False
    except Exception as e:
        err_msg = str(e)
        if "dynamo" in err_msg:
            logger.warning(
                "ModelRegistry: TensorRT export failed (dynamo API mismatch). Using PyTorch .pt. "
                "To avoid this, set VISION_USE_TENSORRT=0 or pin ultralytics to a compatible version."
            )
        else:
            logger.warning(f"ModelRegistry: TensorRT export failed ({e}). Using PyTorch .pt (no accuracy change).")
        return False


class ModelRegistry:
    """Loads each YOLO model exactly once; returns the shared instance + its lock.
    With USE_TENSORRT_IF_AVAILABLE: uses .engine when present, or auto-exports .pt to .engine
    (same imgsz/FP16 as inference). No compromise on accuracy or detection."""
    _models = {}
    _locks = {}
    _init_lock = threading.Lock()

    @staticmethod
    def _is_trt_runtime_mismatch(err):
        """Detect common TensorRT engine/runtime incompatibility signatures."""
        s = str(err).lower()
        markers = (
            "tensorrt model exported with a different version",
            "serialization assertion safeversionread",
            "serialized engine version",
            "create_execution_context",
            "deserialize_cuda_engine",
            "version tag does not match",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _probe_model_runtime(model):
        """Run one tiny dry-run to surface lazy TensorRT backend failures early."""
        side = max(160, int(INFERENCE_IMGSZ))
        probe = np.zeros((side, side, 3), dtype=np.uint8)
        model([probe], conf=0.01, verbose=False, imgsz=side, half=USE_FP16, max_det=1)

    @classmethod
    def get(cls, model_path):
        if model_path not in cls._models:
            with cls._init_lock:
                if model_path not in cls._models:
                    load_path = model_path
                    engine_path = None
                    if USE_TENSORRT_IF_AVAILABLE and model_path.lower().endswith('.pt') and os.path.isfile(model_path):
                        engine_path = model_path[:-3] + '.engine'
                        if os.path.isfile(engine_path):
                            load_path = engine_path
                            logger.info(f"ModelRegistry: Using TensorRT engine {engine_path}")
                        else:
                            if _export_pt_to_tensorrt_engine(model_path, engine_path):
                                load_path = engine_path
                                logger.info(f"ModelRegistry: TensorRT engine ready {engine_path}")
                    if load_path == model_path:
                        logger.info(f"ModelRegistry: Loading {model_path} ...")

                    loaded_model = YOLO(load_path)
                    if load_path.lower().endswith('.engine'):
                        try:
                            cls._probe_model_runtime(loaded_model)
                        except Exception as e:
                            if cls._is_trt_runtime_mismatch(e):
                                logger.warning(
                                    "ModelRegistry: TensorRT engine runtime mismatch for %s (%s). "
                                    "Falling back to .pt and regenerating engine if possible.",
                                    load_path, e
                                )
                                if engine_path and os.path.isfile(engine_path):
                                    try:
                                        os.remove(engine_path)
                                        logger.warning("ModelRegistry: Removed incompatible engine %s", engine_path)
                                    except Exception as rm_e:
                                        logger.warning("ModelRegistry: Could not remove %s (%s)", engine_path, rm_e)

                                # Best-effort rebuild with current runtime, then retry once.
                                reloaded = False
                                if engine_path and _export_pt_to_tensorrt_engine(model_path, engine_path) and os.path.isfile(engine_path):
                                    try:
                                        loaded_model = YOLO(engine_path)
                                        cls._probe_model_runtime(loaded_model)
                                        load_path = engine_path
                                        reloaded = True
                                        logger.info("ModelRegistry: Rebuilt TensorRT engine %s", engine_path)
                                    except Exception as rebuild_e:
                                        logger.warning("ModelRegistry: Rebuilt engine failed (%s). Using .pt.", rebuild_e)

                                if not reloaded:
                                    load_path = model_path
                                    loaded_model = YOLO(model_path)
                            else:
                                raise

                    cls._models[model_path] = loaded_model
                    cls._locks[model_path] = threading.Lock()
                    logger.info(f"ModelRegistry: {load_path} ready")
        return cls._models[model_path], cls._locks[model_path]


class InferenceEngine:
    """Centralized per-detection-type engine.

    One engine per detection type (headcount / entryexit / flapgate).
    Collects the latest frame from every registered camera, runs
    micro-batched YOLO inference through a shared model, and dispatches
    results to per-camera callbacks on a bounded thread pool so the
    inference loop never stalls.
    """

    def __init__(self, model_path, name, target_fps=8, micro_batch=8,
                 imgsz=720, half=False, pool_workers=12, classes=None, nms_iou=0.5):
        self.name = name
        self.model, self.model_lock = ModelRegistry.get(model_path)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.micro_batch = micro_batch
        self.imgsz = imgsz
        self.half = half
        self.pool_workers = pool_workers
        self.classes = classes  # None = all classes; [0] = persons only
        self.nms_iou = nms_iou  # YOLO NMS IoU threshold; lower = fewer duplicate boxes

        self._streams = {}
        self._streams_lock = threading.Lock()

        self.running = False
        self._thread = None

        self.heartbeat = time.time()
        self.total_processed = 0
        self.errors = 0

    def register(self, cam_id, reader, callback, conf=0.5):
        with self._streams_lock:
            self._streams[cam_id] = {
                'reader': reader,
                'callback': callback,
                'conf': conf,
                'last_t': 0,
                'busy': False,
            }

    def unregister(self, cam_id):
        with self._streams_lock:
            self._streams.pop(cam_id, None)

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"IE-{self.name}")
        self._thread.start()
        logger.info(f"InferenceEngine[{self.name}] started")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def stream_count(self):
        return len(self._streams)

    def _dispatch(self, info, frame, dets, masks):
        try:
            info['callback'](frame, dets, masks)
        except Exception as e:
            logger.error(f"IE[{self.name}] callback error: {e}")
            self.errors += 1
        finally:
            info['busy'] = False

    def _run(self):
        pool = ThreadPoolExecutor(max_workers=self.pool_workers,
                                  thread_name_prefix=f"CB-{self.name}")
        try:
            while self.running:
                self.heartbeat = time.time()
                now = time.time()

                with self._streams_lock:
                    snap = [(cid, info) for cid, info in self._streams.items()
                            if not info['busy']
                            and now - info['last_t'] >= self.frame_interval]

                pending = []
                for cid, info in snap:
                    frame = info['reader'].get_frame()
                    if frame is not None:
                        pending.append((cid, frame, info))

                if not pending:
                    time.sleep(0.005)
                    continue

                for i in range(0, len(pending), self.micro_batch):
                    mb = pending[i:i + self.micro_batch]
                    frames = [x[1] for x in mb]
                    confs = [x[2]['conf'] for x in mb]
                    batch_conf = min(confs) if confs else 0.5

                    results = None
                    try:
                        with self.model_lock:
                            results = self.model(
                                frames, conf=batch_conf, classes=self.classes,
                                iou=self.nms_iou, verbose=False, max_det=50,
                                imgsz=self.imgsz, half=self.half)
                    except Exception as e:
                        err_str = str(e)
                        # TensorRT engines are often built with batch=1; run one frame at a time
                        if "not equal to max model size" in err_str or "batch size" in err_str.lower():
                            results = []
                            with self.model_lock:
                                for (cid, frame, info) in mb:
                                    r = self.model(
                                        [frame], conf=info['conf'], classes=self.classes,
                                        iou=self.nms_iou, verbose=False, max_det=50,
                                        imgsz=self.imgsz, half=self.half)
                                    results.append(r[0])
                        else:
                            self.errors += 1
                            logger.error(f"IE[{self.name}] inference: {e}")
                            time.sleep(0.05)
                            continue

                    if results is None:
                        continue

                    try:
                        for j, (cid, frame, info) in enumerate(mb):
                            res = results[j]
                            dets = []
                            masks_list = []
                            has_masks = res.masks is not None
                            h, w = frame.shape[:2]

                            if res.boxes is not None:
                                if has_masks:
                                    mdata = res.masks.data.cpu().numpy()
                                for k, box in enumerate(res.boxes):
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    if box.conf[0].item() < info['conf']:
                                        continue
                                    dets.append([int(x1), int(y1),
                                                 int(x2), int(y2)])
                                    if has_masks:
                                        mr = cv2.resize(mdata[k], (w, h))
                                        masks_list.append(
                                            (mr > 0.5).astype(np.uint8) * 255)

                            info['last_t'] = time.time()
                            info['busy'] = True
                            self.total_processed += 1
                            # Pass raw Results object as third arg when no masks,
                            # so multi-class processors (e.g. PPE) can access class names.
                            third_arg = masks_list if has_masks else res
                            pool.submit(self._dispatch,
                                        info, frame, dets, third_arg)
                    except Exception as e:
                        self.errors += 1
                        logger.error(f"IE[{self.name}] inference: {e}")
                        time.sleep(0.05)
                    finally:
                        if results is not None:
                            del results
                        results = None
        finally:
            pool.shutdown(wait=False)


class CUDACacheSteward:
    """Runs torch.cuda.empty_cache() periodically to curb VRAM creep from PyTorch's allocator cache."""
    _interval = 90
    _running = False
    _thread = None

    @classmethod
    def start(cls):
        if not _torch_cuda_available or cls._running:
            return
        def _loop():
            cls._running = True
            while cls._running:
                time.sleep(cls._interval)
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        cls._thread = threading.Thread(target=_loop, daemon=True, name="CUDACacheSteward")
        cls._thread.start()
        logger.info("CUDACacheSteward started (empty_cache every %ss)", cls._interval)


_engines = {}
_engine_init_lock = threading.Lock()


def get_inference_engine(name, model_path, target_fps=None, classes=None):
    """Lazy-create one InferenceEngine per detection type. Uses resource-tuning constants if target_fps not given."""
    if target_fps is None:
        target_fps = TARGET_FPS
    if name not in _engines:
        with _engine_init_lock:
            if name not in _engines:
                # Headcount uses a tighter YOLO NMS threshold to reduce duplicate boxes
                # on closely spaced heads before the tracker even sees them.
                nms_iou = 0.35 if name == 'headcount' else 0.5
                # PPE (pharmappenew.pt): smaller batch for stability when used with other streams.
                micro_batch = 8 if name == 'ppe' else MICRO_BATCH
                eng = InferenceEngine(
                    model_path, name,
                    target_fps=target_fps,
                    micro_batch=micro_batch,
                    imgsz=INFERENCE_IMGSZ,
                    half=USE_FP16,
                    pool_workers=CALLBACK_POOL_WORKERS,
                    classes=classes,
                    nms_iou=nms_iou,
                )
                eng.start()
                _engines[name] = eng
                logger.info(f"Created InferenceEngine '{name}' (imgsz={INFERENCE_IMGSZ}, fps={target_fps}, fp16={USE_FP16}, classes={classes}, nms_iou={nms_iou})")
    return _engines[name]


class Watchdog:
    """Monitors InferenceEngine heartbeats, auto-restarts stalled engines."""
    STALE_THRESHOLD = 30

    def __init__(self):
        self._engines = []
        self.running = False
        self._thread = None

    def register(self, engine):
        self._engines.append(engine)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="Watchdog")
        self._thread.start()
        logger.info("Watchdog started")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            for eng in self._engines:
                if not eng.running:
                    continue
                age = time.time() - eng.heartbeat
                if age > self.STALE_THRESHOLD and eng.stream_count > 0:
                    logger.warning(
                        f"Watchdog: engine '{eng.name}' heartbeat stale "
                        f"({age:.0f}s), restarting ...")
                    try:
                        eng.stop()
                    except Exception:
                        pass
                    eng.running = False
                    eng.start()
            time.sleep(5)


watchdog = Watchdog()
startup_time = time.time()
_runtime_network_info = {}


def get_local_ip():
    """Prefer the network IP so links work when only that is reachable (e.g. Jupyter on Linux server)."""
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def _runtime_network_context():
    """Summarize the process network namespace for RTSP routing diagnostics."""
    outbound_ip = get_local_ip()
    in_docker = os.path.exists('/.dockerenv')
    bridge_likely = outbound_ip.startswith('172.17.') or outbound_ip.startswith('172.18.')
    return {
        'outbound_ip': outbound_ip,
        'in_docker': in_docker,
        'likely_docker_bridge': bool(in_docker and bridge_likely),
    }


def _probe_tcp_endpoint(host, port, timeout=3.0):
    started = time.time()
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            elapsed_ms = int((time.time() - started) * 1000)
            return {'ok': True, 'elapsed_ms': elapsed_ms}
    except OSError as e:
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            'ok': False,
            'elapsed_ms': elapsed_ms,
            'errno': getattr(e, 'errno', None),
            'error': str(e),
        }


def _is_connectivity_failure(probe):
    err = (probe.get('error') or '').lower()
    errno = probe.get('errno')
    if probe.get('ok'):
        return False
    if errno in (113, 101, 110, 111):
        return True
    return any(token in err for token in (
        'no route to host',
        'network is unreachable',
        'timed out',
        'connection refused',
        'connection reset',
    ))


def _is_route_unreachable(probe):
    err = (probe.get('error') or '').lower()
    errno = probe.get('errno')
    return errno in (113, 101) or 'no route to host' in err or 'network is unreachable' in err


def _log_connectivity_failure_guidance(camera_id, host, port, probe):
    if not _is_connectivity_failure(probe):
        return
    ctx = _runtime_network_context()
    err = (probe.get('error') or 'unknown').lower()
    if ctx['likely_docker_bridge']:
        fix = (
            "Recreate the container with network_mode: host "
            "(docker compose up -d --force-recreate)."
        )
    elif ctx['in_docker']:
        fix = (
            "Docker host networking is active, so this is the same path as running "
            "python3 on the AI server. Ask your network team to allow/route "
            f"TCP {port} from {ctx['outbound_ip']} to {host}. "
            f"Verify on the server: python3 -c \"import socket; "
            f"socket.create_connection(('{host}',{port}),5); print('host OK')\""
        )
    else:
        fix = (
            f"Ask your network team to allow/route TCP {port} from this server to {host}."
        )
    logger.error(
        f"Camera {camera_id}: Cannot reach {host}:{port} from this app process "
        f"(outbound IP {ctx['outbound_ip']}, in_docker={ctx['in_docker']}, "
        f"likely_docker_bridge={ctx['likely_docker_bridge']}, probe_error={err}). "
        f"This is a network reachability issue, not RTSP auth/password. "
        f"172.30.121.x cameras work because that VLAN is reachable from this server. "
        f"{fix}"
    )


def _log_runtime_network_bootstrap():
    global _runtime_network_info
    _runtime_network_info = _runtime_network_context()
    ctx = _runtime_network_info
    logger.info(
        "Runtime network context: outbound_ip=%s, in_docker=%s, likely_docker_bridge=%s",
        ctx['outbound_ip'], ctx['in_docker'], ctx['likely_docker_bridge'],
    )
    if ctx['likely_docker_bridge']:
        logger.error(
            "Docker bridge networking detected (outbound IP %s). "
            "Cross-subnet RTSP cameras such as 192.168.8.x will fail with 'No route to host'. "
            "Recreate the container with network_mode: host.",
            ctx['outbound_ip'],
        )


# ==============================================================
# CAMERA READER — single RTSP connection per camera
# ==============================================================

class CameraReader:
    # FFmpeg/RTSP options for stable HEVC streams over unreliable networks.
    # - tcp: avoids UDP packet loss (root cause of "Could not find ref with POC" errors)
    # - stimeout: socket timeout in microseconds (10s) so open() never hangs forever
    # - max_delay: allow up to 500ms jitter buffer to absorb network bursts
    # - err_detect: ignore non-fatal bitstream errors instead of dropping the frame
    # - reorder_queue_size: larger reorder buffer for HEVC (POC/PPS errors on 1080p/4K streams)
    _RTSP_OPTIONS = (
        "rtsp_transport;tcp|"
        "stimeout;10000000|"
        "max_delay;500000|"
        "err_detect;ignore_err|"
        "reorder_queue_size;3000"
    )
    _RTSP_FALLBACK_OPTIONS = (
        _RTSP_OPTIONS,
        # Some cameras open in standalone viewers but reject the heavier TCP profile above.
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
        "rtsp_transport;udp|stimeout;10000000|max_delay;500000|err_detect;ignore_err",
        "",
    )
    _ffmpeg_env_lock = threading.Lock()
    _MAX_RECONNECT_DELAY = 30   # seconds cap for exponential backoff
    _CONSEC_FAIL_LOG = 5        # log every N consecutive failures to avoid log spam

    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.status = 'initializing'
        self._rtsp_details_logged = False

    def _rtsp_endpoint_details(self):
        """Return non-secret RTSP URL details for diagnostics."""
        try:
            parsed = urlsplit(self.rtsp_url)
            host = parsed.hostname
            port = parsed.port or (554 if parsed.scheme.lower() in ('rtsp', 'rtsps') else None)
            username_present = bool(parsed.username)
            password_present = parsed.password is not None
            password_has_encoded_chars = '%' in (parsed.password or '')
            return {
                'scheme': parsed.scheme or 'unknown',
                'host': host or 'unknown',
                'port': port,
                'path': parsed.path or '/',
                'query_present': bool(parsed.query),
                'username_present': username_present,
                'password_present': password_present,
                'password_has_percent_encoding': password_has_encoded_chars,
                'valid_endpoint': bool(host and port),
            }
        except Exception as e:
            return {
                'scheme': 'unknown',
                'host': 'unknown',
                'port': None,
                'path': '/',
                'query_present': False,
                'username_present': False,
                'password_present': False,
                'password_has_percent_encoding': False,
                'valid_endpoint': False,
                'parse_error': str(e),
            }

    def _probe_rtsp_tcp(self, timeout=3.0):
        details = self._rtsp_endpoint_details()
        if not details['valid_endpoint']:
            return details, {'ok': False, 'error': 'RTSP host/port could not be parsed'}

        probe = _probe_tcp_endpoint(details['host'], details['port'], timeout=timeout)
        if not probe['ok']:
            _log_connectivity_failure_guidance(self.camera_id, details['host'], details['port'], probe)
        return details, probe

    def _log_rtsp_open_diagnostics(self, reason, include_tcp_probe=True):
        details = self._rtsp_endpoint_details()
        logger.info(
            f"Camera {self.camera_id}: RTSP diagnostics ({reason}) - "
            f"scheme={details['scheme']}, host={details['host']}, port={details['port']}, "
            f"path={details['path']}, query_present={details['query_present']}, "
            f"username_present={details['username_present']}, password_present={details['password_present']}, "
            f"password_has_percent_encoding={details['password_has_percent_encoding']}, "
            f"opencv={cv2.__version__}, ffmpeg_options={self._RTSP_OPTIONS}"
        )
        if 'parse_error' in details:
            logger.warning(f"Camera {self.camera_id}: RTSP URL parse error - {details['parse_error']}")

        if include_tcp_probe:
            details, probe = self._probe_rtsp_tcp()
            if probe['ok']:
                logger.info(
                    f"Camera {self.camera_id}: TCP probe OK to {details['host']}:{details['port']} "
                    f"in {probe['elapsed_ms']}ms"
                )
            else:
                logger.warning(
                    f"Camera {self.camera_id}: TCP probe FAILED to {details['host']}:{details['port']} - "
                    f"errno={probe.get('errno')}, elapsed_ms={probe.get('elapsed_ms')}, error={probe.get('error')}"
                )

    def _open_capture_with_options(self, options_label, options):
        prev_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        started = time.time()
        with self._ffmpeg_env_lock:
            if options:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if prev_options is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_options
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)              # absorb HEVC/network jitter on high-res streams (25fps)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)   # 10 s open timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # 10 s per-read timeout
        elapsed_ms = int((time.time() - started) * 1000)
        logger.info(
            f"Camera {self.camera_id}: VideoCapture open completed in {elapsed_ms}ms "
            f"(opened={cap.isOpened()}, backend=CAP_FFMPEG, profile={options_label})"
        )
        return cap

    def _open_capture(self):
        """Open VideoCapture, falling back to alternate RTSP profiles only if needed."""
        last_cap = None
        for idx, options in enumerate(self._RTSP_FALLBACK_OPTIONS, start=1):
            label = f"rtsp_profile_{idx}" if options else "ffmpeg_default"
            cap = self._open_capture_with_options(label, options)
            if cap.isOpened():
                if idx > 1:
                    logger.info(f"Camera {self.camera_id}: Connected using RTSP fallback profile {idx}")
                return cap
            try:
                cap.release()
            except Exception:
                pass
            last_cap = cap
        details = self._rtsp_endpoint_details()
        if details['valid_endpoint']:
            _, probe = self._probe_rtsp_tcp(timeout=1.0)
            if not probe['ok']:
                _log_connectivity_failure_guidance(self.camera_id, details['host'], details['port'], probe)
        return last_cap

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        reconnect_delay = 2     # start with 2 s, doubles on each failure up to _MAX_RECONNECT_DELAY
        consec_fails = 0
        cap = None

        while self.running:
            # --- connect / reconnect ---
            if cap is None or not cap.isOpened():
                self.status = 'reconnecting'
                if consec_fails > 0:
                    delay = min(reconnect_delay * (2 ** (consec_fails - 1)), self._MAX_RECONNECT_DELAY)
                    if consec_fails % self._CONSEC_FAIL_LOG == 1:
                        logger.warning(f"Camera {self.camera_id}: Reconnecting in {delay:.0f}s "
                                       f"(attempt {consec_fails})")
                    time.sleep(delay)
                else:
                    logger.info(f"Camera {self.camera_id}: Connecting to {self.rtsp_url}")

                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                should_log_open_details = (not self._rtsp_details_logged) or (consec_fails % self._CONSEC_FAIL_LOG == 1)
                if should_log_open_details:
                    self._log_rtsp_open_diagnostics(
                        'before open' if not self._rtsp_details_logged else f'retry attempt {consec_fails}',
                        include_tcp_probe=True
                    )
                    self._rtsp_details_logged = True
                cap = self._open_capture()

                if not cap.isOpened():
                    consec_fails += 1
                    if consec_fails == 1 or consec_fails % self._CONSEC_FAIL_LOG == 0:
                        self._log_rtsp_open_diagnostics(f'open failed attempt {consec_fails}', include_tcp_probe=True)
                    continue

                consec_fails = 0
                reconnect_delay = 2
                self.status = 'active'
                logger.info(f"Camera {self.camera_id}: Connected")

            # --- read frame ---
            try:
                ret, frame = cap.read()
                if not ret:
                    # Single failed read: try to drain stale frames before reconnecting
                    consec_fails += 1
                    if consec_fails == 1 or consec_fails % self._CONSEC_FAIL_LOG == 0:
                        try:
                            backend_name = cap.getBackendName() if cap is not None and cap.isOpened() else 'unknown'
                        except Exception:
                            backend_name = 'unknown'
                        logger.warning(
                            f"Camera {self.camera_id}: Frame read failed "
                            f"(consecutive_failures={consec_fails}, backend={backend_name}, "
                            f"capture_opened={cap.isOpened() if cap is not None else False})"
                        )
                    if consec_fails >= 3:
                        logger.warning(
                            f"Camera {self.camera_id}: Releasing capture after {consec_fails} consecutive read failures"
                        )
                        cap.release()
                        cap = None
                    continue

                consec_fails = 0
                self.status = 'active'
                with self.lock:
                    self.frame = frame
                # One-time hint for high-res streams (1080p+): larger imgsz improves accuracy
                if frame is not None and not getattr(self, '_logged_high_res', False):
                    h, w = frame.shape[:2]
                    if w > 1280 or h > 1280:
                        logger.info(
                            f"Camera {self.camera_id}: High-res stream {w}x{h} detected. "
                            "For better accuracy (PPE/heads at distance) set VISION_INFERENCE_IMGSZ=640 or 720."
                        )
                        self._logged_high_res = True

            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Read exception - {e}")
                consec_fails += 1
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self.status = 'stopped'
        logger.info(f"Camera {self.camera_id}: Stopped")

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5.0)


# ==============================================================
# HEAD COUNT PROCESSOR
# ==============================================================

class HeadCountProcessor:
    def __init__(self, camera_id, camera_reader, engine, conf=0.1):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.tracker = HeadCountTracker(max_age=100, min_hits=2, iou_threshold=0.45)

        self.total_entries = 0
        self.seen_ids = set()
        self.current_ids = set()
        self.max_id_seen = 0
        self.stats = {'current_count': 0, 'total_entries': 0, 'max_id': 0, 'status': 'Initializing'}
        self._last_detections = []  # [(bbox, label, color), ...] for combined feed overlay

        self.last_db_save = time.time()
        self.db_interval = 300

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _on_result(self, frame, detections, masks):
        if not self.running:
            return
        try:
            # Suppress duplicate/overlapping boxes before tracking.
            # Use box area as a confidence proxy (larger = more complete detection).
            scores = [(d[2] - d[0]) * (d[3] - d[1]) for d in detections]
            detections = nms_detections(detections, scores, iou_thresh=0.35)

            tracked, current_ids = self.tracker.update(detections)

            new_ids = current_ids - self.seen_ids
            if new_ids:
                self.total_entries += len(new_ids)
                self.seen_ids.update(new_ids)
                for nid in new_ids:
                    if nid > self.max_id_seen:
                        self.max_id_seen = nid

            self.current_ids = current_ids
            self.stats = {
                'current_count': len(current_ids),
                'total_entries': self.total_entries,
                'max_id': self.max_id_seen,
                'status': 'Active'
            }

            vis = frame.copy()
            last_dets = []
            for h in tracked:
                x1, y1, x2, y2 = h['bbox']
                tid = h['id']
                color = ((tid * 50) % 256, (tid * 80 + 100) % 256, (tid * 120 + 50) % 256)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                last_dets.append((tuple(h['bbox']), f"ID:{tid}", color))

            cv2.putText(vis, f"Heads: {len(tracked)}  |  Total: {self.total_entries}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if len(last_dets) > 0:
                _save_alert_event(self.camera_id, 'headcount', vis, last_dets, severity='info',
                                  meta={'current_count': len(current_ids), 'total_entries': self.total_entries})

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"HeadCount {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO headcounts (camera_id, timestamp, current_count, total_entries) VALUES (?,?,?,?)',
                (self.camera_id, ts, self.stats['current_count'], self.stats['total_entries'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"HeadCount DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        """Return list of (bbox, label, color) for combined feed overlay."""
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# ENTRY/EXIT PROCESSOR
# ==============================================================

class EntryExitProcessor:
    def __init__(self, camera_id, camera_reader, entry_zone, exit_zone,
                 canvas_size=(800, 450), engine=None, conf=0.5):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.entry_zone_canvas = entry_zone
        self.exit_zone_canvas = exit_zone
        self.canvas_w, self.canvas_h = canvas_size
        self.entry_zone_video = []
        self.exit_zone_video = []
        self._zones_converted = False
        self.entry_zone_mask = None
        self.exit_zone_mask = None

        # Use HeadCountTracker for persistent IDs and recovery. min_hits=1 so multiple people entering/exiting in one frame are all counted.
        self.tracker = HeadCountTracker(max_age=120, min_hits=1, iou_threshold=0.4)

        self.last_zone = {}
        # Set-based logic so multiple people entering/exiting in the same frame are all counted.
        self.ids_pending_entry = set()   # IDs that have touched exit zone; count when they touch entry
        self.ids_pending_exit = set()     # IDs that have touched entry zone (inside); count when they touch exit
        self.total_entries = 0
        self.total_exits = 0
        self.current_inside = 0
        self.stats = {'entries': 0, 'exits': 0, 'current_inside': 0, 'status': 'Initializing'}

        self.last_db_save = time.time()
        self.db_interval = 300
        self._last_detections = []

    def _convert_zones(self, frame_w, frame_h):
        self.entry_zone_video = [
            (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
            for p in self.entry_zone_canvas
        ]
        self.exit_zone_video = [
            (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
            for p in self.exit_zone_canvas
        ]
        self.entry_zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if len(self.entry_zone_video) >= 3:
            cv2.fillPoly(self.entry_zone_mask, [np.array(self.entry_zone_video, dtype=np.int32)], 255)
        self.exit_zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if len(self.exit_zone_video) >= 3:
            cv2.fillPoly(self.exit_zone_mask, [np.array(self.exit_zone_video, dtype=np.int32)], 255)
        self._zones_converted = True

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _on_result(self, frame, detections, det_masks):
        if not self.running:
            return

        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)

        try:
            tracked, current_ids = self.tracker.update(detections)

            # Strict two-zone order: counter changes only when centroid has touched both zones in order.
            # Set-based logic ensures multiple people entering/exiting in the same frame are all counted.
            # +1: exit first, then entry (person entering). -1: entry first, then exit (person leaving).
            for person in tracked:
                tid = person['id']
                centroid = person['center']
                in_entry = point_in_polygon(centroid, self.entry_zone_video) if len(self.entry_zone_video) >= 3 else False
                in_exit = point_in_polygon(centroid, self.exit_zone_video) if len(self.exit_zone_video) >= 3 else False

                if in_exit:
                    self.ids_pending_entry.add(tid)
                    if tid in self.ids_pending_exit:
                        self.total_exits += 1
                        self.current_inside = max(0, self.current_inside - 1)
                        self.ids_pending_exit.discard(tid)
                    self.last_zone[tid] = 'exit'
                elif in_entry:
                    if tid in self.ids_pending_entry:
                        self.total_entries += 1
                        self.current_inside += 1
                        self.ids_pending_entry.discard(tid)
                    self.last_zone[tid] = 'entry'
                    self.ids_pending_exit.add(tid)
                # else: in neither zone; leave last_zone unchanged

            # Total inside = running count (entry +1, exit -1). Not recomputed from tracked IDs so it stays correct when people move out of entry zone into office or when tracks are lost.
            lost_ids = set(self.last_zone.keys()) - current_ids
            for lid in lost_ids:
                del self.last_zone[lid]
            # Do not subtract from current_inside when we lose track (person may still be inside the office)

            self.stats = {
                'entries': self.total_entries,
                'exits': self.total_exits,
                'current_inside': self.current_inside,
                'status': 'Active'
            }

            vis = frame.copy()
            self._draw_zones(vis)

            last_dets = []
            for person in tracked:
                x1, y1, x2, y2 = person['bbox']
                tid = person['id']
                centroid = person['center']
                # Label from last zone: entry = inside, exit/unknown = outside
                last_z = self.last_zone.get(tid)
                st_display = 'inside' if last_z == 'entry' else 'outside'
                color = (0, 255, 0) if last_z == 'entry' else (0, 165, 255)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"ID:{tid} ({st_display})", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(vis, centroid, 6, (0, 0, 255), -1)
                cv2.circle(vis, centroid, 8, (255, 255, 255), 1)
                last_dets.append((tuple(person['bbox']), f"ID:{tid} ({st_display})", color))

            info_bg = np.zeros((80, 350, 3), dtype=np.uint8)
            vis[5:85, 5:355] = cv2.addWeighted(vis[5:85, 5:355], 0.4, info_bg, 0.6, 0)
            cv2.putText(vis, f"Entries: {self.total_entries}  Exits: {self.total_exits}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Inside: {self.current_inside}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if len(last_dets) > 0:
                sev = 'medium' if (self.total_entries > 0 or self.total_exits > 0) else 'info'
                _save_alert_event(
                    self.camera_id, 'entryexit', vis, last_dets, severity=sev,
                    meta={'entries': self.total_entries, 'exits': self.total_exits, 'inside': self.current_inside}
                )

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"EntryExit {self.camera_id}: Error - {e}")

    def _draw_zones(self, frame):
        if len(self.entry_zone_video) >= 3:
            pts = np.array(self.entry_zone_video, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 150, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (255, 150, 0), 2)
            cx = sum(p[0] for p in self.entry_zone_video) // len(self.entry_zone_video)
            cy = sum(p[1] for p in self.entry_zone_video) // len(self.entry_zone_video)
            cv2.putText(frame, "ENTRY", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        if len(self.exit_zone_video) >= 3:
            pts = np.array(self.exit_zone_video, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            cx = sum(p[0] for p in self.exit_zone_video) // len(self.exit_zone_video)
            cy = sum(p[1] for p in self.exit_zone_video) // len(self.exit_zone_video)
            cv2.putText(frame, "EXIT", (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO entryexit_counts (camera_id, timestamp, entries, exits, current_inside) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, self.total_entries, self.total_exits, self.current_inside)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"EntryExit DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# FLAP GATE TRESPASSING PROCESSOR
# ==============================================================

def _generate_zone_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)))
    return colors

FLAPGATE_ZONE_COLORS = _generate_zone_colors(3)


class FlapGateProcessor:
    def __init__(self, camera_id, camera_reader, gate_zones,
                 canvas_size=(800, 450), engine=None, conf=0.65):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.gate_zones_canvas = gate_zones
        self.canvas_w, self.canvas_h = canvas_size
        self.gate_zones_video = {}
        self._zones_converted = False

        self.person_tracker = {}
        self.next_person_id = 1
        self.zone_occupants = {1: set(), 2: set(), 3: set()}
        self.zone_gate_history = {1: [], 2: [], 3: []}
        self.gate_statuses = {1: 'red', 2: 'red', 3: 'red'}
        self.gate_force_green = {1: False, 2: False, 3: False}
        self.gate_timer_start = {1: 0, 2: 0, 3: 0}
        self.gate_allowed_person = {1: None, 2: None, 3: None}
        self.gate_green_frames = {1: 0, 2: 0, 3: 0}
        self.frame_count = 0
        self.fps_estimate = 25
        self.min_green_frames = 2 * self.fps_estimate

        self.total_trespassing = 0
        self.trespassing_per_gate = {1: 0, 2: 0, 3: 0}
        self.stats = {
            'gate_statuses': {'1': 'red', '2': 'red', '3': 'red'},
            'occupants': {'1': 0, '2': 0, '3': 0},
            'trespassing_total': 0,
            'trespassing_per_gate': {'1': 0, '2': 0, '3': 0},
            'persons_in_frame': 0,
            'status': 'Initializing'
        }

        self.last_db_save = time.time()
        self.db_interval = 300
        self._last_detections = []

    def _convert_zones(self, frame_w, frame_h):
        for zone_id, zone_points in self.gate_zones_canvas.items():
            if zone_points and len(zone_points) >= 3:
                self.gate_zones_video[zone_id] = [
                    (int(p[0] * frame_w / self.canvas_w), int(p[1] * frame_h / self.canvas_h))
                    for p in zone_points
                ]
        self._zones_converted = True

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader,
                             self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _detect_gate_color(self, frame, zone_points):
        if len(zone_points) < 3:
            return 'red'
        height, width = frame.shape[:2]
        zone_mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(zone_points, dtype=np.int32)
        cv2.fillPoly(zone_mask, [pts], 255)
        zone_area = cv2.bitwise_and(frame, frame, mask=zone_mask)
        hsv = cv2.cvtColor(zone_area, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)

        if green_pixels > 50 and green_pixels > red_pixels:
            return 'green'
        return 'red'

    def _on_result(self, frame, detections, det_masks):
        if not self.running:
            return

        h, w = frame.shape[:2]
        if not self._zones_converted:
            self._convert_zones(w, h)

        try:
            display = frame.copy()

            for zone_id, zpts in self.gate_zones_video.items():
                if len(zpts) < 3:
                    continue
                raw_status = self._detect_gate_color(frame, zpts)
                self.zone_gate_history[zone_id].append(raw_status)
                if len(self.zone_gate_history[zone_id]) > 10:
                    self.zone_gate_history[zone_id].pop(0)

                if len(self.zone_gate_history[zone_id]) >= 5:
                    recent = self.zone_gate_history[zone_id][-5:]
                    green_count = recent.count('green')

                    if self.gate_force_green[zone_id]:
                        elapsed = self.frame_count - self.gate_timer_start[zone_id]
                        if elapsed < self.min_green_frames:
                            self.gate_statuses[zone_id] = 'green'
                        else:
                            self.gate_force_green[zone_id] = False
                            self.gate_allowed_person[zone_id] = None
                            self.gate_statuses[zone_id] = 'green' if green_count >= 3 else 'red'
                    else:
                        self.gate_statuses[zone_id] = 'green' if green_count >= 3 else 'red'
                        if self.gate_statuses[zone_id] == 'green' and self.gate_green_frames[zone_id] == 0:
                            self.gate_timer_start[zone_id] = self.frame_count
                            self.gate_force_green[zone_id] = True
                            self.gate_allowed_person[zone_id] = None
                else:
                    self.gate_statuses[zone_id] = raw_status

                if self.gate_statuses[zone_id] == 'green':
                    self.gate_green_frames[zone_id] += 1
                else:
                    self.gate_green_frames[zone_id] = 0
                    self.gate_force_green[zone_id] = False
                    self.gate_allowed_person[zone_id] = None

            for zone_id, zpts in self.gate_zones_video.items():
                if len(zpts) < 3:
                    continue
                pts_arr = np.array(zpts, dtype=np.int32)
                zc = FLAPGATE_ZONE_COLORS[zone_id - 1]
                overlay = display.copy()
                alpha = 0.15 if self.gate_statuses[zone_id] == 'red' else 0.25
                cv2.fillPoly(overlay, [pts_arr], zc)
                display = cv2.addWeighted(display, 1 - alpha, overlay, alpha, 0)
                border_c = (0, 0, 255) if self.gate_statuses[zone_id] == 'red' else (0, 255, 0)
                cv2.polylines(display, [pts_arr], True, border_c, 2)
                for pt in zpts:
                    cv2.circle(display, pt, 4, zc, -1)
                cx = sum(p[0] for p in zpts) // len(zpts)
                cy = sum(p[1] for p in zpts) // len(zpts)
                st = f"Gate {zone_id}: {self.gate_statuses[zone_id].upper()}"
                if self.gate_statuses[zone_id] == 'green' and self.gate_force_green[zone_id]:
                    elapsed = self.frame_count - self.gate_timer_start[zone_id]
                    remaining = max(0, 2.0 - elapsed / self.fps_estimate)
                    st += f" ({remaining:.1f}s)"
                cv2.putText(display, st, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_c, 2)

            current_frame_persons = []
            trespassing_this_frame = set()
            last_dets = []

            # Centroid-based: iterate over detections only (no mask); match persons by bbox IOU
            for i, bbox in enumerate(detections):
                centroid_x = int((bbox[0] + bbox[2]) / 2)
                centroid_y = int((bbox[1] + bbox[3]) / 2)
                centroid = (centroid_x, centroid_y)

                person_id = None
                max_iou = 0.3
                for pid, last_data in self.person_tracker.items():
                    iou = compute_iou(bbox, last_data['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        person_id = pid

                if person_id is None:
                    person_id = self.next_person_id
                    self.next_person_id += 1

                self.person_tracker[person_id] = {
                    'frame': self.frame_count,
                    'centroid': centroid,
                    'bbox': bbox
                }
                current_frame_persons.append(person_id)

                person_in_zones = []
                for zone_id, zpts in self.gate_zones_video.items():
                    if len(zpts) >= 3:
                        pts_arr = np.array(zpts, dtype=np.int32)
                        if cv2.pointPolygonTest(pts_arr, (float(centroid[0]), float(centroid[1])), False) >= 0:
                            person_in_zones.append(zone_id)
                            self.zone_occupants[zone_id].add(person_id)

                for zone_id in person_in_zones:
                    if self.gate_statuses[zone_id] == 'green':
                        if len(self.zone_occupants[zone_id]) > 1:
                            if self.gate_allowed_person[zone_id] is None:
                                self.gate_allowed_person[zone_id] = person_id
                                cx_p = int((bbox[0] + bbox[2]) / 2)
                                cy_p = int(bbox[1]) - 15
                                cv2.putText(display, f"ALLOWED! Gate {zone_id}",
                                            (max(0, cx_p - 80), max(20, cy_p)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            elif person_id != self.gate_allowed_person[zone_id]:
                                self.total_trespassing += 1
                                self.trespassing_per_gate[zone_id] += 1
                                trespassing_this_frame.add(person_id)
                                cx_p = int((bbox[0] + bbox[2]) / 2)
                                cy_p = int(bbox[1]) - 15
                                cv2.putText(display, f"TRESPASSING! Gate {zone_id}",
                                            (max(0, cx_p - 100), max(20, cy_p)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.rectangle(display,
                                              (int(bbox[0]), int(bbox[1])),
                                              (int(bbox[2]), int(bbox[3])),
                                              (0, 0, 255), 3)
                    else:
                        self.gate_allowed_person[zone_id] = None

                box_color = (0, 0, 255) if person_id in trespassing_this_frame else (0, 255, 0)
                cv2.rectangle(display, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)
                cv2.putText(display, f"ID:{person_id}", (centroid[0] - 15, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(display, centroid, 6, (255, 0, 0), -1)
                cv2.circle(display, centroid, 8, (255, 255, 255), 1)
                last_dets.append((tuple(map(int, bbox)), f"ID:{person_id}", box_color))

            for zone_id in self.zone_occupants:
                self.zone_occupants[zone_id] = {
                    pid for pid in self.zone_occupants[zone_id] if pid in current_frame_persons
                }

            info_bg = np.zeros((110, 400, 3), dtype=np.uint8)
            y_start, x_start = 5, 5
            display[y_start:y_start+110, x_start:x_start+400] = cv2.addWeighted(
                display[y_start:y_start+110, x_start:x_start+400], 0.4, info_bg, 0.6, 0
            )
            cv2.putText(display, f"Persons: {len(current_frame_persons)}  Trespassing: {self.total_trespassing}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_pos = 50
            for zid in [1, 2, 3]:
                cnt = len(self.zone_occupants[zid])
                st = self.gate_statuses[zid]
                clr = (0, 0, 255) if st == 'red' else (0, 255, 0)
                txt = f"Gate {zid}: {cnt} person(s) [{st.upper()}]"
                if st == 'green' and self.gate_force_green[zid]:
                    elapsed = self.frame_count - self.gate_timer_start[zid]
                    remaining = max(0, 2.0 - elapsed / self.fps_estimate)
                    txt += f" ({remaining:.1f}s)"
                cv2.putText(display, txt, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
                y_pos += 22

            self.stats = {
                'gate_statuses': {str(k): v for k, v in self.gate_statuses.items()},
                'occupants': {str(k): len(v) for k, v in self.zone_occupants.items()},
                'trespassing_total': self.total_trespassing,
                'trespassing_per_gate': {str(k): v for k, v in self.trespassing_per_gate.items()},
                'persons_in_frame': len(current_frame_persons),
                'status': 'Active'
            }

            with self.lock:
                self.annotated_frame = display
                self._last_detections = last_dets

            if len(last_dets) > 0 and (len(trespassing_this_frame) > 0 or self.total_trespassing > 0):
                _save_alert_event(
                    self.camera_id, 'flapgate', display, last_dets, severity='high',
                    meta={'trespassing_frame': len(trespassing_this_frame), 'trespassing_total': self.total_trespassing}
                )

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                to_remove = [pid for pid, d in self.person_tracker.items()
                             if self.frame_count - d['frame'] > 30]
                for pid in to_remove:
                    del self.person_tracker[pid]
                    for zid in self.zone_occupants:
                        self.zone_occupants[zid].discard(pid)

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"FlapGate {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO flapgate_events (camera_id, timestamp, trespassing_count, persons_in_frame, gate1_status, gate2_status, gate3_status) VALUES (?,?,?,?,?,?,?)',
                (self.camera_id, ts, self.total_trespassing, self.stats['persons_in_frame'],
                 self.gate_statuses.get(1, 'red'), self.gate_statuses.get(2, 'red'), self.gate_statuses.get(3, 'red'))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"FlapGate DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# PPE SAHI HELPERS (optional sliced inference for small/far people)
# ==============================================================

def _SAHI_empty_result():
    """Result-like object with no boxes for _on_result when SAHI returns nothing."""
    torch = __import__('torch')
    class _Boxes:
        xyxy = torch.tensor([]).reshape(0, 4)
        cls = torch.tensor([], dtype=torch.long)
        conf = torch.tensor([])
        def __len__(self):
            return 0
    class _R:
        boxes = _Boxes()
        names = {}
    return _R()

def _SAHI_result_from_lists(xyxy_list, cls_list, conf_list, names):
    """Build Result-like object from SAHI prediction lists for _on_result."""
    torch = __import__('torch')
    if not xyxy_list:
        return _SAHI_empty_result()
    class _Boxes:
        xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
        cls = torch.tensor(cls_list, dtype=torch.long)
        conf = torch.tensor(conf_list, dtype=torch.float32)
        def __len__(self):
            return self.xyxy.shape[0]
    names_dict = dict(names) if names else {}
    class _R:
        boxes = _Boxes()
        names = names_dict
    return _R()


# ==============================================================
# PPE DETECTION PROCESSOR (pharmappenew.pt via InferenceEngine or SAHI)
# ==============================================================

def _draw_premium_ppe_box(frame, x1, y1, x2, y2, label, color_bgr):
    """Draw a premium, readable bounding box and label for PPE (enterprise-grade look)."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Subtle outer border for depth and readability on any background
    border_thick = 2
    shadow = (30, 30, 30)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), shadow, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, border_thick)
    # Label: pill-style background for clarity
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness_text = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness_text)
    pad_x, pad_y = 10, 5
    lx1 = x1
    ly1 = max(y1 - th - 2 * pad_y, 4)
    lx2 = min(x1 + tw + 2 * pad_x, frame.shape[1] - 2)
    ly2 = y1
    # Filled label background (same color as box, slightly transparent look via border)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color_bgr, -1)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 1)
    # White text for high contrast and readability
    text_color = (255, 255, 255)
    cv2.putText(frame, label, (x1 + pad_x, y1 - pad_y - 2), font, font_scale, text_color, thickness_text, cv2.LINE_AA)

class PPEDetectionProcessor:
    """Detects PPE compliance (helmets, vests, etc.) using pharmappenew.pt.
    Annotates frames with bounding boxes, class labels and a compliance summary.
    Tracks compliant vs. violation counts and persists to ppe_events table."""

    # Class IDs that indicate a PPE *violation* (no helmet / no hard hat etc.) — drawn in red bbox.
    VIOLATION_KEYWORDS = ('no-helmet', 'no-hardhat', 'no_hard_hat', 'no-gloves',
                          'no-mask', 'no-glasses', 'no-boots', 'no-earmuff',
                          'no_helmet', 'no_hardhat', 'no_gloves',
                          'no_mask', 'no_glasses', 'no_boots', 'no_earmuff',
                          'no hard hat', 'no hardhat',
                          'violation', 'non-compliant', 'noncompliant')
    # Require higher confidence for violations to reduce false positives (e.g. false "No vest").
    VIOLATION_CONF_MIN = 0.1
    # Min bbox area (pixels²) to count as violation; smaller = far/tiny = skip to avoid false no-vest.
    MIN_VIOLATION_AREA = 2500

    def __init__(self, camera_id, camera_reader, engine, conf=0.4):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()

        self.compliant_count = 0
        self.violation_count = 0
        self.total_persons = 0
        self.stats = {
            'compliant_count': 0,
            'violation_count': 0,
            'total_persons': 0,
            'status': 'Initializing'
        }

        self.last_db_save = time.time()
        self.db_interval = 300
        self.use_sahi = USE_SAHI_PPE and _SAHI_AVAILABLE
        self._sahi_thread = None
        self._last_detections = []

    def start(self):
        self.running = True
        if self.use_sahi and get_sliced_prediction and AutoDetectionModel:
            self._sahi_thread = threading.Thread(target=self._sahi_loop, daemon=True)
            self._sahi_thread.start()
        else:
            self.engine.register(self.camera_id, self.camera_reader,
                                self._on_result, self.conf)

    def stop(self):
        self.running = False
        if self.use_sahi:
            if self._sahi_thread is not None:
                self._sahi_thread.join(timeout=5.0)
        else:
            self.engine.unregister(self.camera_id)

    def _sahi_loop(self):
        """Run PPE inference via SAHI (sliced) for small/far people when VISION_PPE_USE_SAHI=1."""
        try:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=PPE_MODEL_PATH,
                confidence_threshold=self.conf,
                device=device,
            )
        except Exception as e:
            logger.error(f"PPE SAHI init {self.camera_id}: {e}")
            return
        last_t = 0
        frame_interval = 1.0 / 9.0
        while self.running:
            try:
                now = time.time()
                if now - last_t < frame_interval:
                    time.sleep(0.02)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.04)
                    continue
                last_t = now
                result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=720,
                    slice_width=720,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )
                if not result or not getattr(result, 'object_prediction_list', None):
                    self._on_result(frame, [], _SAHI_empty_result())
                    continue
                preds = result.object_prediction_list
                xyxy_list = []
                cls_list = []
                conf_list = []
                names = {}
                for p in preds:
                    b = p.bbox
                    xyxy_list.append([b.minx, b.miny, b.maxx, b.maxy])
                    cid = getattr(p.category, 'id', 0)
                    cls_list.append(cid)
                    names[cid] = getattr(p.category, 'name', str(cid))
                    conf_list.append(float(p.score.value))
                import torch as _torch
                raw = _SAHI_result_from_lists(xyxy_list, cls_list, conf_list, names)
                dets = [[int(a), int(b), int(c), int(d)] for a, b, c, d in xyxy_list]
                self._on_result(frame, dets, raw)
            except Exception as e:
                logger.error(f"PPE SAHI {self.camera_id}: {e}")
                time.sleep(0.1)

    def _is_violation(self, class_name):
        name_lower = class_name.lower()
        return any(kw in name_lower for kw in self.VIOLATION_KEYWORDS)

    def _is_vest_blue(self, frame, xyxy):
        """Check if the vest region in the bbox is blue/shade of blue (BGR frame). Returns True for Vest, False for No vest."""
        try:
            h_img, w_img = frame.shape[:2]
            x1 = max(0, min(int(xyxy[0]), w_img - 1))
            y1 = max(0, min(int(xyxy[1]), h_img - 1))
            x2 = max(0, min(int(xyxy[2]), w_img))
            y2 = max(0, min(int(xyxy[3]), h_img))
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                return False
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Blue in OpenCV HSV: H ~100-130 (0-180), S and V reasonable
            h, s, v = cv2.split(hsv)
            mask_bright = (v > 30) & (s > 25)
            if not np.any(mask_bright):
                return False
            h_mean = np.mean(h[mask_bright])
            s_mean = np.mean(s[mask_bright])
            # Blue hue range (shades of blue): ~95 to 135
            if 95 <= h_mean <= 135 and s_mean > 25:
                return True
            return False
        except Exception:
            return False

    def _on_result(self, frame, detections, raw_result):
        if not self.running:
            return
        try:
            vis = frame.copy()

            compliant = 0
            violations = 0
            boxes_with_cls = []

            # raw_result is the YOLO Results object (passed by InferenceEngine when no masks).
            if raw_result is not None and hasattr(raw_result, 'boxes') and raw_result.boxes is not None:
                names = raw_result.names if hasattr(raw_result, 'names') else {}
                boxes_data = raw_result.boxes
                for i in range(len(boxes_data)):
                    try:
                        xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes_data.cls[i].cpu().numpy())
                        conf_val = float(boxes_data.conf[i].cpu().numpy())
                        cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                        cn = cls_name.lower()
                        # Omit NO_vest entirely: do not detect or show "No vest".
                        if 'vest' in cn and ('no' in cn or 'no_vest' in cn or 'no-vest' in cn):
                            continue
                        # Vest: only show bbox for blue vests; skip yellow/orange (no bbox).
                        if 'vest' in cn and not self._is_vest_blue(vis, xyxy):
                            continue
                        is_viol_class = self._is_violation(cls_name)
                        if is_viol_class:
                            if conf_val < max(self.conf, self.VIOLATION_CONF_MIN):
                                continue
                            # Skip tiny/far boxes for violations to avoid false positives on small people.
                            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                            if area < self.MIN_VIOLATION_AREA:
                                continue
                        else:
                            if conf_val < self.conf:
                                continue
                        if 'glove' in cn or 'boot' in cn:
                            continue
                        boxes_with_cls.append((xyxy, cls_id, cls_name, conf_val))
                    except Exception:
                        continue
            else:
                # Fallback when raw_result is a masks list (shouldn't happen for PPE but handled safely)
                for bbox in detections:
                    boxes_with_cls.append((bbox, 0, 'detection', 0.0))

            last_dets = []
            for (xyxy, cls_id, cls_name, conf_val) in boxes_with_cls:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                display_name = cls_name
                # NO hard Hat / no hardhat etc. are violations (red bbox). Vest "No vest" is not shown (filtered above).
                is_viol = self._is_violation(cls_name)
                # Vest (compliant): only show Vest when blue; other vest classes already filtered
                if 'vest' in cls_name.lower():
                    if self._is_vest_blue(vis, xyxy):
                        display_name = 'Vest'
                        is_viol = False
                if is_viol:
                    violations += 1
                    color = (0, 0, 255)   # BGR red for violations
                    label = f"{display_name} {conf_val:.2f}"
                else:
                    compliant += 1
                    color = (0, 180, 90)   # BGR emerald green for compliant
                    label = f"{display_name} {conf_val:.2f}"

                _draw_premium_ppe_box(vis, x1, y1, x2, y2, label, color)
                last_dets.append(((x1, y1, x2, y2), label, color))

            total = compliant + violations
            self.compliant_count = compliant
            self.violation_count = violations
            self.total_persons = total

            self.stats = {
                'compliant_count': compliant,
                'violation_count': violations,
                'total_persons': total,
                'status': 'Active'
            }

            # Premium info strip: dark semi-transparent panel with clear typography
            info_h, info_w = 72, 400
            info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
            info_bg[:] = (28, 28, 32)
            vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
            cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
            cv2.putText(vis, f"PPE  |  Compliant: {compliant}   Violations: {violations}",
                        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            viol_color = (0, 0, 255) if violations > 0 else (0, 200, 100)
            cv2.putText(vis, f"Total: {total}",
                        (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, viol_color, 2, cv2.LINE_AA)

            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if violations > 0:
                _save_alert_event(
                    self.camera_id, 'ppe', vis, last_dets, severity='high',
                    meta={'violations': violations, 'compliant': compliant, 'total': total}
                )

            now = time.time()
            if now - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = now

        except Exception as e:
            logger.error(f"PPE {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO ppe_events (camera_id, timestamp, compliant_count, violation_count, total_persons) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, self.stats['compliant_count'],
                 self.stats['violation_count'], self.stats['total_persons'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"PPE DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


class FireSmokeDetectionProcessor:
    """Detect fire/smoke events using dedicated model weights."""
    FIRE_KEYWORDS = ('fire', 'flame')
    SMOKE_KEYWORDS = ('smoke',)

    def __init__(self, camera_id, camera_reader, engine, conf=0.35):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.engine = engine
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self.stats = {'fire_count': 0, 'smoke_count': 0, 'total': 0, 'status': 'Initializing'}
        self.last_db_save = time.time()
        self.db_interval = 120
        self._last_detections = []

    def start(self):
        self.running = True
        self.engine.register(self.camera_id, self.camera_reader, self._on_result, self.conf)

    def stop(self):
        self.running = False
        self.engine.unregister(self.camera_id)

    def _kind(self, cls_name):
        n = (cls_name or '').lower()
        if any(k in n for k in self.FIRE_KEYWORDS):
            return 'fire'
        if any(k in n for k in self.SMOKE_KEYWORDS):
            return 'smoke'
        return ''

    def _on_result(self, frame, detections, raw_result):
        if not self.running:
            return
        try:
            vis = frame.copy()
            names = raw_result.names if (raw_result is not None and hasattr(raw_result, 'names')) else {}
            boxes_data = raw_result.boxes if (raw_result is not None and hasattr(raw_result, 'boxes')) else None
            fire_count, smoke_count = 0, 0
            last_dets = []
            if boxes_data is not None:
                for i in range(len(boxes_data)):
                    try:
                        xyxy = boxes_data.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes_data.cls[i].cpu().numpy())
                        conf_val = float(boxes_data.conf[i].cpu().numpy())
                        if conf_val < self.conf:
                            continue
                        cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                        kind = self._kind(cls_name)
                        if not kind:
                            continue
                        x1, y1, x2, y2 = map(int, xyxy)
                        if kind == 'fire':
                            fire_count += 1
                            color = (0, 0, 255)
                            label = f"Fire {conf_val:.2f}"
                        else:
                            smoke_count += 1
                            color = (0, 165, 255)
                            label = f"Smoke {conf_val:.2f}"
                        _draw_premium_ppe_box(vis, x1, y1, x2, y2, label, color)
                        last_dets.append(((x1, y1, x2, y2), label, color))
                    except Exception:
                        continue
            total = fire_count + smoke_count
            self.stats = {'fire_count': fire_count, 'smoke_count': smoke_count, 'total': total, 'status': 'Active'}
            with self.lock:
                self.annotated_frame = vis
                self._last_detections = last_dets

            if total > 0:
                _save_alert_event(
                    self.camera_id, 'fire_smoke', vis, last_dets, severity='high',
                    meta={'fire_count': fire_count, 'smoke_count': smoke_count, 'total': total}
                )
            if time.time() - self.last_db_save >= self.db_interval:
                self._save_to_db()
                self.last_db_save = time.time()
        except Exception as e:
            logger.error(f"FireSmoke {self.camera_id}: Error - {e}")

    def _save_to_db(self):
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                'INSERT INTO fire_smoke_events (camera_id, timestamp, fire_count, smoke_count, total_detections) VALUES (?,?,?,?,?)',
                (self.camera_id, ts, int(self.stats['fire_count']), int(self.stats['smoke_count']), int(self.stats['total']))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"FireSmoke DB save error: {e}")

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


# ==============================================================
# ANPR DETECTION PROCESSOR (LibreYOLO vehicle + plate + speed)
# ==============================================================

ANPR_VEHICLE_CLASS_NAMES = {
    'bicycle', 'bike', 'bus', 'car', 'motorbike', 'motorcycle', 'truck', 'van'
}


def _anpr_resolve_weight_path(weight_path):
    candidates = []
    if weight_path:
        candidates.append(weight_path)
    if weight_path:
        base_name = os.path.basename(weight_path)
        if os.path.isabs(weight_path):
            candidates.extend([
                os.path.join(BASE_DIR, base_name),
                os.path.join(BASE_DIR, 'models', base_name),
                os.path.join(BASE_DIR, 'weights', base_name),
            ])
        else:
            candidates.extend([
                os.path.join(BASE_DIR, weight_path),
                os.path.join(BASE_DIR, 'models', base_name),
                os.path.join(BASE_DIR, 'weights', base_name),
            ])
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[0] if candidates else weight_path


def _anpr_class_label(names, cls_id):
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id)))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _anpr_vehicle_class_filter(names):
    class_ids = []
    items = names.items() if isinstance(names, dict) else enumerate(names or [])
    for cls_id, name in items:
        if str(name).lower() in ANPR_VEHICLE_CLASS_NAMES:
            class_ids.append(int(cls_id))
    return class_ids or None


def _anpr_color_for_id(track_id):
    rng = np.random.default_rng(int(track_id))
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def _anpr_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


class ANPRModelRegistry:
    _plate_model = None
    _vehicle_model = None
    _ocr_model = None
    _lock = threading.Lock()
    _infer_lock = threading.Lock()

    @classmethod
    def get(cls):
        if not _LIBREYOLO_AVAILABLE:
            raise RuntimeError('LibreYOLO is not installed. Install the libreyolo package for ANPR.')
        if not _PADDLEOCR_AVAILABLE:
            raise RuntimeError('PaddleOCR is not installed. Install paddleocr for ANPR plate reading.')

        plate_path = _anpr_resolve_weight_path(ANPR_PLATE_MODEL_PATH)
        vehicle_path = _anpr_resolve_weight_path(ANPR_VEHICLE_MODEL_PATH)
        if not os.path.isfile(plate_path):
            raise RuntimeError(f'ANPR plate model not found: {plate_path}')
        if not os.path.isfile(vehicle_path):
            raise RuntimeError(f'ANPR vehicle model not found: {vehicle_path}')

        with cls._lock:
            if cls._plate_model is None:
                cls._plate_model = _LibreYOLO(plate_path)
                fuse_fn = getattr(cls._plate_model, 'fuse', None)
                if callable(fuse_fn):
                    fuse_fn()
                logger.info('ANPR plate LibreYOLO loaded: %s', plate_path)
            if cls._vehicle_model is None:
                cls._vehicle_model = _LibreYOLO(vehicle_path)
                fuse_fn = getattr(cls._vehicle_model, 'fuse', None)
                if callable(fuse_fn):
                    fuse_fn()
                logger.info('ANPR vehicle LibreYOLO loaded: %s', vehicle_path)
            if cls._ocr_model is None:
                cls._ocr_model = _PaddleOCR(
                    use_textline_orientation=True,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    enable_mkldnn=False,
                    lang='en',
                )
                logger.info('ANPR PaddleOCR loaded')
        return cls._plate_model, cls._vehicle_model, cls._ocr_model, cls._infer_lock


class ANPRPlateOcrCache:
    def __init__(self):
        self.entries = []

    def get(self, box, frame_idx):
        best_entry, best_iou = None, 0.0
        for entry in self.entries:
            score = _anpr_iou(box, entry['box'])
            if score > best_iou:
                best_iou, best_entry = score, entry
        if best_entry is None or best_iou < 0.35:
            return '', True
        best_entry['box'] = box
        best_entry['last_seen'] = frame_idx
        text = best_entry.get('text', '')
        should_refresh = (not text or frame_idx - int(best_entry.get('ocr_frame', 0)) >= ANPR_OCR_REFRESH_FRAMES)
        return text, should_refresh

    def update(self, box, text, frame_idx):
        best_entry, best_iou = None, 0.0
        for entry in self.entries:
            score = _anpr_iou(box, entry['box'])
            if score > best_iou:
                best_iou, best_entry = score, entry
        if best_entry is not None and best_iou >= 0.35:
            best_entry['box'] = box
            best_entry['last_seen'] = frame_idx
            best_entry['ocr_frame'] = frame_idx
            if text:
                best_entry['text'] = text
        else:
            self.entries.append({'box': box, 'text': text, 'last_seen': frame_idx, 'ocr_frame': frame_idx})
        self.entries = [
            entry for entry in self.entries
            if frame_idx - int(entry.get('last_seen', frame_idx)) <= ANPR_OCR_REFRESH_FRAMES
        ]


class ANPRDetectionProcessor:
    def __init__(self, camera_id, camera_reader, conf=ANPR_VEHICLE_CONF, speed_threshold_kmh=50.0, meters_per_pixel=ANPR_METERS_PER_PIXEL):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.conf = float(conf)
        self.speed_threshold_kmh = float(speed_threshold_kmh)
        self.meters_per_pixel = float(meters_per_pixel)
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._thread = None
        self._last_detections = []
        self.stats = {'vehicles': 0, 'plates': 0, 'overspeeding': 0, 'status': 'Initializing'}
        self.track_history = defaultdict(list)
        self.last_points = {}
        self.last_frames = {}
        self.last_times = {}
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        self.speeds = {}
        self.plates = {}
        self.vehicle_types = {}
        self.last_event_save = {}
        self.ocr_cache = ANPRPlateOcrCache()
        self.cached_plates = []

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'ANPR-{self.camera_id}')
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)

    def _ocr_plate_text(self, plate_crop, ocr_model):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ocr_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = ocr_model.predict(ocr_input)
        texts = []
        if result:
            ocr_res = result[0]
            rec_texts = ocr_res.get('rec_texts', []) if isinstance(ocr_res, dict) else []
            rec_scores = ocr_res.get('rec_scores', []) if isinstance(ocr_res, dict) else []
            for txt, score in zip(rec_texts, rec_scores):
                if score > 0.4:
                    texts.append(txt)
        plate_text = ' '.join(texts)
        return ''.join(ch for ch in plate_text if ch.isalnum()).upper()

    def _detect_plates(self, frame_bgr, plate_model, ocr_model, frame_idx):
        result = plate_model(
            frame_bgr, conf=ANPR_PLATE_CONF, iou=ANPR_IOU, imgsz=ANPR_IMGSZ,
            max_det=ANPR_MAX_DET, save=False, color_format='bgr'
        )
        boxes = getattr(result, 'boxes', None)
        plates = []
        if boxes is None or len(boxes) == 0:
            return plates
        h, w = frame_bgr.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        ocr_count = 0
        for box, conf in zip(xyxy, conf_arr):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            plate_box = (x1, y1, x2, y2)
            text, should_ocr = self.ocr_cache.get(plate_box, frame_idx)
            crop = frame_bgr[y1:y2, x1:x2]
            if should_ocr and crop.size > 0 and ocr_count < ANPR_MAX_OCR_PLATES_PER_FRAME:
                try:
                    text = self._ocr_plate_text(crop, ocr_model)
                    self.ocr_cache.update(plate_box, text, frame_idx)
                    ocr_count += 1
                except Exception as e:
                    logger.warning('ANPR OCR %s: %s', self.camera_id, e)
            plates.append({
                'box': plate_box,
                'text': text or 'PLATE',
                'conf': float(conf),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        return plates

    def _associate_plate(self, vehicle_box, plates):
        x1, y1, x2, y2 = vehicle_box
        best_plate, best_score = None, -1.0
        for plate in plates:
            cx, cy = plate['center']
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            px1, py1, px2, py2 = plate['box']
            score = float(plate['conf']) + max(1, (px2 - px1) * (py2 - py1)) / 100000.0
            if score > best_score:
                best_score, best_plate = score, plate
        if best_plate is None:
            return ''
        return '' if best_plate['text'] == 'PLATE' else best_plate['text']

    def _speed_kmh(self, track_id, point, frame_idx, fps, current_time):
        if track_id not in self.last_points:
            return None
        prev_point = self.last_points[track_id]
        prev_frame = self.last_frames[track_id]
        frame_delta = max(1, frame_idx - prev_frame)
        if frame_delta < ANPR_SPEED_SAMPLE_FRAMES:
            return None
        prev_time = self.last_times.get(track_id)
        seconds = (current_time - prev_time) if prev_time is not None else (frame_delta / max(1.0, float(fps or 30.0)))
        seconds = max(1e-3, seconds)
        pixel_distance = float(np.hypot(point[0] - prev_point[0], point[1] - prev_point[1]))
        speed = (pixel_distance * self.meters_per_pixel / seconds) * 3.6
        self.speed_history[track_id].append(speed)
        return float(np.mean(self.speed_history[track_id]))

    def _save_event(self, track_id, vehicle_type, plate_text, speed, conf):
        now = time.time()
        key = int(track_id)
        last = self.last_event_save.get(key, 0)
        if now - last < ANPR_EVENT_SAVE_INTERVAL_SEC:
            return
        if not plate_text and speed is None:
            return
        overspeed = bool(speed is not None and speed > self.speed_threshold_kmh)
        try:
            conn = get_db()
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                '''INSERT INTO anpr_events
                   (camera_id, timestamp, track_id, vehicle_type, license_plate, speed_kmh, speed_threshold_kmh, overspeeding, confidence)
                   VALUES (?,?,?,?,?,?,?,?,?)''',
                (
                    self.camera_id, ts, int(track_id), str(vehicle_type or ''),
                    str(plate_text or ''), float(speed) if speed is not None else None,
                    float(self.speed_threshold_kmh), 1 if overspeed else 0, float(conf or 0.0),
                )
            )
            conn.commit()
            conn.close()
            self.last_event_save[key] = now
        except Exception as e:
            logger.error('ANPR DB save error: %s', e)

    def _run(self):
        try:
            plate_model, vehicle_model, ocr_model, infer_lock = ANPRModelRegistry.get()
            vehicle_names = getattr(vehicle_model, 'names', {})
            vehicle_classes = _anpr_vehicle_class_filter(vehicle_names)
            fps = max(1.0, ANPR_TARGET_FPS)
            tracker = _LibreByteTracker(
                config=_LibreTrackConfig(
                    track_high_thresh=max(0.05, self.conf),
                    track_low_thresh=0.1,
                    new_track_thresh=max(0.05, self.conf),
                    match_thresh=0.8,
                    track_buffer=30,
                    frame_rate=max(1, int(fps)),
                    minimum_consecutive_frames=1,
                )
            )
            self.stats['status'] = 'Active'
        except Exception as e:
            self.stats['status'] = f'Init error: {e}'
            logger.error('ANPR %s: init error - %s', self.camera_id, e)
            return

        frame_idx = 0
        last_t = 0.0
        frame_interval = 1.0 / max(1.0, ANPR_TARGET_FPS)
        while self.running:
            now = time.time()
            if now - last_t < frame_interval:
                time.sleep(0.01)
                continue
            frame = self.camera_reader.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            last_t = now
            frame_idx += 1
            try:
                with infer_lock:
                    vehicle_result = vehicle_model(
                        frame, conf=self.conf, iou=ANPR_IOU, imgsz=ANPR_IMGSZ,
                        classes=vehicle_classes, max_det=ANPR_MAX_DET,
                        save=False, color_format='bgr'
                    )
                    tracked_result = tracker.update(vehicle_result)
                    if frame_idx == 1 or frame_idx % ANPR_PLATE_DETECT_INTERVAL == 0:
                        self.cached_plates = self._detect_plates(frame, plate_model, ocr_model, frame_idx)

                vis = frame.copy()
                h, w = vis.shape[:2]
                line_y = int(h * 0.55)
                cv2.line(vis, (80, line_y), (max(80, w - 80), line_y), (255, 0, 255), 2)
                cv2.putText(vis, f'ANPR speed limit: {self.speed_threshold_kmh:.0f} km/h',
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                for plate in self.cached_plates:
                    x1, y1, x2, y2 = plate['box']
                    _draw_premium_ppe_box(vis, x1, y1, x2, y2, f"{plate['text']} {plate['conf']:.2f}", (0, 160, 0))

                boxes = getattr(tracked_result, 'boxes', None)
                track_ids = getattr(tracked_result, 'track_id', None)
                last_dets = []
                vehicles, overspeeding = 0, 0
                plate_count = 0
                if boxes is not None and len(boxes) > 0 and track_ids is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    ids = track_ids.cpu().numpy().astype(int)
                    for box, conf_val, cls_id, track_id in zip(xyxy, confs, classes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        vehicles += 1
                        vehicle_type = _anpr_class_label(vehicle_names, int(cls_id))
                        self.vehicle_types[int(track_id)] = vehicle_type
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        bottom_center = ((x1 + x2) / 2, y2)
                        speed = self._speed_kmh(int(track_id), bottom_center, frame_idx, fps, time.time())
                        if speed is not None:
                            self.speeds[int(track_id)] = speed
                            self.last_points[int(track_id)] = bottom_center
                            self.last_frames[int(track_id)] = frame_idx
                            self.last_times[int(track_id)] = time.time()
                        if int(track_id) not in self.last_points:
                            self.last_points[int(track_id)] = bottom_center
                            self.last_frames[int(track_id)] = frame_idx
                            self.last_times[int(track_id)] = time.time()

                        plate_text = self._associate_plate((x1, y1, x2, y2), self.cached_plates)
                        if plate_text:
                            self.plates[int(track_id)] = plate_text
                        plate_label = self.plates.get(int(track_id), '')
                        if plate_label:
                            plate_count += 1

                        speed_val = self.speeds.get(int(track_id))
                        is_over = bool(speed_val is not None and speed_val > self.speed_threshold_kmh)
                        if is_over:
                            overspeeding += 1
                        speed_text = f'{speed_val:.1f} km/h' if speed_val is not None else '-- km/h'
                        label = f"{vehicle_type} ID:{int(track_id)} {plate_label or 'NO PLATE'} {speed_text}"
                        color = (0, 0, 255) if is_over else _anpr_color_for_id(int(track_id))
                        _draw_premium_ppe_box(vis, x1, y1, x2, y2, label, color)
                        if is_over:
                            cv2.putText(vis, 'Overspeeding!', (x1, min(h - 10, y2 + 25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                        hist = self.track_history[int(track_id)]
                        hist.append(center)
                        if len(hist) > 30:
                            hist.pop(0)
                        if len(hist) > 1:
                            pts = np.array(hist, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(vis, [pts], False, color, 2, cv2.LINE_AA)

                        last_dets.append(((x1, y1, x2, y2), label, color))
                        self._save_event(int(track_id), vehicle_type, plate_label, speed_val, conf_val)

                self.stats = {
                    'vehicles': vehicles,
                    'plates': plate_count,
                    'overspeeding': overspeeding,
                    'speed_threshold_kmh': self.speed_threshold_kmh,
                    'status': 'Active',
                }
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
            except Exception as e:
                self.stats['status'] = f'Frame error: {e}'
                logger.error('ANPR %s: frame error - %s', self.camera_id, e)
                time.sleep(0.05)


# ==============================================================
# BETA MODEL DETECTION (OWLv2 VLM open-vocabulary)
# ==============================================================

_owlv2_processor = None
_owlv2_model = None
_owlv2_lock = threading.Lock()
_owlv2_infer_lock = threading.Lock()


def _get_owlv2():
    """Lazy-load OWLv2 processor and model once (shared across all beta processors)."""
    global _owlv2_processor, _owlv2_model
    with _owlv2_lock:
        if _owlv2_processor is not None:
            return _owlv2_processor, _owlv2_model
    if not _OWLV2_AVAILABLE:
        return None, None
    # Evict Florence + Qwen outside OWLv2 locks to avoid deadlocks with VLM global lock.
    _evict_vlms_except('owlv2')
    dev = _vlm_beta_device()
    with _owlv2_lock:
        if _owlv2_processor is None:
            try:
                # Prefer local model path (offline, faster cold start). Accept either a snapshot dir or a file inside it.
                local_path = OWLV2_LOCAL_PATH
                if local_path:
                    if os.path.isfile(local_path):
                        local_path = os.path.dirname(local_path)
                    _owlv2_processor = Owlv2Processor.from_pretrained(local_path, local_files_only=True)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(local_path, local_files_only=True)
                    logger.info(f"OWLv2 loaded from local path: {local_path}")
                else:
                    _owlv2_processor = Owlv2Processor.from_pretrained(OWLV2_MODEL_ID)
                    _owlv2_model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL_ID)
                    logger.info(f"OWLv2 loaded: {OWLV2_MODEL_ID}")
                if torch is not None:
                    _owlv2_model = _owlv2_model.to(dev)
                    if dev == 'cuda':
                        try:
                            _owlv2_model = _owlv2_model.half()
                        except Exception:
                            pass
                try:
                    _owlv2_model.eval()
                except Exception:
                    pass
                logger.info(f"OWLv2 using device: {dev} (set VISION_VLM_DEVICE=cuda|cpu)")
            except Exception as e:
                logger.error(f"OWLv2 load failed: {e}")
        return _owlv2_processor, _owlv2_model


_florence2_processor = None
_florence2_model = None
_florence2_lock = threading.Lock()
_florence2_infer_lock = threading.Lock()


def _florence2_local_dir():
    """Resolve local Florence-2 snapshot directory (must contain config.json)."""
    candidates = []
    if FLORENCE2_LOCAL_PATH:
        p = FLORENCE2_LOCAL_PATH
        if os.path.isfile(p):
            p = os.path.dirname(p)
        candidates.append(p)
    candidates.extend([
        os.path.join(BASE_DIR, 'hf_florence2_cache', 'Florence-2-large'),
        os.path.join(BASE_DIR, '.hf_florence2_cache', 'Florence-2-large'),
        os.path.join(BASE_DIR, 'florence2_model2', 'Florence-2-large'),
        os.path.join(BASE_DIR, 'hf_florence2_cache', 'Florence-2-base'),
        os.path.join(BASE_DIR, '.hf_florence2_cache', 'Florence-2-base'),
    ])
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, 'config.json')):
            return d
    return None


def _florence2_ready():
    return bool(_FLORENCE2_AVAILABLE and _florence2_local_dir() and Image is not None)


def _get_florence2():
    """Lazy-load Florence-2 processor and model once (shared across beta processors)."""
    global _florence2_processor, _florence2_model
    if not _FLORENCE2_AVAILABLE or _FlorenceProcessorClass is None or _FlorenceModelClass is None:
        return None, None
    with _florence2_lock:
        if _florence2_processor is not None:
            return _florence2_processor, _florence2_model
    local_dir = _florence2_local_dir()
    if not local_dir:
        return None, None
    _evict_vlms_except('florence2')
    dev = _vlm_beta_device()
    with _florence2_lock:
        if _florence2_processor is None:
            try:
                _florence2_processor = _FlorenceProcessorClass.from_pretrained(
                    local_dir, local_files_only=True, trust_remote_code=True
                )
                load_kw = {'local_files_only': True, 'trust_remote_code': True}
                if torch is not None:
                    load_kw['torch_dtype'] = torch.float16 if dev == 'cuda' else torch.float32
                _florence2_model = _FlorenceModelClass.from_pretrained(local_dir, **load_kw)
                if torch is not None:
                    _florence2_model = _florence2_model.to(dev)
                try:
                    _florence2_model.eval()
                except Exception:
                    pass
                logger.info(f"Florence-2 loaded ({dev}) from local path: {local_dir}")
            except Exception as e:
                logger.error(f"Florence-2 load failed: {e}")
                _florence2_processor = None
                _florence2_model = None
        return _florence2_processor, _florence2_model


def _parse_prompt_labels(prompt_text):
    """Parse comma-separated prompt into list of label strings (strip, non-empty)."""
    if not prompt_text or not isinstance(prompt_text, str):
        return []
    return [s.strip() for s in prompt_text.split(',') if s.strip()]


class BetaDetectionProcessor:
    """Open-vocabulary detection using OWLv2. Labels come from user prompts; each camera can have multiple prompts."""
    BETA_FRAME_INTERVAL = 1.0 / 3.0  # ~3 fps inference to reduce load

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'status': 'Initializing'}

    def start(self):
        if not _OWLV2_AVAILABLE:
            self.stats['status'] = 'OWLv2 not available'
            return
        labels = self._all_labels()
        if not labels:
            self.stats['status'] = 'No labels'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_labels(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            for lbl in _parse_prompt_labels(pt):
                if lbl and lbl.lower() not in seen:
                    seen.add(lbl.lower())
                    out.append(lbl)
        return out

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_owlv2()
        if proc is None or model is None:
            self.stats['status'] = 'Model unavailable'
            self.running = False
            return
        labels = self._all_labels()
        if not labels:
            self.stats['status'] = 'No labels'
            self.running = False
            return
        text_labels = [labels]
        last_t = 0
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                inputs = proc(text=text_labels, images=pil_image, return_tensors="pt")
                if torch is not None and hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                # Avoid concurrent forwards on a single shared model (keeps VRAM stable on multi-camera setups).
                with _owlv2_infer_lock:
                    if torch is not None:
                        with torch.inference_mode():
                            if hasattr(model, "device") and str(model.device) != "cpu":
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    outputs = model(**inputs)
                            else:
                                outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                target_sizes = torch.tensor([(pil_image.height, pil_image.width)])
                if torch is not None and hasattr(model, 'device') and str(model.device) != 'cpu':
                    target_sizes = target_sizes.to(model.device)
                results = proc.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=self.conf, text_labels=text_labels
                )
                vis = frame.copy()
                last_dets = []
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.get("boxes")
                    scores = r.get("scores")
                    text_labels_out = r.get("text_labels", [])
                    if boxes is not None and scores is not None:
                        for box, score, label in zip(boxes, scores, text_labels_out):
                            if score.item() < self.conf:
                                continue
                            x1, y1, x2, y2 = box.tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            color = (0, 200, 120)
                            lbl = f"{label} {score.item():.2f}"
                            _draw_premium_ppe_box(vis, x1, y1, x2, y2, lbl, color)
                            last_dets.append(((x1, y1, x2, y2), lbl, color))
                info_h, info_w = 72, 380
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (OWLv2)  |  Detections: {len(last_dets)}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"Labels: {', '.join(labels[:6])}{'...' if len(labels) > 6 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'status': 'Active'}
                if len(last_dets) > 0:
                    _save_alert_event(
                        self.camera_id, 'beta_owlv2', vis, last_dets, severity='medium',
                        meta={'model': 'owlv2', 'detections': len(last_dets)}
                    )
            except Exception as e:
                logger.error(f"Beta {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


def _florence_parse_grounding_boxes(parsed, task_prompt):
    """Extract bboxes + labels from Florence-2 post_process_generation output."""
    if not parsed:
        return [], []
    if task_prompt in parsed and isinstance(parsed[task_prompt], dict):
        d = parsed[task_prompt]
        return list(d.get('bboxes') or []), list(d.get('labels') or [])
    for _k, v in parsed.items():
        if isinstance(v, dict) and 'bboxes' in v:
            return list(v.get('bboxes') or []), list(v.get('labels') or [])
    return [], []


class BetaFlorenceDetectionProcessor:
    """Instruction-driven anomaly monitoring via Florence-2 VQA-style prompting."""
    BETA_FRAME_INTERVAL = 1.0 / 3.0
    _FLO_TASK = "<VQA>"

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'alerts': 0, 'status': 'Initializing'}
        self._rule_alert_last_ts = {}

    def start(self):
        if not _florence2_ready():
            self.stats['status'] = 'Florence-2 not available'
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_rules(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            instr = (pt or '').strip()
            if instr and instr.lower() not in seen:
                seen.add(instr.lower())
                out.append(instr)
        return out

    def _build_monitor_prompt(self, instruction):
        return (
            f"{self._FLO_TASK} You are an intelligent video anomaly monitor. "
            f"Instruction: {instruction}\n"
            f"Check only what is visible in this frame. "
            f"If an instructed anomaly/safety violation is present, respond exactly in one line: "
            f"ALERT:<short_label>. Otherwise respond exactly: CLEAR"
        )

    @staticmethod
    def _parse_monitor_response(text):
        t = (text or '').strip()
        tl = t.lower()
        if tl.startswith('alert:'):
            return True, t.split(':', 1)[1].strip() or 'anomaly'
        if 'alert:' in tl:
            idx = tl.find('alert:')
            tail = t[idx + len('alert:'):].strip()
            tail = tail.splitlines()[0].strip() if tail else ''
            return True, tail or 'anomaly'
        if tl.startswith('alert'):
            tail = t[len('alert'):].strip(' :.-')
            return True, (tail.splitlines()[0].strip() if tail else 'anomaly')
        if '\n' in tl:
            first = tl.splitlines()[0].strip()
            if first.startswith('alert:'):
                lbl = first.split(':', 1)[1].strip()
                return True, lbl or 'anomaly'
        return False, ''

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_florence2()
        if proc is None or model is None:
            self.stats['status'] = 'Florence-2 not loaded'
            self.running = False
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            self.running = False
            return
        last_t = 0
        total_alerts = 0
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                vis = frame.copy()
                last_dets = []
                triggered_label = ''
                triggered_rule = ''
                for rule in rules:
                    prompt = self._build_monitor_prompt(rule)
                    inputs = proc(text=prompt, images=pil_image, return_tensors="pt")
                    if torch is not None and hasattr(model, 'parameters'):
                        try:
                            device = next(model.parameters()).device
                        except Exception:
                            device = None
                        if device is not None:
                            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    with _florence2_infer_lock:
                        if torch is not None:
                            with torch.inference_mode():
                                dev = next(model.parameters()).device
                                if str(dev) != "cpu":
                                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                                        generated_ids = model.generate(
                                            **inputs,
                                            max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                            num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                            do_sample=False
                                        )
                                else:
                                    generated_ids = model.generate(
                                        **inputs,
                                        max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                        num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                        do_sample=False
                                    )
                        else:
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=int(os.environ.get('VISION_FLORENCE_MONITOR_MAX_NEW_TOKENS', '48')),
                                num_beams=int(os.environ.get('VISION_FLORENCE_MONITOR_NUM_BEAMS', '3')),
                                do_sample=False
                            )
                    generated_text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    is_alert, label = self._parse_monitor_response(generated_text)
                    if is_alert:
                        triggered_label = label
                        triggered_rule = rule
                        break

                emit_alert = False
                if triggered_label:
                    h, w = vis.shape[:2]
                    x1, y1, x2, y2 = 8, 8, max(12, w - 8), max(12, h - 8)
                    _draw_premium_ppe_box(vis, x1, y1, x2, y2, f"Anomaly: {triggered_label}", (0, 0, 255))
                    last_dets.append(((x1, y1, x2, y2), f"anomaly:{triggered_label}", (0, 0, 255)))
                    # Tiny per-instruction cooldown: prevent identical consecutive-frame alert spam.
                    key = ((triggered_rule or '').strip().lower(), (triggered_label or '').strip().lower())
                    now_ts = time.time()
                    last_ts = self._rule_alert_last_ts.get(key, 0.0)
                    emit_alert = (now_ts - last_ts) >= max(0.0, VISION_FLORENCE_RULE_ALERT_COOLDOWN_SEC)
                    if emit_alert:
                        self._rule_alert_last_ts[key] = now_ts
                        total_alerts += 1
                info_h, info_w = 72, 420
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (Florence-2 Monitor)  |  Alerts: {total_alerts}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                rule_txt = triggered_rule if triggered_rule else "No instructed anomaly in frame"
                cv2.putText(vis, f"{rule_txt[:60]}{'...' if len(rule_txt) > 60 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'alerts': total_alerts, 'status': 'Active'}
                if len(last_dets) > 0 and emit_alert:
                    _save_alert_event(
                        self.camera_id, 'beta_florence2_anomaly', vis, last_dets, severity='high',
                        meta={'model': 'florence2_monitor', 'instruction': triggered_rule, 'alert_label': triggered_label}
                    )
            except Exception as e:
                logger.error(f"Beta (Florence-2) {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


class BetaQwenDetectionProcessor:
    """Instruction-driven anomaly monitoring via Qwen vision-language model."""
    BETA_FRAME_INTERVAL = 1.0 / 2.0

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.conf = conf
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._last_detections = []
        self._thread = None
        self.stats = {'detections': 0, 'alerts': 0, 'status': 'Initializing'}
        self._anomaly_state = {}  # (rule, label) -> {'active': bool, 'last_seen': ts, 'last_emit': ts}

    def start(self):
        if not _qwen25vl_ready():
            self.stats['status'] = 'Qwen not available'
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _all_rules(self):
        out = []
        seen = set()
        for pt in self.prompt_texts:
            instr = (pt or '').strip()
            if instr and instr.lower() not in seen:
                seen.add(instr.lower())
                out.append(instr)
        return out

    @staticmethod
    def _parse_monitor_response(text):
        t = (text or '').strip()
        tl = t.lower()
        if tl.startswith('alert:'):
            return True, t.split(':', 1)[1].strip() or 'anomaly'
        if 'alert:' in tl:
            idx = tl.find('alert:')
            tail = t[idx + len('alert:'):].strip()
            tail = tail.splitlines()[0].strip() if tail else ''
            return True, tail or 'anomaly'
        if tl.startswith('alert'):
            tail = t[len('alert'):].strip(' :.-')
            return True, (tail.splitlines()[0].strip() if tail else 'anomaly')
        return False, ''

    @staticmethod
    def _rule_categories(rule_text):
        s = (rule_text or '').lower()
        cats = []
        if any(k in s for k in ('ppe', 'hard hat', 'hardhat', 'helmet', 'lab coat', 'safety vest', 'no ppe')):
            cats.append('no_ppe')
        if any(k in s for k in ('dropped box', 'dropped boxes', 'box on floor', 'boxes on the floor', 'fallen box')):
            cats.append('dropped_box')
        if any(k in s for k in ('ladder',)):
            cats.append('ladder')
        if any(k in s for k in ('obstruction', 'obstructions', 'blocked aisle', 'aisle')):
            cats.append('aisle_obstruction')
        if not cats:
            cats.append('generic_anomaly')
        return cats

    @staticmethod
    def _category_prompt(rule_text, category):
        base = (
            "You are an intelligent video anomaly monitor. "
            f"Instruction context: {rule_text}\n"
            "Check only what is visible in this frame. "
        )
        if category == 'no_ppe':
            return base + (
                "Task: detect people without required PPE (white hard hat and safety blue lab coat). "
                "If present respond exactly: ALERT:No PPE usage. Otherwise respond exactly: CLEAR"
            )
        if category == 'dropped_box':
            return base + (
                "Task: detect dropped boxes on the floor (not on shelves, not held by people). "
                "If present respond exactly: ALERT:Dropped boxes on floor. Otherwise respond exactly: CLEAR"
            )
        if category == 'ladder':
            return base + (
                "Task: detect unsafe ladder usage. "
                "If present respond exactly: ALERT:Unsafe ladder usage. Otherwise respond exactly: CLEAR"
            )
        if category == 'aisle_obstruction':
            return base + (
                "Task: detect aisle obstruction/blocking by objects that hinder movement. "
                "If present respond exactly: ALERT:Aisle obstruction. Otherwise respond exactly: CLEAR"
            )
        return base + (
            "Task: detect any anomaly relevant to instruction. "
            "If present respond exactly: ALERT:Anomaly detected. Otherwise respond exactly: CLEAR"
        )

    def _qwen_gen(self, proc, model, text_in, image_obj):
        ins = None
        if image_obj is not None and hasattr(proc, 'apply_chat_template'):
            try:
                msgs = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_obj},
                        {"type": "text", "text": text_in},
                    ],
                }]
                ins = proc.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                ins = None
        if ins is None:
            ins = proc(text=text_in, images=image_obj, return_tensors='pt')
        if torch is not None and hasattr(model, 'parameters'):
            dvc = next(model.parameters()).device
            ins = {k: v.to(dvc) if hasattr(v, 'to') else v for k, v in ins.items()}
        with _qwen25vl_infer_lock:
            if torch is not None:
                with torch.inference_mode():
                    oids = model.generate(
                        **ins,
                        max_new_tokens=int(os.environ.get('VISION_QWEN_BETA_MONITOR_MAX_NEW_TOKENS', '32')),
                        num_beams=int(os.environ.get('VISION_QWEN_BETA_MONITOR_NUM_BEAMS', '1')),
                        do_sample=False,
                        repetition_penalty=1.05,
                    )
            else:
                oids = model.generate(
                    **ins,
                    max_new_tokens=int(os.environ.get('VISION_QWEN_BETA_MONITOR_MAX_NEW_TOKENS', '32')),
                    num_beams=int(os.environ.get('VISION_QWEN_BETA_MONITOR_NUM_BEAMS', '1')),
                    do_sample=False,
                    repetition_penalty=1.05,
                )
        try:
            in_len = int(ins["input_ids"].shape[-1]) if "input_ids" in ins else 0
            gen_only = oids[:, in_len:] if in_len > 0 else oids
            out = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        except Exception:
            out = proc.batch_decode(oids, skip_special_tokens=True)[0].strip()
        return _clean_qwen_text(out)

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        proc, model = _get_qwen25vl()
        if proc is None or model is None:
            self.stats['status'] = 'Qwen not loaded'
            self.running = False
            return
        rules = self._all_rules()
        if not rules:
            self.stats['status'] = 'No instructions'
            self.running = False
            return
        last_t = 0
        total_alerts = 0
        clear_grace = float(os.environ.get('VISION_QWEN_BETA_CLEAR_GRACE_SEC', '20'))
        reminder_sec = float(os.environ.get('VISION_QWEN_BETA_REMINDER_SEC', '900'))
        while self.running:
            try:
                now = time.time()
                if now - last_t < self.BETA_FRAME_INTERVAL:
                    time.sleep(0.03)
                    continue
                frame = self.camera_reader.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                last_t = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                if max(pil_image.size) > 1024:
                    pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
                vis = frame.copy()
                last_dets = []
                triggered_labels = []
                triggered_rule = ''
                for rule in rules:
                    rule_hits = []
                    for cat in self._rule_categories(rule):
                        out = self._qwen_gen(proc, model, self._category_prompt(rule, cat), pil_image)
                        is_alert, lbl = self._parse_monitor_response(out)
                        if is_alert:
                            rule_hits.append(lbl or cat.replace('_', ' '))
                    if rule_hits:
                        triggered_rule = rule
                        triggered_labels = rule_hits
                        break

                emit_labels = []
                if triggered_labels:
                    h, w = vis.shape[:2]
                    x1, y1, x2, y2 = 8, 8, max(12, w - 8), max(12, h - 8)
                    label_text = ", ".join(triggered_labels[:3])
                    _draw_premium_ppe_box(vis, x1, y1, x2, y2, f"Anomaly: {label_text}", (0, 0, 255))
                    last_dets.append(((x1, y1, x2, y2), f"anomaly:{label_text}", (0, 0, 255)))
                    now_ts = time.time()
                    seen_keys = set()
                    for lbl in triggered_labels:
                        key = ((triggered_rule or '').strip().lower(), (lbl or '').strip().lower())
                        seen_keys.add(key)
                        st = self._anomaly_state.get(key, {'active': False, 'last_seen': 0.0, 'last_emit': 0.0})
                        should_emit = (not st['active']) or ((now_ts - st['last_emit']) >= max(0.0, reminder_sec))
                        st['active'] = True
                        st['last_seen'] = now_ts
                        if should_emit:
                            st['last_emit'] = now_ts
                            emit_labels.append(lbl)
                            total_alerts += 1
                        self._anomaly_state[key] = st
                    # mark stale active anomalies as cleared after grace period
                    for k, st in list(self._anomaly_state.items()):
                        if k not in seen_keys and st.get('active') and (now_ts - st.get('last_seen', 0.0)) >= max(0.0, clear_grace):
                            st['active'] = False
                            self._anomaly_state[k] = st
                else:
                    now_ts = time.time()
                    for k, st in list(self._anomaly_state.items()):
                        if st.get('active') and (now_ts - st.get('last_seen', 0.0)) >= max(0.0, clear_grace):
                            st['active'] = False
                            self._anomaly_state[k] = st
                info_h, info_w = 72, 430
                info_bg = np.zeros((info_h, info_w, 3), dtype=np.uint8)
                info_bg[:] = (28, 28, 32)
                vis[6:6+info_h, 8:8+info_w] = cv2.addWeighted(vis[6:6+info_h, 8:8+info_w], 0.35, info_bg, 0.65, 0)
                cv2.rectangle(vis, (8, 6), (8+info_w, 6+info_h), (70, 70, 78), 1)
                cv2.putText(vis, f"Beta (Qwen Monitor)  |  Alerts: {total_alerts}",
                            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                rule_txt = triggered_rule if triggered_rule else "No instructed anomaly in frame"
                cv2.putText(vis, f"{rule_txt[:60]}{'...' if len(rule_txt) > 60 else ''}",
                            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 2, cv2.LINE_AA)
                with self.lock:
                    self.annotated_frame = vis
                    self._last_detections = last_dets
                self.stats = {'detections': len(last_dets), 'alerts': total_alerts, 'status': 'Active'}
                if len(last_dets) > 0 and emit_labels:
                    _save_alert_event(
                        self.camera_id, 'beta_qwen_anomaly', vis, last_dets, severity='high',
                        meta={
                            'model': 'qwen_monitor',
                            'instruction': triggered_rule,
                            'alert_labels': triggered_labels,
                            'new_alert_labels': emit_labels,
                            'alert_label': ", ".join(triggered_labels),
                        }
                    )
            except Exception as e:
                logger.error(f"Beta (Qwen) {self.camera_id}: {e}")
                self.stats['status'] = str(e)[:40]
                time.sleep(0.2)
        _maybe_unload_beta_vlms_if_idle()

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self._last_detections)


def _make_beta_processor(camera_id, camera_reader, prompt_texts, conf, backend):
    """Choose OWLv2 or Florence-2 beta processor without altering the OWLv2 class."""
    b = (backend or 'owlv2').strip().lower()
    if b == 'florence2':
        return BetaFlorenceDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)
    if b == 'qwen':
        return BetaQwenDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)
    return BetaDetectionProcessor(camera_id, camera_reader, prompt_texts, conf=conf)


# ==============================================================
# GLOBAL STATE
# ==============================================================

camera_readers = {}         # camera_id -> CameraReader
headcount_procs = {}        # camera_id -> HeadCountProcessor
entryexit_procs = {}        # camera_id -> EntryExitProcessor
flapgate_procs = {}         # camera_id -> FlapGateProcessor
ppe_procs = {}              # camera_id -> PPEDetectionProcessor
fire_smoke_procs = {}       # camera_id -> FireSmokeDetectionProcessor
anpr_procs = {}             # camera_id -> ANPRDetectionProcessor
beta_procs = {}             # camera_id -> BetaDetectionProcessor


# ==============================================================
# FACE RECOGNITION (FR) MODULE
# Integrates logic from trialinsightfaceattend_1.py and
# ultimatefacedataset_streamlit_2.py exactly as-is, adapted for Flask.
# ==============================================================

try:
    import pickle
    import hashlib
    import insightface
    from insightface.model_zoo import get_model as _insightface_get_model
    import onnxruntime as _ort
    _FR_INSIGHTFACE_AVAILABLE = True
except ImportError:
    _insightface_get_model = None
    _FR_INSIGHTFACE_AVAILABLE = False
    logger.warning('FR: insightface not installed. pip install insightface onnxruntime')

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _FR_DeepSort
    _FR_DEEPSORT_AVAILABLE = True
except ImportError:
    _FR_DeepSort = None
    _FR_DEEPSORT_AVAILABLE = False

_FR_YOLO_AVAILABLE = True  # YOLO already imported at module level

# ── FR paths ──
# DATA_DIR is already resolved from VISION_DATA_DIR env var (set to /app/data in Docker,
# or BASE_DIR/data locally). All FR paths derive from it so they always point inside the
# bind-mounted volume — exactly where the employee images live on disk.
FR_AUTOFACEDATA_BASE_DIR = os.path.join(DATA_DIR, 'autofacedata')
FR_AUTOFACEDATA_CSV     = os.path.join(DATA_DIR, 'autofacedata.csv')
# AntelopeV2 model files: override with VISION_FR_ANTELOPEV2_PATH env var if needed,
# otherwise fall back to data/antelopev2/ (inside the same bind-mounted volume).
FR_ANTELOPEV2_PATH = os.environ.get('VISION_FR_ANTELOPEV2_PATH', '').strip() or os.path.join(DATA_DIR, 'antelopev2')
FR_FACE_MODEL_PATH = os.path.join(BASE_DIR, 'bestempfacereal.pt')
FR_DETECTION_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'scrfd_10g_bnkps.onnx')
FR_RECOGNITION_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'glintr100.onnx')
FR_GENDER_AGE_MODEL = os.path.join(FR_ANTELOPEV2_PATH, 'genderage.onnx')

logger.info(
    f'FR paths resolved — DATA_DIR={DATA_DIR!r}  '
    f'base_dir={FR_AUTOFACEDATA_BASE_DIR!r}  '
    f'csv={FR_AUTOFACEDATA_CSV!r}  '
    f'antelopev2={FR_ANTELOPEV2_PATH!r}'
)

# ── FR recognition constants (from trialinsightfaceattend_1.py) ──
FR_RECOGNITION_THRESHOLD = 0.30
FR_CACHE_FILENAME = 'insightface_embeddings_antelopev2.pkl'
FR_MIN_SIDE_FOR_EMBEDDING = 300
FR_DETECTION_RESIZE_WIDTH = 640
FR_INSIGHTFACE_DET_SIZE = (640, 640)
FR_INSIGHTFACE_DET_THRESH = 0.25
FR_DISPLAY_CONFIDENCE_THRESHOLD = 0.55
FR_FACE_PALETTE = [
    (72, 187, 99), (255, 165, 0), (203, 192, 255),
    (147, 255, 255), (255, 144, 238), (128, 255, 203), (185, 218, 255),
]
FR_UNKNOWN_COLOR = (80, 80, 255)
FR_BOX_THICKNESS = 3
FR_CORNER_RADIUS = 10
FR_LABEL_PADDING = 10
FR_FONT_SCALE_LABEL = 0.8
FR_FONT_SCALE_SUB = 0.65
FR_FONT_THICKNESS = 2

# ── FR data collection constants (from ultimatefacedataset_streamlit_2.py) ──
FR_NUM_IMAGES_PER_FACE = 50
FR_FACE_CLASS_ID = 0
FR_CONF_THRESHOLD = 0.05
FR_NMS_IOU = 0.40
FR_FACE_BOX_EXPAND_RATIO = 1.2
FR_DETECT_EVERY_N_FRAMES = 2
FR_SHARPNESS_THRESHOLD = 100.0


# ── Drawing utilities from trialinsightfaceattend_1.py ──

def _fr_resize_for_detection(frame, max_width=None):
    if max_width is None:
        max_width = FR_DETECTION_RESIZE_WIDTH
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / w
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, w / new_w, h / new_h


def _fr_scale_bbox(bbox, sx, sy):
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _fr_color_for_identity(name):
    if name is None:
        return FR_UNKNOWN_COLOR
    idx = hash(name) % len(FR_FACE_PALETTE)
    return FR_FACE_PALETTE[idx]


def _fr_draw_rounded_rect(img, x1, y1, x2, y2, color, thickness, radius=None):
    if radius is None:
        radius = FR_CORNER_RADIUS
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


def _fr_draw_bbox_with_label(img, x1, y1, x2, y2, label, sublabel, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, FR_FONT_THICKNESS)
    tw2, th2 = 0, 0
    if sublabel:
        (tw2, th2), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, FR_FONT_THICKNESS)
    label_w = max(tw, tw2) + FR_LABEL_PADDING * 2
    label_h = th + (th2 + FR_LABEL_PADDING if sublabel else 0) + FR_LABEL_PADDING
    ly1 = max(0, y1 - label_h - 4)
    ly2 = y1 - 4
    lx1 = max(0, min(x1, img.shape[1] - label_w))
    lx2 = min(img.shape[1], lx1 + label_w)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, FR_BOX_THICKNESS)
    cv2.putText(img, label, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, (0, 0, 0), FR_FONT_THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(img, label, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING),
                cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_LABEL, (255, 255, 255), FR_FONT_THICKNESS, cv2.LINE_AA)
    if sublabel:
        cv2.putText(img, sublabel, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, (0, 0, 0), FR_FONT_THICKNESS + 2, cv2.LINE_AA)
        cv2.putText(img, sublabel, (lx1 + FR_LABEL_PADDING, ly1 + th + FR_LABEL_PADDING + th2 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FR_FONT_SCALE_SUB, (255, 255, 255), FR_FONT_THICKNESS, cv2.LINE_AA)
    _fr_draw_rounded_rect(img, x1, y1, x2, y2, color, FR_BOX_THICKNESS)


def _fr_cache_path(csv_path):
    key = f'v8_antelopev2_final|{csv_path}|{FR_AUTOFACEDATA_BASE_DIR}|antelopev2'
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.{h}')


# ── CSV helpers from ultimatefacedataset_streamlit_2.py ──

def _fr_ensure_dirs(person_name=None):
    os.makedirs(FR_AUTOFACEDATA_BASE_DIR, exist_ok=True)
    if person_name:
        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir
    return FR_AUTOFACEDATA_BASE_DIR


def _fr_get_or_create_csv():
    fieldnames = ['person_name', 'file_path']
    if os.path.exists(FR_AUTOFACEDATA_CSV):
        rows = []
        try:
            with open(FR_AUTOFACEDATA_CSV, 'r', encoding='utf-8', newline='') as f:
                reader_csv = csv.DictReader(f)
                fn = reader_csv.fieldnames or fieldnames
                for row in reader_csv:
                    rows.append(dict(row))
            return rows, fn
        except Exception as e:
            # Log but do NOT recreate/wipe the file — return empty rows so callers
            # can still append new data without losing whatever is on disk.
            logger.warning(f'FR: Could not read CSV at {FR_AUTOFACEDATA_CSV}: {e}')
            return [], fieldnames
    # CSV doesn't exist yet — create it with the header only.
    _fr_ensure_dirs()
    try:
        with open(FR_AUTOFACEDATA_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f'FR: Created new CSV at {FR_AUTOFACEDATA_CSV}')
    except Exception as e:
        logger.error(f'FR: Could not create CSV at {FR_AUTOFACEDATA_CSV}: {e}')
    return [], fieldnames


def _fr_save_csv(rows, fieldnames):
    """Write rows to the CSV atomically (write to .tmp then rename) to avoid corruption."""
    tmp_path = FR_AUTOFACEDATA_CSV + '.tmp'
    try:
        _fr_ensure_dirs()
        with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, FR_AUTOFACEDATA_CSV)
        logger.info(f'FR: CSV saved — {len(rows)} row(s) → {FR_AUTOFACEDATA_CSV}')
        return True
    except Exception as e:
        logger.error(f'FR: CSV save failed ({FR_AUTOFACEDATA_CSV}): {e}')
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False


def _fr_update_csv_for_person(person_name, paths_list):
    rows, fieldnames = _fr_get_or_create_csv()
    path_str = ','.join(paths_list)
    found = False
    for row in rows:
        if str(row.get('person_name', '')).strip() == str(person_name).strip():
            row['file_path'] = path_str
            found = True
            break
    if not found:
        rows.append({'person_name': person_name, 'file_path': path_str})
    return _fr_save_csv(rows, fieldnames)


# ── Data collection helpers from ultimatefacedataset_streamlit_2.py ──

def _fr_expand_bbox(bbox, frame_shape, ratio=None):
    if ratio is None:
        ratio = FR_FACE_BOX_EXPAND_RATIO
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_shape[:2]
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return (x1, y1, x2, y2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = cw * ratio, ch * ratio
    nx1, ny1 = int(cx - nw / 2), int(cy - nh / 2)
    nx2, ny2 = int(cx + nw / 2), int(cy + nh / 2)
    return (max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2))


def _fr_extract_face_crop(frame, bbox, expanded=True):
    if expanded:
        bbox = _fr_expand_bbox(bbox, frame.shape)
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _fr_is_frame_usable(frame):
    if frame is None or frame.size == 0:
        return False
    try:
        mean = np.mean(frame)
        std = np.std(frame)
        if std < 5.0 or mean < 3 or mean > 252:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var >= 15
    except Exception:
        return False


def _fr_is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < FR_SHARPNESS_THRESHOLD


def _fr_is_crop_usable(crop):
    if crop is None or crop.size == 0:
        return False
    try:
        h, w = crop.shape[:2]
        if w < 20 or h < 20:
            return False
        if _fr_is_blurry(crop):
            return False
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 8 or mean < 8 or mean > 247:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= 12
    except Exception:
        return False


class _FR_SimpleTrack:
    __slots__ = ('track_id', '_bbox')

    def __init__(self, tid, bbox):
        self.track_id = tid
        self._bbox = bbox

    def to_ltrb(self):
        return self._bbox

    def is_confirmed(self):
        return True


def _fr_reader_thread(rtsp_url, state):
    """Dedicated RTSP reader – exact logic from ultimatefacedataset_streamlit_2.py."""
    prev = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            state['error'] = 'Could not open RTSP stream. Check URL and network.'
            return
        while not state.get('stop', False):
            ret, frame = cap.read()
            if ret and frame is not None:
                state['latest_frame'] = frame
                state['latest_ok'] = True
            else:
                state['latest_ok'] = False
            time.sleep(0)
        cap.release()
    except Exception as e:
        state['error'] = str(e)
    finally:
        if prev is not None:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = prev
        else:
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)


def _fr_processor_thread(state, person_name):
    """Face detection + image saving thread – exact logic from ultimatefacedataset_streamlit_2.py."""
    from collections import defaultdict as _fr_dd
    _fr_ensure_dirs(person_name)
    if not _FR_YOLO_AVAILABLE:
        state['error'] = 'ultralytics not installed. pip install ultralytics'
        return
    if not _FR_DEEPSORT_AVAILABLE:
        state['error'] = 'deep_sort_realtime not installed. pip install deep-sort-realtime'
        return
    if not os.path.exists(FR_FACE_MODEL_PATH):
        state['error'] = f'Face model not found: {FR_FACE_MODEL_PATH}'
        return

    model = YOLO(FR_FACE_MODEL_PATH)
    tracker = _FR_DeepSort(
        max_age=30, n_init=3, nms_max_overlap=0.7,
        max_cosine_distance=0.3, nn_budget=100,
        embedder='mobilenet', half=True, bgr=True, embedder_gpu=True,
    )
    next_persistent_id = 1
    track_to_persistent = {}
    persistent_counts = _fr_dd(int)
    persistent_paths = _fr_dd(list)
    last_bbox_by_track = {}
    frame_index = 0
    last_good_display = None
    last_save_time = time.time()
    save_interval = 1.0 / 3.0

    try:
        while not state.get('stop', False):
            frame = state.get('latest_frame')
            if frame is None:
                time.sleep(0.01)
                continue
            frame = frame.copy()
            frame_index += 1

            if not _fr_is_frame_usable(frame) or _fr_is_blurry(frame):
                if last_good_display is not None:
                    state['frame'] = last_good_display
                time.sleep(0.001)
                continue

            display_frame = frame.copy()
            run_detection = (frame_index % FR_DETECT_EVERY_N_FRAMES == 1)

            if run_detection:
                results = model(frame, conf=FR_CONF_THRESHOLD, iou=FR_NMS_IOU,
                                classes=[FR_FACE_CLASS_ID], verbose=False)
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) != FR_FACE_CLASS_ID:
                            continue
                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                        bconf = float(box.conf[0])
                        detections.append(([int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)], bconf, 'face'))
                tracked = tracker.update_tracks(detections, frame=frame)
            else:
                tracked = []
                for tid, bbox in list(last_bbox_by_track.items()):
                    if tid in track_to_persistent:
                        tracked.append(_FR_SimpleTrack(tid, bbox))

            for track in tracked:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                raw_bbox = (x1, y1, x2, y2)
                bbox = _fr_expand_bbox(raw_bbox, frame.shape)
                x1, y1, x2, y2 = bbox
                last_bbox_by_track[track_id] = bbox

                if track_id not in track_to_persistent:
                    track_to_persistent[track_id] = next_persistent_id
                    next_persistent_id += 1
                persistent_id = track_to_persistent[track_id]

                color = (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f'ID {persistent_id}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                current_time = time.time()
                if (persistent_counts[persistent_id] < FR_NUM_IMAGES_PER_FACE and
                        current_time - last_save_time >= save_interval):
                    crop = _fr_extract_face_crop(frame, bbox, expanded=True)
                    if crop is not None and _fr_is_crop_usable(crop):
                        crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LANCZOS4)
                        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name)
                        fname = f'face_{person_name}_{persistent_counts[persistent_id]:03d}.jpg'
                        out_path = os.path.join(person_dir, fname)
                        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        abs_path = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name, fname)
                        persistent_paths[persistent_id].append(abs_path)
                        persistent_counts[persistent_id] += 1
                        last_save_time = current_time
                        if persistent_counts[persistent_id] == FR_NUM_IMAGES_PER_FACE:
                            _fr_update_csv_for_person(person_name, persistent_paths[persistent_id])
                            state['completed_ids'] = state.get('completed_ids', set()) | {persistent_id}
                            state['csv_updated'] = True

                cv2.putText(display_frame, f'{persistent_counts[persistent_id]}/{FR_NUM_IMAGES_PER_FACE}',
                            (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if run_detection:
                current_ids = {t.track_id for t in tracked if t.is_confirmed()}
                for tid in list(last_bbox_by_track.keys()):
                    if tid not in current_ids:
                        last_bbox_by_track.pop(tid, None)

            out_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            last_good_display = out_rgb
            state['frame'] = out_rgb
            state['counts'] = dict(persistent_counts)
            state['completed'] = list(state.get('completed_ids', set()))
    except Exception as e:
        state['error'] = str(e)
        logger.error(f'FR data collection processor error: {e}')


class FaceRecognitionProcessor:
    """
    Live face recognition processor for a camera stream.
    Core logic from trialinsightfaceattend_1.py (AttendanceSystem), adapted for Flask.
    """

    def __init__(self, camera_id, camera_reader, threshold=None):
        self.camera_id = camera_id
        self.camera_reader = camera_reader
        self.threshold = threshold if threshold is not None else FR_RECOGNITION_THRESHOLD
        self.running = False
        self.annotated_frame = None
        self.lock = threading.Lock()
        self._thread = None
        self.face_app = None
        self.known_embeddings = []
        self.known_names = []
        self.known_embeddings_matrix = None
        self._ready = False
        self._init_error = None
        self.stats = {'status': 'Initializing', 'faces_detected': 0, 'recognized': 0}
        self.last_detections = []
        self._diag_tick = 0

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_annotated_frame(self):
        with self.lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_last_detections(self):
        with self.lock:
            return list(self.last_detections)

    def _run(self):
        try:
            self._init_insightface()
            self._load_embeddings()
            self._ready = True
            self.stats['status'] = 'Active'
        except Exception as e:
            self._init_error = str(e)
            self.stats['status'] = f'Init error: {e}'
            logger.error(f'FR {self.camera_id}: Init error - {e}')
            return

        frame_index = 0
        while self.running:
            frame = self.camera_reader.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            frame_index += 1
            if frame_index % 3 != 0:
                time.sleep(0.01)
                continue
            try:
                annotated, results = self._process_frame(frame)
                recognized = sum(1 for r in results if r.get('name') is not None)
                self.stats = {
                    'status': 'Active',
                    'faces_detected': len(results),
                    'recognized': recognized,
                }
                with self.lock:
                    self.annotated_frame = annotated
                    # Keep a generic overlay list so /api/feed_snapshot composite mode can draw FR too.
                    dets = []
                    for r in results:
                        bbox = r.get('bbox')
                        if not bbox or len(bbox) != 4:
                            continue
                        name = r.get('name')
                        conf = float(r.get('confidence', 0.0))
                        color = _fr_color_for_identity(name)
                        label = (name if name else 'Unknown') + f' {conf:.0%}'
                        dets.append((bbox, label, color))
                    self.last_detections = dets
                self._diag_tick += 1
                if self._diag_tick % 90 == 0:
                    logger.info(
                        f'FR {self.camera_id}: frame_diag faces_detected={len(results)} '
                        f'recognized={recognized} known_embs={len(self.known_embeddings)} '
                        f'match_threshold={self.threshold} draw_threshold={FR_DISPLAY_CONFIDENCE_THRESHOLD}'
                    )
            except Exception as e:
                logger.error(f'FR {self.camera_id}: Frame processing error - {e}')
                time.sleep(0.05)

    def _init_insightface(self):
        """Initialize AntelopeV2 directly. From trialinsightfaceattend_1.py _init_antelopev2_direct()."""
        if not _FR_INSIGHTFACE_AVAILABLE:
            raise RuntimeError('insightface not installed. pip install insightface onnxruntime')
        if not os.path.exists(FR_DETECTION_MODEL):
            raise RuntimeError(f'Detection model not found: {FR_DETECTION_MODEL}')
        if not os.path.exists(FR_RECOGNITION_MODEL):
            raise RuntimeError(f'Recognition model not found: {FR_RECOGNITION_MODEL}')

        os.environ['INSIGHTFACE_MODELS_DIR'] = FR_ANTELOPEV2_PATH
        det_model = _insightface_get_model(FR_DETECTION_MODEL)
        det_model.prepare(ctx_id=0, input_size=FR_INSIGHTFACE_DET_SIZE)
        if hasattr(det_model, 'threshold'):
            det_model.threshold = FR_INSIGHTFACE_DET_THRESH

        rec_model = _insightface_get_model(FR_RECOGNITION_MODEL)
        rec_model.prepare(ctx_id=0)

        genderage_model = None
        if os.path.exists(FR_GENDER_AGE_MODEL):
            genderage_model = _insightface_get_model(FR_GENDER_AGE_MODEL)
            genderage_model.prepare(ctx_id=0)

        det_thresh = FR_INSIGHTFACE_DET_THRESH

        class _SimpleInsightFace:
            def __init__(self_, det, rec, ga, thresh):
                self_.det_model = det
                self_.rec_model = rec
                self_.genderage_model = ga
                self_.det_thresh = thresh

            def get(self_, img):
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                try:
                    bboxes, kpss = self_.det_model.detect(img)
                except Exception:
                    bboxes, kpss = self_.det_model.detect(img, threshold=self_.det_thresh)
                if bboxes is None or bboxes.shape[0] == 0:
                    return []
                if bboxes.shape[1] >= 5:
                    mask = bboxes[:, 4] >= self_.det_thresh
                    bboxes = bboxes[mask]
                    if kpss is not None:
                        kpss = kpss[mask]
                if bboxes.shape[0] == 0:
                    return []
                from insightface.app.common import Face
                faces = []
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i][:4]
                    kps = kpss[i] if kpss is not None else None
                    face = Face(bbox=bbox, kps=kps)
                    face.det_score = float(bboxes[i][4]) if bboxes.shape[1] >= 5 else 1.0
                    self_.rec_model.get(img, face)
                    faces.append(face)
                return faces

        self.face_app = _SimpleInsightFace(det_model, rec_model, genderage_model, det_thresh)
        logger.info(f'FR {self.camera_id}: AntelopeV2 initialized successfully')

    def _load_embeddings(self):
        """Load or build embeddings from CSV. From trialinsightfaceattend_1.py _ensure_embeddings()."""
        if not os.path.exists(FR_AUTOFACEDATA_CSV):
            logger.warning(f'FR: CSV not found at {FR_AUTOFACEDATA_CSV}. No faces recognized until data is collected.')
            return
        import pandas as pd
        self._df = None
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                self._df = pd.read_csv(FR_AUTOFACEDATA_CSV, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if self._df is None:
            self._df = pd.read_csv(FR_AUTOFACEDATA_CSV, encoding='utf-8', encoding_errors='replace')

        logger.info(f'FR _load_embeddings: CSV has {len(self._df)} row(s), '
                    f'columns={list(self._df.columns)}')

        if 'person_name' not in self._df.columns or 'file_path' not in self._df.columns:
            logger.error(f'FR: CSV missing expected columns. '
                         f'Got {list(self._df.columns)}, need person_name + file_path')
            return

        cache_file = _fr_cache_path(FR_AUTOFACEDATA_CSV)
        current_employees = set(str(r['person_name']) for _, r in self._df.iterrows())
        cached_embs, cached_names = [], []
        cache_exists = False

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if data.get('model') == 'antelopev2' and data.get('names'):
                    cached_embs = data['embeddings']
                    cached_names = data['names']
                    cache_exists = True
            except Exception:
                pass

        cached_set = set(cached_names)
        new_employees = current_employees - cached_set

        if not new_employees and cache_exists:
            self.known_embeddings = cached_embs
            self.known_names = cached_names
            if self.known_embeddings:
                self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)
            logger.info(f'FR: Loaded {len(self.known_embeddings)} cached embeddings for {len(set(self.known_names))} people')
            return

        new_embs, new_names = [], []
        logger.info(f'FR: Building embeddings for {len(new_employees)} new employee(s): {sorted(new_employees)}')
        for _, row in self._df.iterrows():
            name = str(row['person_name'])
            if name not in new_employees:
                continue
            file_path = str(row.get('file_path', ''))
            paths = [p.strip() for p in file_path.split(',') if p.strip()]
            person_ok = 0
            person_missing = 0
            person_no_face = 0
            for img_path in paths:
                # Absolute paths (written by current code) are used directly.
                # Relative paths (legacy entries) are resolved against the base dir.
                if os.path.isabs(img_path):
                    full = img_path
                else:
                    full = os.path.join(FR_AUTOFACEDATA_BASE_DIR, name, os.path.basename(img_path))
                    if not os.path.exists(full):
                        full = os.path.join(FR_AUTOFACEDATA_BASE_DIR, img_path)
                    if not os.path.exists(full):
                        full = img_path
                if not os.path.exists(full):
                    person_missing += 1
                    continue
                img = cv2.imread(full)
                if img is None:
                    person_missing += 1
                    continue
                h, w = img.shape[:2]
                if max(h, w) < FR_MIN_SIDE_FOR_EMBEDDING:
                    scale = FR_MIN_SIDE_FOR_EMBEDDING / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
                emb = self._get_embedding(img)
                if emb is not None:
                    new_embs.append(emb)
                    new_names.append(name)
                    person_ok += 1
                else:
                    person_no_face += 1
            logger.info(f'FR embeddings: {name!r} — paths={len(paths)}  '
                        f'ok={person_ok}  missing={person_missing}  no_face={person_no_face}')

        if cache_exists:
            self.known_embeddings = cached_embs + new_embs
            self.known_names = cached_names + new_names
        else:
            self.known_embeddings = new_embs
            self.known_names = new_names

        if self.known_embeddings:
            self.known_embeddings_matrix = np.array(self.known_embeddings, dtype=np.float32)

        try:
            os.makedirs(FR_AUTOFACEDATA_BASE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'model': 'antelopev2',
                    'embeddings': self.known_embeddings,
                    'names': self.known_names,
                    'rollnos': self.known_names,
                }, f)
        except Exception as e:
            logger.warning(f'FR: Cache save error: {e}')

        logger.info(f'FR: Built {len(self.known_embeddings)} embeddings for {len(set(self.known_names))} people')

    def reload_embeddings(self):
        """Rebuild embeddings after adding or removing people."""
        import glob as _fr_glob
        self.known_embeddings = []
        self.known_names = []
        self.known_embeddings_matrix = None
        try:
            for f in _fr_glob.glob(os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.*')):
                os.remove(f)
        except Exception:
            pass
        try:
            self._load_embeddings()
        except Exception as e:
            logger.error(f'FR {self.camera_id}: Reload embeddings error: {e}')

    def _get_embedding(self, img):
        if self.face_app is None:
            return None
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb)
            if not faces:
                return None
            best = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            emb = best.normed_embedding
            if emb is None or len(emb) == 0:
                return None
            emb = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            return None

    def _match_embedding_fast(self, query_emb):
        """Fast cosine similarity match. From trialinsightfaceattend_1.py."""
        if query_emb is None or self.known_embeddings_matrix is None or len(self.known_embeddings) == 0:
            return None, 0.0
        sims = self.known_embeddings_matrix @ query_emb
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score > self.threshold:
            return self.known_names[best_idx], best_score
        return None, best_score

    def _process_frame(self, frame):
        """Process a single frame for recognition. From trialinsightfaceattend_1.py process_frame()."""
        if frame is None or self.face_app is None:
            return frame, []
        display = frame.copy()
        small, sx, sy = _fr_resize_for_detection(frame, max_width=FR_DETECTION_RESIZE_WIDTH)
        try:
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(rgb_small)
            results = []
            for face in faces:
                bbox_small = face.bbox.astype(int).tolist()
                bbox_full = _fr_scale_bbox(bbox_small, sx, sy)
                embedding = face.normed_embedding
                if embedding is not None:
                    emb = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    name, confidence = self._match_embedding_fast(emb)
                else:
                    name, confidence = None, 0.0
                results.append({'bbox': bbox_full, 'name': name, 'confidence': float(confidence)})
                if confidence >= FR_DISPLAY_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = bbox_full
                    color = _fr_color_for_identity(name)
                    label = name if name else 'Unknown'
                    sublabel = f'{confidence:.0%}'
                    _fr_draw_bbox_with_label(display, x1, y1, x2, y2, label, sublabel, color)
            return display, results
        except Exception:
            return display, []


fr_procs = {}           # camera_id -> FaceRecognitionProcessor
fr_collect_states = {}  # session_id -> collection state dict


# ==============================================================
# FLASK APP
# ==============================================================

app = Flask(__name__)
_sk = (os.environ.get('VISION_SECRET_KEY') or '').strip()
app.config['SECRET_KEY'] = _sk if _sk else 'vision-ai-secret-change-me'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)


# ---------- Auth (session + roles: admin / user) ----------

_ONLINE_SESSION_MAX_MIN = int(os.environ.get('VISION_ONLINE_SESSION_MINUTES', '45') or '45')


def _auth_touch():
    tok = session.get('auth_token')
    if not tok:
        return
    try:
        conn = get_db()
        conn.execute(
            'UPDATE user_active_sessions SET last_activity=CURRENT_TIMESTAMP WHERE token=?',
            (tok,),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _auth_cleanup_sessions():
    try:
        conn = get_db()
        conn.execute(
            "DELETE FROM user_active_sessions WHERE datetime(last_activity) < datetime('now', ?)",
            (f'-{_ONLINE_SESSION_MAX_MIN} minutes',),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@app.before_request
def _require_authentication():
    if request.method == 'OPTIONS':
        return None
    path = request.path or ''
    if path.startswith('/static'):
        return None
    if path == '/api/health':
        return None
    # Allow auth endpoints by path so session checks never block them (endpoint can be unset in edge cases).
    if path in ('/login', '/api/login', '/api/logout'):
        return None
    ep = request.endpoint
    if ep in ('page_login', 'api_login', 'api_logout'):
        return None
    if not session.get('user_id'):
        if path.startswith('/api/'):
            return jsonify({'error': 'Unauthorized', 'login': url_for('page_login')}), 401
        nxt = path if path != '/login' else '/'
        return redirect(url_for('page_login', next=nxt))
    g.auth_user_id = session.get('user_id')
    g.auth_username = session.get('username')
    g.auth_level = session.get('level')
    _auth_touch()
    return None


@app.context_processor
def _inject_auth_template():
    return {
        'auth_username': session.get('username') or '',
        'auth_level': session.get('level') or '',
    }


def _require_admin_api():
    if session.get('level') != 'admin':
        return jsonify({'error': 'Administrator access required'}), 403
    return None


def _system_kv_get(key, default):
    try:
        conn = get_db()
        row = conn.execute('SELECT value_json FROM system_config_kv WHERE key=?', (key,)).fetchone()
        conn.close()
        if not row:
            return default
        return json.loads(row['value_json'])
    except Exception:
        return default


def _system_kv_set(key, obj):
    conn = get_db()
    conn.execute(
        'INSERT OR REPLACE INTO system_config_kv (key, value_json, updated_time) VALUES (?,?,CURRENT_TIMESTAMP)',
        (key, json.dumps(obj, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


# ---------- Helper: MJPEG generators ----------

def _gen_raw(camera_id):
    while True:
        reader = camera_readers.get(camera_id)
        if reader:
            frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_headcount(camera_id):
    while True:
        proc = headcount_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_entryexit(camera_id):
    while True:
        proc = entryexit_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_flapgate(camera_id):
    while True:
        proc = flapgate_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_ppe(camera_id):
    while True:
        proc = ppe_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is None:
                # Show live stream while waiting for first PPE result (avoids blank feed on live camera)
                reader = camera_readers.get(camera_id)
                if reader:
                    frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_fire_smoke(camera_id):
    while True:
        proc = fire_smoke_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is None:
                reader = camera_readers.get(camera_id)
                if reader:
                    frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_anpr(camera_id):
    while True:
        proc = anpr_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is None:
                reader = camera_readers.get(camera_id)
                if reader:
                    frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_beta(camera_id):
    while True:
        proc = beta_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is None:
                reader = camera_readers.get(camera_id)
                if reader:
                    frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_fr(camera_id):
    while True:
        proc = fr_procs.get(camera_id)
        if proc:
            frame = proc.get_annotated_frame()
            if frame is None:
                reader = camera_readers.get(camera_id)
                if reader:
                    frame = reader.get_frame()
            if frame is not None:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_fr_collect(session_id):
    while True:
        state = fr_collect_states.get(session_id)
        if state:
            frame_rgb = state.get('frame')
            if frame_rgb is not None:
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


def _gen_workforce(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return
    if machine_id not in workforce_video_readers:
        path = _resolve_workforce_video_path(machine_id)
        if path:
            ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
    while True:
        frame = _workforce_get_frame(machine_id)
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)


def _gen_ups(panel_id):
    while True:
        frame = None
        proc = ups_procs.get(panel_id)
        if proc:
            frame = proc.get_annotated_frame()
        if frame is None:
            reader = ups_video_readers.get(panel_id)
            if reader:
                frame = reader.get_frame()
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        else:
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Loading UPS panel stream...', (120, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.04)


# ---------- Pages ----------

@app.route('/login', methods=['GET'])
def page_login():
    if session.get('user_id'):
        return redirect(request.args.get('next') or '/')
    return render_template('login.html', next=request.args.get('next') or '/')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    conn = get_db()
    row = conn.execute(
        'SELECT id, username, password_hash, level FROM app_users WHERE username=? COLLATE NOCASE',
        (username,),
    ).fetchone()
    conn.close()
    if not row or not check_password_hash(row['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = secrets.token_hex(24)
    ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
    conn = get_db()
    conn.execute(
        'INSERT INTO user_active_sessions (token, user_id, username, level, ip) VALUES (?,?,?,?,?)',
        (token, row['id'], row['username'], row['level'], ip),
    )
    conn.commit()
    conn.close()
    session['user_id'] = row['id']
    session['username'] = row['username']
    session['level'] = row['level']
    session['auth_token'] = token
    session.permanent = True
    return jsonify({'ok': True, 'username': row['username'], 'level': row['level']})


@app.route('/api/logout', methods=['GET', 'POST'])
def api_logout():
    tok = session.get('auth_token')
    if tok:
        try:
            conn = get_db()
            conn.execute('DELETE FROM user_active_sessions WHERE token=?', (tok,))
            conn.commit()
            conn.close()
        except Exception:
            pass
    session.clear()
    return jsonify({'ok': True})


@app.route('/api/auth/me', methods=['GET'])
def api_auth_me():
    if not session.get('user_id'):
        return jsonify({'authenticated': False}), 401
    return jsonify(
        {
            'authenticated': True,
            'username': session.get('username'),
            'level': session.get('level'),
        }
    )


@app.route('/')
def page_main():
    return render_template('main.html')


@app.route('/ai-config')
def page_ai_config():
    return render_template('ai_config.html')


@app.route('/zone-config/<camera_id>')
def page_zone_config(camera_id):
    return render_template('zone_config.html', camera_id=camera_id)


@app.route('/flapgate-zone-config/<camera_id>')
def page_flapgate_zone_config(camera_id):
    return render_template('flapgate_zone_config.html', camera_id=camera_id)


@app.route('/dashboard')
def page_dashboard():
    return render_template('unified_dashboard.html')


@app.route('/static/logo.jpeg')
def legacy_logo_jpeg():
    """Serve Sutek logo for old Pixecore logo.jpeg links."""
    return redirect('/static/logo.png', code=302)


@app.route('/alerts')
def page_alerts():
    return render_template('alerts.html')


@app.route('/assigned-engineers')
def page_assigned_engineers():
    return render_template('assigned_engineers.html')


@app.route('/plant-map')
def page_plant_map():
    return render_template('plant_map.html')


@app.route('/alerts/<int:alert_id>')
def page_alert_incident(alert_id):
    return render_template('incident_investigation.html', alert_id=alert_id)


@app.route('/settings')
def page_settings():
    return redirect(url_for('page_system'), code=302)


@app.route('/system')
def page_system():
    return render_template('system.html')


@app.route('/modules')
def page_modules():
    return render_template('modules.html')


@app.route('/modules/workforce-monitoring')
def page_workforce_module():
    return render_template('modules.html', active_module='workforce')


@app.route('/modules/workforce-monitoring/config')
def page_workforce_config():
    return redirect(url_for('page_workforce_apply'), code=302)


@app.route('/modules/workforce-monitoring/apply')
def page_workforce_apply():
    bind_workforce_videos_from_folder(get_db, WORKFORCE_VIDEOS_DIR)
    ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
    return render_template(
        'workforce_apply.html',
        module='workforce',
        workforce_videos=_workforce_videos_payload(),
    )


def _workforce_videos_payload():
    """machine_id -> {url, name, version} for template and API."""
    vmap = get_workforce_video_map(get_db, WORKFORCE_VIDEOS_DIR)
    videos = {}
    for mid in WORKFORCE_MACHINE_IDS:
        if mid in vmap:
            ver = workforce_video_version(vmap[mid])
            videos[mid] = {
                'url': url_for('feed_workforce', machine_id=mid) + f'?v={ver}',
                'name': os.path.basename(vmap[mid]),
                'version': ver,
                'stream': 'mjpeg',
            }
    return videos


def _resolve_workforce_video_path(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return None
    vmap = get_workforce_video_map(get_db, WORKFORCE_VIDEOS_DIR)
    return vmap.get(machine_id)


@app.route('/api/workforce/videos', methods=['GET'])
def api_workforce_videos():
    bind_workforce_videos_from_folder(get_db, WORKFORCE_VIDEOS_DIR)
    ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
    return jsonify({'videos': _workforce_videos_payload(), 'playback_only': True})


@app.route('/api/workforce/video/<machine_id>')
def api_workforce_video(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'error': 'Unknown machine'}), 404
    bind_workforce_videos_from_folder(get_db, WORKFORCE_VIDEOS_DIR)
    path = _resolve_workforce_video_path(machine_id)
    if not path or not os.path.isfile(path):
        logger.warning('Workforce video missing for %s (path=%r)', machine_id, path)
        return jsonify({'error': 'Video not found'}), 404
    resp = send_file(
        path,
        mimetype='video/mp4',
        conditional=True,
        max_age=0,
        etag=True,
        last_modified=os.path.getmtime(path),
    )
    resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
    resp.headers['Accept-Ranges'] = 'bytes'
    return resp


def _workforce_dashboard_status():
    """Demo-filled dashboard for playback-only live view."""
    vmap = get_workforce_video_map(get_db, WORKFORCE_VIDEOS_DIR)
    if workforce_procs:
        data = get_aggregate_status()
        for m in data.get('machines', []):
            mid = m.get('machine_id')
            if mid in vmap:
                ver = workforce_video_version(vmap[mid])
                m['video_url'] = url_for('feed_workforce', machine_id=mid) + f'?v={ver}'
                m['video_name'] = os.path.basename(vmap[mid])
        data['playback_only'] = False
        return data
    return get_workforce_demo_status(vmap)


@app.route('/modules/ups-panel')
def page_ups_module():
    return render_template('modules.html', active_module='ups')


@app.route('/modules/ups-panel/config')
def page_ups_config():
    return redirect(url_for('page_ups_apply'), code=302)


@app.route('/modules/ups-panel/apply')
def page_ups_apply():
    return render_template('ups_apply.html', module='panel')


# ---------- Workforce Monitoring API ----------


def _safe_save_video(file_storage, dest_dir, fname):
    """Save upload via temp file; use timestamped name if target is locked (e.g. OpenCV reader)."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)
    tmp = dest + '.uploading'
    file_storage.save(tmp)
    try:
        if os.path.isfile(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        os.replace(tmp, dest)
        return fname
    except OSError:
        base, ext = os.path.splitext(fname)
        fname = f'{base}_{int(time.time())}{ext}'
        dest = os.path.join(dest_dir, fname)
        try:
            os.replace(tmp, dest)
            return fname
        except OSError:
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise


_workforce_person_model = None
_workforce_ppe_model = None
_workforce_models_lock = threading.Lock()
_wf_start_lock = threading.Lock()
_wf_start_status = {}


def _get_workforce_models():
    global _workforce_person_model, _workforce_ppe_model
    with _workforce_models_lock:
        if _workforce_person_model is None:
            _workforce_person_model = YOLO(HEADCOUNT_MODEL_PATH)
        if _workforce_ppe_model is None:
            _workforce_ppe_model = YOLO(PPE_MODEL_PATH)
        return _workforce_person_model, _workforce_ppe_model


def _preload_workforce_models():
    try:
        _get_workforce_models()
        logger.info('Workforce YOLO models preloaded')
    except Exception as e:
        logger.error(f'Workforce model preload failed: {e}')


def _wf_clear_start_status(machine_id=None):
    with _wf_start_lock:
        if machine_id:
            _wf_start_status.pop(machine_id, None)
        else:
            _wf_start_status.clear()


def _workforce_get_frame(machine_id):
    proc = workforce_procs.get(machine_id)
    if proc:
        frame = proc.get_annotated_frame()
        if frame is not None:
            return frame
    reader = workforce_video_readers.get(machine_id)
    if reader:
        return reader.get_frame()
    return None


def _ups_get_frame(panel_id):
    proc = ups_procs.get(panel_id)
    if proc:
        frame = proc.get_annotated_frame()
        if frame is not None:
            return frame
    reader = ups_video_readers.get(panel_id)
    if reader:
        return reader.get_frame()
    return None


def _workforce_start_bg(machine_id, video_path, conf):
    global _wf_start_status
    try:
        person_m, ppe_m = _get_workforce_models()
        ok, msg = wf_start_machine(
            machine_id, video_path, person_m, ppe_m, conf,
            _save_alert_event, get_db, ALERTS_DIR,
        )
        conn = get_db()
        conn.execute(
            'UPDATE workforce_machines SET enabled=? WHERE machine_id=?',
            (1 if ok else 0, machine_id),
        )
        conn.commit()
        conn.close()
        with _wf_start_lock:
            _wf_start_status[machine_id] = {'ok': ok, 'message': msg, 'done': True}
        logger.info(f'WF bg start {machine_id}: ok={ok} msg={msg}')
    except Exception as e:
        logger.exception(f'WF bg start {machine_id}')
        with _wf_start_lock:
            _wf_start_status[machine_id] = {'ok': False, 'message': str(e), 'done': True}


@app.route('/api/workforce/machines', methods=['GET'])
def api_workforce_machines_list():
    conn = get_db()
    rows = conn.execute('SELECT * FROM workforce_machines ORDER BY machine_id').fetchall()
    conn.close()
    out = []
    for r in rows:
        if r['machine_id'] not in WORKFORCE_MACHINE_IDS:
            continue
        proc = workforce_procs.get(r['machine_id'])
        reader = workforce_video_readers.get(r['machine_id'])
        out.append({
            'machine_id': r['machine_id'],
            'cam_label': r['cam_label'],
            'video_path': r['video_path'] or '',
            'video_name': os.path.basename(r['video_path'] or '') if r['video_path'] else '',
            'enabled': bool(r['enabled']),
            'config': json.loads(r['config_json'] or '{}'),
            'running': proc is not None,
            'video_status': (reader.status if reader else 'inactive'),
            'stats': proc.stats if proc else None,
        })
    return jsonify({'machines': out})


@app.route('/api/workforce/machines/<machine_id>/upload', methods=['POST'])
def api_workforce_upload(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'ok': False, 'error': 'Invalid machine id'}), 400
    if 'video' not in request.files:
        return jsonify({'ok': False, 'error': 'No video file'}), 400
    f = request.files['video']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.mp4', '.avi', '.mov', '.mkv'):
        return jsonify({'error': 'Only video files (.mp4, .avi, .mov, .mkv) allowed'}), 400
    safe = secure_filename(f.filename)
    fname = f'{machine_id}_{safe}'
    wf_stop_machine(machine_id)
    _wf_clear_start_status(machine_id)
    time.sleep(0.4)
    try:
        fname = _safe_save_video(f, WORKFORCE_VIDEOS_DIR, fname)
    except OSError as e:
        logger.error(f'WF upload {machine_id}: {e}')
        return jsonify({'ok': False, 'error': 'Could not save video — stop monitoring and try again'}), 500
    if not os.path.isfile(os.path.join(WORKFORCE_VIDEOS_DIR, fname)):
        return jsonify({'ok': False, 'error': 'Video file could not be saved'}), 500
    conn = get_db()
    conn.execute(
        'UPDATE workforce_machines SET video_path=?, updated_time=CURRENT_TIMESTAMP WHERE machine_id=?',
        (fname, machine_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'video_path': fname, 'video_name': os.path.basename(fname).split('_', 1)[-1] if '_' in fname else safe})


@app.route('/api/workforce/machines/<machine_id>/enable', methods=['POST'])
def api_workforce_enable(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'error': 'Invalid machine id'}), 400
    data = request.get_json(silent=True) or {}
    enabled = 1 if data.get('enabled', True) else 0
    conn = get_db()
    conn.execute('UPDATE workforce_machines SET enabled=? WHERE machine_id=?', (enabled, machine_id))
    conn.commit()
    row = conn.execute('SELECT * FROM workforce_machines WHERE machine_id=?', (machine_id,)).fetchone()
    conn.close()
    if enabled:
        if not row or not row['video_path']:
            return jsonify({'ok': False, 'error': 'No video configured — upload a file first'}), 400
        person_m, ppe_m = _get_workforce_models()
        vp = row['video_path']
        if not os.path.isabs(vp):
            vp = os.path.join(WORKFORCE_VIDEOS_DIR, vp)
        if not os.path.isfile(vp):
            return jsonify({'ok': False, 'error': f'Video file not found: {os.path.basename(vp)}'}), 400
        cfg = json.loads(row['config_json'] or '{}')
        ok, msg = wf_start_machine(machine_id, vp, person_m, ppe_m, cfg.get('confidence', 0.35),
                                   _save_alert_event, get_db, ALERTS_DIR)
        return jsonify({'ok': ok, 'running': ok, 'message': msg,
                        'error': None if ok else msg})
    wf_stop_machine(machine_id)
    _wf_clear_start_status(machine_id)
    return jsonify({'ok': True, 'message': 'stopped'})


@app.route('/api/workforce/frame/<machine_id>')
def api_workforce_frame(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'error': 'Invalid machine id'}), 400
    if machine_id not in workforce_video_readers:
        path = _resolve_workforce_video_path(machine_id)
        if path:
            ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
    frame = _workforce_get_frame(machine_id)
    if frame is None:
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No stream', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
        _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return Response(buf.tobytes(), mimetype='image/jpeg')
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/ups/frame/<panel_id>')
def api_ups_frame(panel_id):
    if panel_id not in UPS_PANEL_IDS:
        return jsonify({'error': 'Invalid panel id'}), 400
    frame = _ups_get_frame(panel_id)
    if frame is None:
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No stream', (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 1)
        _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return Response(buf.tobytes(), mimetype='image/jpeg')
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/workforce/machines/<machine_id>/start', methods=['POST'])
def api_workforce_start(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'ok': False, 'error': 'Invalid machine id'}), 400
    if 'video' in request.files and request.files['video'].filename:
        f = request.files['video']
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ('.mp4', '.avi', '.mov', '.mkv'):
            return jsonify({'ok': False, 'error': 'Only video files allowed'}), 400
        safe = secure_filename(f.filename)
        fname = f'{machine_id}_{safe}'
        wf_stop_machine(machine_id)
        _wf_clear_start_status(machine_id)
        time.sleep(0.4)
        try:
            fname = _safe_save_video(f, WORKFORCE_VIDEOS_DIR, fname)
        except OSError as e:
            logger.error(f'WF start upload {machine_id}: {e}')
            return jsonify({'ok': False, 'error': 'Could not save video — try again'}), 500
        conn = get_db()
        conn.execute(
            'UPDATE workforce_machines SET video_path=?, updated_time=CURRENT_TIMESTAMP WHERE machine_id=?',
            (fname, machine_id)
        )
        conn.commit()
        conn.close()
    conn = get_db()
    row = conn.execute('SELECT * FROM workforce_machines WHERE machine_id=?', (machine_id,)).fetchone()
    conn.close()
    if not row or not row['video_path']:
        return jsonify({'ok': False, 'error': 'Choose a video file first'}), 400
    vp = row['video_path']
    if not os.path.isabs(vp):
        vp = os.path.join(WORKFORCE_VIDEOS_DIR, vp)
    if not os.path.isfile(vp):
        return jsonify({'ok': False, 'error': f'Video file not found: {os.path.basename(vp)}'}), 400
    cfg = json.loads(row['config_json'] or '{}')
    conf = cfg.get('confidence', 0.35)
    with _wf_start_lock:
        cur = _wf_start_status.get(machine_id, {})
        if cur.get('done') is False:
            return jsonify({'ok': True, 'pending': True, 'message': 'Already starting...'})
        _wf_start_status[machine_id] = {'ok': None, 'message': 'starting', 'done': False}
    threading.Thread(
        target=_workforce_start_bg,
        args=(machine_id, vp, conf),
        daemon=True,
        name=f'WF-Start-{machine_id}',
    ).start()
    return jsonify({
        'ok': True,
        'pending': True,
        'running': False,
        'message': 'Starting...',
        'video_name': os.path.basename(row['video_path']),
    })


@app.route('/api/workforce/machines/<machine_id>/start-status', methods=['GET'])
def api_workforce_start_status(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'ok': False, 'error': 'Invalid machine id'}), 400
    with _wf_start_lock:
        st = dict(_wf_start_status.get(machine_id, {'done': True, 'ok': None, 'message': ''}))
    running = machine_id in workforce_procs
    reader = workforce_video_readers.get(machine_id)
    st['running'] = running
    st['video_status'] = reader.status if reader else 'inactive'
    if st.get('done') and st.get('ok') is False:
        st['error'] = st.get('message') or 'Start failed'
    return jsonify(st)


@app.route('/api/workforce/start-all', methods=['POST'])
def api_workforce_start_all():
    person_m, ppe_m = _get_workforce_models()
    conn = get_db()
    rows = conn.execute('SELECT * FROM workforce_machines WHERE video_path IS NOT NULL AND video_path != ""').fetchall()
    started = 0
    for row in rows:
        mid = row['machine_id']
        vp = row['video_path']
        if not os.path.isabs(vp):
            vp = os.path.join(WORKFORCE_VIDEOS_DIR, vp)
        if os.path.isfile(vp):
            cfg = json.loads(row['config_json'] or '{}')
            wf_start_machine(mid, vp, person_m, ppe_m, cfg.get('confidence', 0.35),
                             _save_alert_event, get_db, ALERTS_DIR)
            conn.execute('UPDATE workforce_machines SET enabled=1 WHERE machine_id=?', (mid,))
            started += 1
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'started': started})


@app.route('/api/workforce/stop-all', methods=['POST'])
def api_workforce_stop_all():
    for mid in list(workforce_procs.keys()):
        wf_stop_machine(mid)
    _wf_clear_start_status()
    conn = get_db()
    conn.execute('UPDATE workforce_machines SET enabled=0')
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/workforce/status', methods=['GET'])
def api_workforce_status():
    data = _workforce_dashboard_status()
    persist_workforce_dashboard(
        get_db, data, get_utilization_chart(get_db), {
            'compliance_pct': data.get('ppe_compliance_pct', 87),
            'shift_violations': 3,
            'today_violations': 5,
            'active_violations': data.get('ppe_violations', 0),
        },
    )
    return jsonify(data)


@app.route('/api/workforce/chart', methods=['GET'])
def api_workforce_chart():
    return jsonify(get_utilization_chart(get_db))


@app.route('/api/workforce/ppe-stats', methods=['GET'])
def api_workforce_ppe_stats():
    data = _workforce_dashboard_status()
    stats = get_ppe_stats(get_db)
    shift_v = stats['shift'] if stats['shift'] else 3
    today_v = stats['today'] if stats['today'] else 5
    return jsonify({
        'compliance_pct': data.get('ppe_compliance_pct', 87),
        'shift_violations': shift_v,
        'today_violations': today_v,
        'active_violations': data.get('ppe_violations', 0),
    })


def _chat_parse_hours(text):
    t = (text or '').lower()
    for pat in (
        r'(\d+)\s*(?:hours?|hrs?|h)\b',
        r'past\s+(\d+)',
        r'last\s+(\d+)',
    ):
        m = re.search(pat, t)
        if m:
            return max(1, min(int(m.group(1)), 72))
    return 1


def _chat_parse_machine_id(text):
    t = (text or '').lower()
    m = re.search(r'machine[\s#-]*(\d+)', t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 6:
            return f'MACHINE-{n:02d}'
    m = re.search(r'\bm(\d+)\b', t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 6:
            return f'MACHINE-{n:02d}'
    return None


def _chat_workforce_answer(question):
    q = (question or '').strip()
    ql = q.lower()
    hours = _chat_parse_hours(ql)
    machine_id = _chat_parse_machine_id(ql)

    if not machine_id and re.search(r'\b(all|every)\b.*machine|machines\b', ql):
        machine_id = None
    elif not machine_id and re.search(r'machine|mach\b', ql):
        machine_id = 'MACHINE-01'

    wants_alerts = any(w in ql for w in ('alert', 'violation', 'ppe', 'manpower', 'incident', 'event', 'generated'))
    wants_status = any(w in ql for w in ('status', 'worker', 'running', 'manned', 'online', 'count'))
    wants_kpi = any(w in ql for w in ('kpi', 'total', 'how many', 'compliance', 'camera', 'utilization', 'chart', 'dashboard'))

    lines = []
    conn = get_db()
    try:
        snap_hints = chat_answer_from_snapshots(conn, 'workforce', question)
        if snap_hints:
            lines.extend(snap_hints)
    finally:
        pass

    dash = _workforce_dashboard_status()
    if wants_kpi or (not wants_alerts and not wants_status):
        lines.append(
            f"Workforce dashboard: {dash['manned']} manned, {dash['unmanned']} unmanned, "
            f"{dash['ppe_violations']} PPE violations, {dash['cameras_online']}/6 cameras live, "
            f"PPE compliance {dash.get('ppe_compliance_pct', 87)}%."
        )

    if wants_status or (not wants_alerts and machine_id):
        proc = workforce_procs.get(machine_id) if machine_id else None
        if proc:
            s = proc.stats
            lines.append(
                f"{machine_id} now: {s.get('worker_count', 0)} workers, "
                f"status {s.get('status', 'unknown')}, "
                f"PPE violations {s.get('helmet_violations', 0)}."
            )
        else:
            for m in dash.get('machines', []):
                if machine_id and m.get('machine_id') != machine_id:
                    continue
                if machine_id or m.get('machine_id') == 'MACHINE-01':
                    lines.append(
                        f"{m.get('machine_id')}: {m.get('worker_count', 0)} workers, "
                        f"status={m.get('status')}, badge={m.get('badge')}."
                    )
                    if machine_id:
                        break

    if wants_alerts or not wants_status:
        since_mod = f'-{int(hours)} hours'
        try:
            if machine_id:
                rows = conn.execute(
                    '''SELECT alert_label, severity, created_time
                       FROM alert_events
                       WHERE detection_type='workforce' AND camera_id=?
                         AND datetime(created_time) >= datetime('now', ?)
                       ORDER BY created_time DESC LIMIT 30''',
                    (machine_id, since_mod),
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT camera_id, alert_label, severity, created_time
                       FROM alert_events
                       WHERE detection_type='workforce'
                         AND datetime(created_time) >= datetime('now', ?)
                       ORDER BY created_time DESC LIMIT 40''',
                    (since_mod,),
                ).fetchall()
        except Exception:
            rows = []

        proc = workforce_procs.get(machine_id) if machine_id else None
        if proc:
            for a in list(proc.recent_alerts)[:5]:
                lines.append(f"Recent (live): {a.get('title', 'Alert')} — {a.get('sub', '')}")

        if not rows:
            demo_alerts = dash.get('alerts') or []
            if demo_alerts and (wants_alerts or 'alert' in ql):
                lines.append(f"Active dashboard alerts ({len(demo_alerts)}):")
                for a in demo_alerts[:6]:
                    lines.append(f"• {a.get('title', 'Alert')} — {a.get('sub', '')} ({a.get('time', '')})")
            elif wants_alerts:
                target = machine_id or 'all machines'
                lines.append(f"No alerts for {target} in the past {hours} hour(s).")
        else:
            lines.append(f"Found {len(rows)} alert(s) in the past {hours} hour(s)" +
                         (f" for {machine_id}:" if machine_id else ':'))
            for r in rows[:12]:
                if machine_id:
                    lines.append(f"• {r['created_time']} — {r['alert_label']} ({r['severity']})")
                else:
                    lines.append(f"• {r['camera_id']} {r['created_time']} — {r['alert_label']}")

    conn.close()

    if not lines:
        return (
            "Try asking: \"machine 1 past 1 hr what alerts generated\", "
            "\"how many machines manned\", or \"ppe compliance\"."
        )
    seen = set()
    out = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return '\n'.join(out)


def _chat_panel_answer(question):
    ql = (question or '').lower()
    hours = _chat_parse_hours(ql)
    panel_id = 'UPS-PANEL-01'
    wants_alerts = any(w in ql for w in ('alert', 'fault', 'alarm', 'event', 'incident', 'generated'))
    wants_status = any(w in ql for w in ('status', 'voltage', 'current', 'meter', 'online', 'running', 'ac', 'dc'))
    wants_events = 'event' in ql or wants_alerts
    wants_summary = any(w in ql for w in ('summary', 'normal', 'warning', 'total panel'))

    lines = []
    conn = get_db()
    try:
        snap_hints = chat_answer_from_snapshots(conn, 'panel', question)
        if snap_hints:
            lines.extend(snap_hints)
    except Exception:
        pass

    proc = ups_procs.get(panel_id)
    panel_data = get_panel_status(panel_id)
    summary = ups_get_summary()

    if wants_status or wants_summary or (not wants_alerts and not wants_events):
        if proc:
            lines.append(
                f"Panel status (live): AC {proc.ac_voltage:.1f}V, DC {proc.dc_voltage:.1f}V, "
                f"current {proc.dc_current:.1f}A, online={proc.online}."
            )
        else:
            lines.append(
                f"Panel status: AC {panel_data.get('ac_voltage', 415):.1f}V, "
                f"DC {panel_data.get('dc_voltage', 125.1):.1f}V, "
                f"current {panel_data.get('dc_current', 284):.1f}A, "
                f"online={panel_data.get('online', True)}."
            )
        if wants_summary:
            lines.append(
                f"Fleet summary: {summary.get('normal', 9)} normal, "
                f"{summary.get('warning', 1)} warning, {summary.get('alarm', 2)} alarm."
            )

    if wants_events or wants_alerts:
        since_mod = f'-{int(hours)} hours'
        try:
            ev_rows = conn.execute(
                '''SELECT event_name, status, duration_sec, timestamp
                   FROM ups_panel_events
                   WHERE panel_id=? AND datetime(timestamp) >= datetime('now', ?)
                   ORDER BY timestamp DESC LIMIT 20''',
                (panel_id, since_mod),
            ).fetchall()
            alert_rows = conn.execute(
                '''SELECT alert_label, severity, created_time
                   FROM alert_events
                   WHERE detection_type='ups_panel' AND camera_id=?
                     AND datetime(created_time) >= datetime('now', ?)
                   ORDER BY created_time DESC LIMIT 20''',
                (panel_id, since_mod),
            ).fetchall()
        except Exception:
            ev_rows = []
            alert_rows = []

        if proc and getattr(proc, 'recent_events', None):
            for ev in list(proc.recent_events)[:3]:
                lines.append(f"Recent: {ev.get('event', 'Event')} — {ev.get('time', '')}")

        if ev_rows:
            lines.append(f"Panel events (past {hours}h): {len(ev_rows)}")
            for r in ev_rows[:8]:
                lines.append(f"• {r['timestamp']} — {r['event_name']} ({r['status']})")
        elif panel_data.get('recent_events'):
            lines.append('Recent dashboard events:')
            for ev in panel_data['recent_events'][:5]:
                lines.append(f"• {ev.get('time', '')} — {ev.get('event', '')} ({ev.get('status', '')})")

        if alert_rows:
            lines.append(f"Alerts (past {hours}h): {len(alert_rows)}")
            for r in alert_rows[:8]:
                lines.append(f"• {r['created_time']} — {r['alert_label']} ({r['severity']})")
        elif not ev_rows and wants_alerts:
            lines.append(f"No panel events or alerts in the past {hours} hour(s).")

    conn.close()

    if not lines:
        return 'Try: "past 1 hr what alerts generated", "panel status", or "ac voltage".'
    seen = set()
    out = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return '\n'.join(out)


@app.route('/api/module-chat', methods=['POST'])
def api_module_chat():
    try:
        data = request.get_json(silent=True) or {}
        module = (data.get('module') or '').strip().lower()
        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        if module == 'workforce':
            answer = _chat_workforce_answer(question)
        elif module == 'panel':
            answer = _chat_panel_answer(question)
        else:
            return jsonify({'error': 'Unknown module. Reload the page and try again.'}), 400
        try:
            conn = get_db()
            log_chat_message(conn, module, 'user', question)
            log_chat_message(conn, module, 'assistant', answer)
            conn.commit()
            conn.close()
        except Exception:
            pass
        return jsonify({'answer': answer})
    except Exception as e:
        logger.exception('module-chat failed')
        return jsonify({'error': f'Assistant error: {e}'}), 500


# ---------- UPS Panel Monitoring API ----------


@app.route('/api/ups/panels', methods=['GET'])
def api_ups_panels_list():
    conn = get_db()
    rows = conn.execute('SELECT * FROM ups_panels ORDER BY panel_id').fetchall()
    conn.close()
    out = []
    for r in rows:
        proc = ups_procs.get(r['panel_id'])
        reader = ups_video_readers.get(r['panel_id'])
        out.append({
            'panel_id': r['panel_id'],
            'cam_label': r['cam_label'],
            'video_path': r['video_path'] or '',
            'video_name': os.path.basename(r['video_path'] or '') if r['video_path'] else '',
            'enabled': bool(r['enabled']),
            'config': json.loads(r['config_json'] or '{}'),
            'running': proc is not None,
            'video_status': reader.status if reader else 'inactive',
            'stats': proc.stats if proc else None,
        })
    return jsonify({'panels': out})


@app.route('/api/ups/panels/<panel_id>/upload', methods=['POST'])
def api_ups_upload(panel_id):
    if panel_id not in UPS_PANEL_IDS:
        return jsonify({'error': 'Invalid panel id'}), 400
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    f = request.files['video']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.mp4', '.avi', '.mov', '.mkv'):
        return jsonify({'error': 'Only video files (.mp4, .avi, .mov, .mkv) allowed'}), 400
    safe = secure_filename(f.filename) or f'video{ext}'
    fname = f'{panel_id}_{safe}'
    ups_stop_panel(panel_id)
    time.sleep(0.25)
    try:
        fname = _safe_save_video(f, UPS_VIDEOS_DIR, fname)
    except OSError as e:
        logger.error(f'UPS upload {panel_id}: {e}')
        return jsonify({'error': 'Could not save video — stop monitoring and try again'}), 500
    dest = os.path.join(UPS_VIDEOS_DIR, fname)
    if not os.path.isfile(dest) or os.path.getsize(dest) < 1000:
        return jsonify({'error': 'Video file could not be saved'}), 500
    conn = get_db()
    conn.execute(
        'UPDATE ups_panels SET video_path=?, updated_time=CURRENT_TIMESTAMP WHERE panel_id=?',
        (fname, panel_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'video_path': fname, 'video_name': safe})


@app.route('/api/ups/panels/<panel_id>/enable', methods=['POST'])
def api_ups_enable(panel_id):
    if panel_id not in UPS_PANEL_IDS:
        return jsonify({'error': 'Invalid panel id'}), 400
    data = request.get_json(silent=True) or {}
    enabled = 1 if data.get('enabled', True) else 0
    conn = get_db()
    conn.execute('UPDATE ups_panels SET enabled=? WHERE panel_id=?', (enabled, panel_id))
    conn.commit()
    row = conn.execute('SELECT * FROM ups_panels WHERE panel_id=?', (panel_id,)).fetchone()
    conn.close()
    if enabled and row and row['video_path']:
        vp = row['video_path']
        if not os.path.isabs(vp):
            vp = os.path.join(UPS_VIDEOS_DIR, vp)
        cfg = json.loads(row['config_json'] or '{}')
        from ups_panel_monitoring import DEFAULT_ROIS
        roi = cfg.get('rois', DEFAULT_ROIS)
        ok, msg = ups_start_panel(panel_id, vp, roi, _save_alert_event, get_db)
        return jsonify({'ok': ok, 'message': msg})
    ups_stop_panel(panel_id)
    return jsonify({'ok': True, 'message': 'stopped'})


@app.route('/api/ups/start', methods=['POST'])
def api_ups_start():
    panel_id = 'UPS-PANEL-01'
    if 'video' in request.files and request.files['video'].filename:
        f = request.files['video']
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ('.mp4', '.avi', '.mov', '.mkv'):
            return jsonify({'ok': False, 'error': 'Only video files allowed'}), 400
        safe = secure_filename(f.filename) or f'video{ext}'
        fname = f'{panel_id}_{safe}'
        ups_stop_panel(panel_id)
        time.sleep(0.25)
        try:
            fname = _safe_save_video(f, UPS_VIDEOS_DIR, fname)
        except OSError as e:
            logger.error(f'UPS start upload {panel_id}: {e}')
            return jsonify({'ok': False, 'error': 'Could not save video — stop monitoring and try again'}), 500
        dest = os.path.join(UPS_VIDEOS_DIR, fname)
        if not os.path.isfile(dest):
            return jsonify({'ok': False, 'error': 'Video save failed'}), 500
        conn = get_db()
        conn.execute(
            'UPDATE ups_panels SET video_path=?, updated_time=CURRENT_TIMESTAMP WHERE panel_id=?',
            (fname, panel_id)
        )
        conn.commit()
        conn.close()

    conn = get_db()
    row = conn.execute('SELECT * FROM ups_panels WHERE panel_id=?', (panel_id,)).fetchone()
    conn.close()
    if not row or not row['video_path']:
        return jsonify({'ok': False, 'error': 'Choose a video file first'}), 400
    vp = row['video_path']
    if not os.path.isabs(vp):
        vp = os.path.join(UPS_VIDEOS_DIR, vp)
    if not os.path.isfile(vp):
        return jsonify({'ok': False, 'error': f'Video file not found: {os.path.basename(vp)}'}), 400
    cfg = json.loads(row['config_json'] or '{}')
    from ups_panel_monitoring import DEFAULT_ROIS
    roi = cfg.get('rois', DEFAULT_ROIS)
    ok, msg = ups_start_panel(panel_id, vp, roi, _save_alert_event, get_db)
    conn = get_db()
    conn.execute('UPDATE ups_panels SET enabled=1 WHERE panel_id=?', (panel_id,))
    conn.commit()
    conn.close()
    running = panel_id in ups_procs
    reader = ups_video_readers.get(panel_id)
    has_frame = reader.get_frame() is not None if reader else False
    success = ok and running and has_frame
    logger.info(
        f'UPS start {panel_id}: ok={ok} running={running} has_frame={has_frame} '
        f'video={os.path.basename(vp)} reader_status={reader.status if reader else "none"}'
    )
    if not success:
        return jsonify({
            'ok': False,
            'running': running,
            'error': msg or 'Failed to start video processing',
            'video_name': os.path.basename(row['video_path']),
        }), 500
    return jsonify({
        'ok': True,
        'running': True,
        'message': msg,
        'video_name': os.path.basename(row['video_path']),
        'video_status': reader.status if reader else 'inactive',
        'stream_url': f'/video_feed/ups/{panel_id}',
    })


@app.route('/api/ups/stop', methods=['POST'])
def api_ups_stop():
    ups_stop_panel('UPS-PANEL-01')
    conn = get_db()
    conn.execute('UPDATE ups_panels SET enabled=0 WHERE panel_id=?', ('UPS-PANEL-01',))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/ups/status', methods=['GET'])
def api_ups_status():
    panel = get_panel_status('UPS-PANEL-01')
    summary = ups_get_summary()
    conn = get_db()
    row = conn.execute('SELECT video_path, enabled FROM ups_panels WHERE panel_id=?', ('UPS-PANEL-01',)).fetchone()
    conn.close()
    if row:
        panel['video_name'] = os.path.basename(row['video_path'] or '') if row['video_path'] else ''
        panel['enabled'] = bool(row['enabled'])
    panel_id = 'UPS-PANEL-01'
    reader = ups_video_readers.get(panel_id)
    panel['running'] = panel_id in ups_procs
    panel['has_video'] = bool(row and row['video_path']) if row else False
    panel['stream_url'] = f'/video_feed/ups/{panel_id}'
    if reader and not panel.get('video_status'):
        panel['video_status'] = reader.status
    payload = {'panel': panel, 'summary': summary}
    persist_panel_dashboard(get_db, payload, ups_get_trend_data(get_db))
    return jsonify(payload)


@app.route('/api/ups/trend', methods=['GET'])
def api_ups_trend():
    return jsonify(ups_get_trend_data(get_db))


# ---------- System: users, sessions, storage & local config ----------


@app.route('/api/system/users', methods=['GET'])
def api_system_users_list():
    err = _require_admin_api()
    if err:
        return err
    conn = get_db()
    rows = conn.execute(
        'SELECT id, username, level, created_time FROM app_users ORDER BY username COLLATE NOCASE'
    ).fetchall()
    conn.close()
    return jsonify(
        {
            'users': [
                {
                    'id': r['id'],
                    'username': r['username'],
                    'level': r['level'],
                    'created_time': r['created_time'],
                }
                for r in rows
            ]
        }
    )


@app.route('/api/system/users', methods=['POST'])
def api_system_users_add():
    err = _require_admin_api()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    level = (data.get('level') or 'user').strip().lower()
    if level not in ('admin', 'user'):
        level = 'user'
    if not username or len(username) < 2:
        return jsonify({'error': 'Username must be at least 2 characters'}), 400
    if not password or len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400
    h = generate_password_hash(password)
    try:
        conn = get_db()
        conn.execute(
            'INSERT INTO app_users (username, password_hash, level) VALUES (?,?,?)',
            (username, h, level),
        )
        conn.commit()
        uid = conn.execute('SELECT last_insert_rowid() AS id').fetchone()['id']
        conn.close()
        return jsonify({'ok': True, 'id': uid})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409


@app.route('/api/system/users/<int:user_id>', methods=['PUT'])
def api_system_users_update(user_id):
    err = _require_admin_api()
    if err:
        return err
    data = request.get_json(silent=True) or {}
    level = data.get('level')
    password = data.get('password')
    conn = get_db()
    row = conn.execute('SELECT id, level FROM app_users WHERE id=?', (user_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    if level is not None:
        level = str(level).strip().lower()
        if level not in ('admin', 'user'):
            conn.close()
            return jsonify({'error': 'Invalid level'}), 400
        if row['level'] == 'admin' and level == 'user':
            ac = conn.execute("SELECT COUNT(*) AS c FROM app_users WHERE level='admin'").fetchone()['c']
            if ac <= 1:
                conn.close()
                return jsonify({'error': 'Cannot demote the last administrator'}), 400
        conn.execute('UPDATE app_users SET level=? WHERE id=?', (level, user_id))
    if password:
        if len(str(password)) < 4:
            conn.close()
            return jsonify({'error': 'Password must be at least 4 characters'}), 400
        conn.execute(
            'UPDATE app_users SET password_hash=? WHERE id=?',
            (generate_password_hash(str(password)), user_id),
        )
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/system/users/<int:user_id>', methods=['DELETE'])
def api_system_users_delete(user_id):
    err = _require_admin_api()
    if err:
        return err
    if int(user_id) == int(session.get('user_id') or 0):
        return jsonify({'error': 'You cannot delete your own account while logged in'}), 400
    conn = get_db()
    row = conn.execute('SELECT level FROM app_users WHERE id=?', (user_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    if row['level'] == 'admin':
        ac = conn.execute("SELECT COUNT(*) AS c FROM app_users WHERE level='admin'").fetchone()['c']
        if ac <= 1:
            conn.close()
            return jsonify({'error': 'Cannot delete the last administrator'}), 400
    conn.execute('DELETE FROM app_users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/system/online-users', methods=['GET'])
def api_system_online_users():
    _auth_cleanup_sessions()
    conn = get_db()
    rows = conn.execute(
        """
        SELECT username, level, ip, last_activity
        FROM user_active_sessions
        WHERE datetime(last_activity) >= datetime('now', ?)
        ORDER BY last_activity DESC
        """,
        (f'-{_ONLINE_SESSION_MAX_MIN} minutes',),
    ).fetchall()
    conn.close()
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                'no': i + 1,
                'username': r['username'],
                'level': 'Administrator' if r['level'] == 'admin' else 'User',
                'ip': r['ip'] or '—',
                'last_activity': r['last_activity'],
            }
        )
    return jsonify({'sessions': out})


@app.route('/api/system/record-schedules', methods=['GET'])
def api_record_schedules_get():
    data = _system_kv_get('record_schedules', {})
    return jsonify({'schedules': data})


@app.route('/api/system/record-schedules', methods=['POST'])
def api_record_schedules_save():
    data = request.get_json(silent=True) or {}
    schedules = data.get('schedules')
    if not isinstance(schedules, dict):
        return jsonify({'error': 'schedules object required'}), 400
    _system_kv_set('record_schedules', schedules)
    return jsonify({'ok': True})


@app.route('/api/system/holidays', methods=['GET'])
def api_holidays_get():
    data = _system_kv_get(
        'holidays',
        {
            'holiday': [],
            'other': [],
        },
    )
    return jsonify(data)


@app.route('/api/system/holidays', methods=['POST'])
def api_holidays_save():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid body'}), 400
    payload = {
        'holiday': data.get('holiday') if isinstance(data.get('holiday'), list) else [],
        'other': data.get('other') if isinstance(data.get('other'), list) else [],
    }
    _system_kv_set('holidays', payload)
    return jsonify({'ok': True})


@app.route('/api/system/local-settings', methods=['GET'])
def api_local_settings_get():
    data = _system_kv_get('local_settings', _local_settings_default())
    return jsonify(_local_settings_for_api(data))


@app.route('/api/system/local-settings', methods=['POST'])
def api_local_settings_save():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid body'}), 400
    existing = _system_kv_get('local_settings', _local_settings_default())
    merged = _merge_local_settings(data, existing)
    _system_kv_set('local_settings', merged)
    return jsonify({'ok': True})


@app.route('/api/system/email-alerts/test', methods=['POST'])
def api_email_alerts_test():
    err = _require_admin_api()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    existing = _system_kv_get('local_settings', _local_settings_default())
    cfg = _merge_local_settings({'email_alerts': body.get('email_alerts') or body}, existing).get('email_alerts', {})
    sender = (cfg.get('sender_email') or '').strip()
    try:
        _send_smtp_email(
            sender,
            sender,
            'Vision AI — Test Email',
            'This is a test alert email from the Vision AI platform. SMTP settings are working.',
            cfg.get('app_password'),
            cfg.get('smtp_host') or 'smtp.gmail.com',
            cfg.get('smtp_port') or 587,
        )
        return jsonify({'ok': True, 'message': f'Test email sent to {sender}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def _detect_local_scan_subnet():
    """Auto-detect the current host subnet (application server)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        return local_ip.rsplit('.', 1)[0] + '.0/24'
    except Exception:
        return '192.168.1.0/24'


def _extract_ip_from_rtsp_url(rtsp_url):
    try:
        host = urlsplit(rtsp_url or '').hostname
        if host:
            ipaddress.ip_address(host)
            return host
    except Exception:
        pass
    m = re.search(r'@(\d{1,3}(?:\.\d{1,3}){3})', rtsp_url or '')
    if m:
        try:
            ipaddress.ip_address(m.group(1))
            return m.group(1)
        except ValueError:
            pass
    return None


def _infer_camera_subnet_from_db():
    """Most common /24 prefix from configured camera RTSP URLs."""
    try:
        conn = get_db()
        rows = conn.execute(
            'SELECT rtsp_url FROM cameras WHERE rtsp_url IS NOT NULL AND rtsp_url != ""'
        ).fetchall()
        conn.close()
        counts = {}
        for row in rows:
            ip = _extract_ip_from_rtsp_url(row['rtsp_url'])
            if not ip:
                continue
            cidr = '.'.join(ip.split('.')[:3]) + '.0/24'
            counts[cidr] = counts.get(cidr, 0) + 1
        if counts:
            return max(counts, key=counts.get)
    except Exception:
        pass
    return None


def _resolve_camera_scan_subnet(requested=''):
    """Subnet for IP camera discovery (camera VLAN, not the app server)."""
    if requested and requested != 'detect_only':
        try:
            return str(ipaddress.ip_network(requested, strict=False))
        except ValueError:
            logger.warning('Invalid scan subnet requested: %r', requested)
    if VISION_CAMERA_SCAN_SUBNET:
        try:
            return str(ipaddress.ip_network(VISION_CAMERA_SCAN_SUBNET, strict=False))
        except ValueError:
            logger.warning('Invalid VISION_CAMERA_SCAN_SUBNET=%r', VISION_CAMERA_SCAN_SUBNET)
    inferred = _infer_camera_subnet_from_db()
    if inferred:
        return inferred
    local = _detect_local_scan_subnet()
    logger.info(
        'Camera scan using server subnet %s. Set VISION_CAMERA_SCAN_SUBNET or pass ?subnet= for the camera VLAN.',
        local,
    )
    return local


def _read_probe_response(sock, limit=2048):
    chunks = []
    total = 0
    while total < limit:
        try:
            chunk = sock.recv(min(512, limit - total))
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if len(chunk) < 512:
            break
    return b''.join(chunks).decode('latin-1', errors='ignore')


def _guess_camera_vendor(signature_text):
    text = (signature_text or '').lower()
    if any(token in text for token in ('hikvision', 'app-webs', 'isapi')):
        return 'Hikvision'
    if any(token in text for token in ('cp plus', 'cpplus')):
        return 'CP Plus'
    if 'honeywell' in text:
        return 'Honeywell'
    if any(token in text for token in ('dahua', 'dhip', 'magicbox')):
        return 'Dahua'
    if 'onvif' in text:
        return 'ONVIF'
    if any(token in text for token in ('network camera', 'ip camera', 'ipcam', 'nvr')):
        return 'Generic Camera'
    return ''


def _probe_rtsp_service(ip_str, port, timeout=0.7):
    try:
        with socket.create_connection((ip_str, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            request_data = (
                f"OPTIONS rtsp://{ip_str}:{port}/ RTSP/1.0\r\n"
                "CSeq: 1\r\n"
                "User-Agent: VisionAI-Discovery/1.0\r\n"
                "\r\n"
            )
            sock.sendall(request_data.encode('ascii', errors='ignore'))
            response_text = _read_probe_response(sock, limit=1024)
        response_lower = response_text.lower()
        if 'rtsp/' not in response_lower:
            return None
        return {
            'port': port,
            'vendor': _guess_camera_vendor(response_text) or 'Generic Camera',
            'signature': response_text[:300]
        }
    except Exception:
        return None


def _probe_http_camera_service(ip_str, port, timeout=0.7):
    probe_paths = (
        '/ISAPI/System/deviceInfo',
        '/onvif/device_service',
        '/cgi-bin/magicBox.cgi?action=getSystemInfo',
        '/',
    )
    for path in probe_paths:
        try:
            with socket.create_connection((ip_str, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                request_data = (
                    f"GET {path} HTTP/1.1\r\n"
                    f"Host: {ip_str}\r\n"
                    "User-Agent: VisionAI-Discovery/1.0\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                )
                sock.sendall(request_data.encode('ascii', errors='ignore'))
                response_text = _read_probe_response(sock, limit=2048)
        except Exception:
            continue

        if not response_text:
            continue

        response_lower = response_text.lower()
        vendor = _guess_camera_vendor(response_text)
        looks_like_camera = (
            bool(vendor) or
            ('www-authenticate:' in response_lower and any(
                token in response_lower for token in ('hikvision', 'dahua', 'camera', 'nvr', 'onvif')
            )) or
            ('server:' in response_lower and any(
                token in response_lower for token in ('app-webs', 'hikvision', 'dahua', 'ip camera')
            ))
        )
        if looks_like_camera:
            return {
                'port': port,
                'vendor': vendor or 'Generic Camera',
                'signature': response_text[:400]
            }
    return None


def _subnet_mask_from_cidr(subnet_cidr):
    try:
        network = ipaddress.ip_network(subnet_cidr, strict=False)
        return str(network.netmask)
    except Exception:
        return '255.255.255.0'


def _protocol_label(vendor):
    text = (vendor or '').lower()
    if 'hikvision' in text:
        return 'HIKVISION'
    if 'honeywell' in text:
        return 'HONEYWELL'
    if 'cp plus' in text or 'cpplus' in text:
        return 'CP PLUS'
    if 'dahua' in text:
        return 'DAHUA'
    if 'onvif' in text:
        return 'ONVIF'
    return 'HIKVISION' if text else 'GENERIC'


def _lookup_mac_address(ip_str):
    """Best-effort MAC lookup from the local ARP table."""
    import re
    import subprocess
    try:
        if os.name == 'nt':
            out = subprocess.check_output(['arp', '-a', ip_str], timeout=2, text=True, errors='ignore')
        else:
            out = subprocess.check_output(['arp', '-n', ip_str], timeout=2, text=True, errors='ignore')
        match = re.search(r'([0-9A-Fa-f]{2}[-:]){5}[0-9A-Fa-f]{2}', out)
        if match:
            return match.group(0).replace('-', ':').upper()
    except Exception:
        pass
    return ''


def _extract_device_metadata(http_matches):
    serial_no = ''
    firmware_version = ''
    for match in http_matches or []:
        sig = match.get('signature', '') or ''
        serial_match = re.search(r'<serialNumber>([^<]+)</serialNumber>', sig, re.I)
        if serial_match:
            serial_no = serial_match.group(1).strip()
        model_match = re.search(r'<model>([^<]+)</model>', sig, re.I)
        if model_match and not serial_no:
            serial_no = model_match.group(1).strip()
        fw_match = re.search(r'<firmwareVersion>([^<]+)</firmwareVersion>', sig, re.I)
        if fw_match:
            firmware_version = fw_match.group(1).strip()
        if serial_no and firmware_version:
            break
    return serial_no, firmware_version


def _probe_camera_host(ip_str, subnet_mask='255.255.255.0'):
    """Return a validated camera/NVR host, or None for unrelated devices."""
    rtsp_match = None
    for port in (554, 8554):
        rtsp_match = _probe_rtsp_service(ip_str, port)
        if rtsp_match:
            break

    http_matches = []
    for port in (80, 8080, 8000):
        http_match = _probe_http_camera_service(ip_str, port)
        if http_match:
            http_matches.append(http_match)

    if not rtsp_match and not http_matches:
        return None

    vendor = ''
    for match in [rtsp_match] + http_matches:
        if match and match.get('vendor') and match['vendor'] != 'Generic Camera':
            vendor = match['vendor']
            break
    if not vendor:
        vendor = (rtsp_match or http_matches[0]).get('vendor', 'Generic Camera')

    preferred_port = rtsp_match['port'] if rtsp_match else http_matches[0]['port']
    management_port = 8000
    for match in http_matches:
        if match.get('port') in (8000, 80, 8080):
            management_port = match['port']
            break

    device_type = 'IP Camera'
    signature_text = ' '.join(match.get('signature', '') for match in ([rtsp_match] + http_matches) if match)
    if 'nvr' in signature_text.lower():
        device_type = 'NVR / Camera'

    serial_no, firmware_version = _extract_device_metadata(http_matches)
    if not serial_no:
        serial_no = vendor or 'Unknown'
    if not firmware_version:
        firmware_version = '—'

    return {
        'ip': ip_str,
        'port': preferred_port,
        'management_port': management_port,
        'channels': 1,
        'protocol': _protocol_label(vendor),
        'subnet_mask': subnet_mask,
        'mac_address': _lookup_mac_address(ip_str) or '—',
        'serial_no': serial_no,
        'firmware_version': firmware_version,
        'status': 'Online',
        'type': device_type,
        'model': vendor,
        'make': vendor,
    }


@app.route('/api/scan_network', methods=['GET'])
def api_scan_network():
    """Scan the camera subnet and return only validated camera/NVR hosts."""
    requested_subnet = request.args.get('subnet', '').strip()
    subnet = _resolve_camera_scan_subnet(requested_subnet)
    if requested_subnet == 'detect_only':
        return jsonify({'status': 'success', 'subnet': subnet, 'timestamp': datetime.now().isoformat()})

    try:
        network = ipaddress.ip_network(subnet, strict=False)
        subnet_mask = _subnet_mask_from_cidr(subnet)
        found = []
        with ThreadPoolExecutor(max_workers=48) as ex:
            futures = {
                ex.submit(_probe_camera_host, str(ip), subnet_mask): str(ip)
                for ip in network.hosts()
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    found.append(result)
        found.sort(key=lambda d: [int(x) for x in d['ip'].split('.')])
        return jsonify({'status': 'success', 'cameras': found,
                        'subnet': subnet, 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f'Network scan error: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _build_multilingual_prompt(language, mode, request_text):
    lang = (language or 'English').strip()
    if lang.lower() == 'kannada':
        lang = 'Kannada (ಕನ್ನಡ script)'
    mode_txt = 'voice+text' if mode == 'voice' else 'text'
    user_req = (request_text or '').strip()
    if not user_req:
        user_req = "Analyze the incident snapshot, provide regular detection commentary, root cause analysis, and specific corrective actions."
    return (
        f"You are an incident investigator assistant. Respond ONLY in {lang}. "
        f"Mode: {mode_txt}. Carefully analyze only what is visible in the image. "
        f"Do not invent facts. Do not output system/user/assistant tags. "
        f"Return exactly four markdown sections with these headings only:\n"
        f"## Incident Snapshot Observations\n"
        f"## Regular Detection Commentary\n"
        f"## Root Cause Analysis\n"
        f"## Corrective Action Recommendations\n"
        f"User request: {user_req}"
    )


def _build_analysis_prompt(language, user_req, odd_context='', alert_context=''):
    lang = (language or 'English').strip()
    req = (user_req or 'Analyze this alert snapshot and provide concise operational insights.').strip()
    odd = (odd_context or '').strip()
    alert_ctx = (alert_context or '').strip()
    odd_part = f"\nOperational Design Domain (ODD) reference:\n{odd}\n" if odd else ""
    alert_part = f"\nALERT FOCUS CONTEXT (highest priority):\n{alert_ctx}\n" if alert_ctx else ""
    return (
        f"Respond ONLY in {lang}. "
        f"Carefully analyze the provided alert snapshot, focusing on all observable details. "
        f"Prioritize the ALERT FOCUS CONTEXT and explain this specific anomaly event, not generic scene description. "
        f"If ALERT FOCUS CONTEXT conflicts with visible evidence, explicitly state uncertainty but still provide anomaly-focused corrective actions. "
        f"Do NOT produce broad facility/industry descriptions unless they directly explain the anomaly trigger. "
        f"Commentary must mention the anomaly trigger explicitly (from ALERT FOCUS CONTEXT). "
        f"Root cause must explain why that specific alert happened. "
        f"Recommendations must be concrete, immediate, and operationally actionable for this anomaly (at least 2 actions). "
        f"Recommend a clear and specific next action that directly follows from what is visible, chat interaction, avoiding any assumptions or invented elements."
        f"{alert_part}"
        f"{odd_part}"
        f"Return strict JSON with keys exactly:\n"
        f"commentary, root_cause, recommendations, next_action, odd_deviations.\n"
        f"Each value must be a short paragraph in {lang}. "
        f"No markdown. No additional keys. User request: {req}"
    )


def _build_chat_prompt(language, user_text, odd_context=''):
    lang = (language or 'English').strip()
    q = (user_text or '').strip()
    odd = (odd_context or '').strip()
    odd_part = f"\nODD reference (optional): {odd}\n" if odd else ""
    return (
        f"Respond ONLY in {lang}. "
        f"You are an incident chatbot and must answer only the user's question about this snapshot.\n"
        f"Rules:\n"
        f"- Stay on the asked question only.\n"
        f"- Use only what is clearly visible in the image.\n"
        f"- If detail is unclear, say it is not clearly visible.\n"
        f"- Do not invent objects, colors, counts, or events.\n"
        f"- Do not output markdown headings, JSON, or role tags.\n"
        f"{odd_part}"
        f"User question: {q}"
    )


def _build_chat_facts_prompt():
    """Extract stable visual facts first, then answer questions from facts."""
    return (
        "Analyze the image and return strict JSON only with keys:\n"
        "people_count, chair_count, visible_person_clothing_colors, notable_objects.\n"
        "Rules:\n"
        "- people_count and chair_count must be integers.\n"
        "- visible_person_clothing_colors must be a short array of color words that are clearly visible on people only.\n"
        "- notable_objects must be a short array (e.g., turnstile, gate, desk, chair).\n"
        "- Do not infer hidden or unclear details.\n"
        "- No markdown, no extra keys, no prose."
    )


def _extract_json_block(text):
    if not text:
        return None
    s = text.find('{')
    e = text.rfind('}')
    if s < 0 or e < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e+1])
    except Exception:
        return None


def _extract_jsonish_value(text, key):
    """Best-effort extractor for malformed JSON-like model output."""
    if not text:
        return ""
    token = f'"{key}"'
    i = text.find(token)
    if i < 0:
        return ""
    c = text.find(':', i + len(token))
    if c < 0:
        return ""
    j = c + 1
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    if j >= len(text):
        return ""
    if text[j] == '"':
        j += 1
        out = []
        esc = False
        while j < len(text):
            ch = text[j]
            if esc:
                out.append(ch)
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                break
            else:
                out.append(ch)
            j += 1
        return ''.join(out).strip()
    # Unquoted fallback: read until comma or close brace.
    k = j
    while k < len(text) and text[k] not in ",}":
        k += 1
    return text[j:k].strip().strip('"').strip()


def _extract_jsonish_int(text, key, default=-1):
    v = _extract_jsonish_value(text, key)
    if not v:
        return default
    num = ''.join(ch for ch in v if (ch.isdigit() or ch == '-'))
    try:
        return int(num)
    except Exception:
        return default


def _derive_recommendations_from_alert_context(alert_context):
    """Build concrete corrective actions from alert metadata labels."""
    ctx = (alert_context or "").strip()
    if not ctx:
        return ""

    parts = []
    for key in ("new_alert_labels", "anomaly_labels", "alert_label"):
        m = re.search(rf"\b{key}=([^|]+)", ctx, flags=re.IGNORECASE)
        if m:
            parts.append(m.group(1).strip())
    blob = " | ".join(parts) if parts else ctx

    labels = []
    for raw in blob.split(","):
        t = (raw or "").strip().strip(".")
        if not t:
            continue
        t = re.sub(r"^\s*anomaly\s*:\s*", "", t, flags=re.IGNORECASE).strip()
        if t:
            labels.append(t)
    # Deduplicate while preserving order.
    uniq = []
    seen = set()
    for lb in labels:
        k = lb.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(lb)
    labels = uniq

    actions = []
    lower_blob = " ".join(lb.lower() for lb in labels) if labels else blob.lower()

    if any(x in lower_blob for x in ("ppe", "hard hat", "lab coat", "helmet", "boots", "vest")):
        actions.append("Stop the affected task, enforce required PPE use immediately, and resume only after visual PPE compliance verification.")
    if any(x in lower_blob for x in ("dropped box", "boxes", "box on floor", "floor")):
        actions.append("Clear loose boxes from walking/working paths, place them in designated storage, and mark the zone safe before restart.")
    if "ladder" in lower_blob:
        actions.append("Re-check ladder setup and usage controls (stability, spotter, three-point contact) before allowing continued work at height.")
    if any(x in lower_blob for x in ("aisle", "obstruction", "blocked path")):
        actions.append("Remove aisle obstructions immediately and re-open the marked access route to maintain emergency and material-flow safety.")
    if "fire" in lower_blob:
        actions.append("Activate fire response protocol now: isolate area, verify extinguisher readiness, and escalate to emergency response if flame persists.")
    if "smoke" in lower_blob:
        actions.append("Treat smoke as active hazard: isolate equipment source, improve ventilation, and run immediate thermal/electrical inspection.")

    if not actions:
        lbl = ", ".join(labels) if labels else "the detected anomaly"
        actions.append(f"Verify and contain {lbl} at the exact location shown in the snapshot.")
        actions.append("Apply the relevant site safety checklist and record corrective closure evidence before normal operations continue.")

    return "Immediate corrective actions: " + " ".join(actions[:3])


def _clean_qwen_text(txt):
    """Remove prompt echo / chat role tags / degenerate punctuation runs."""
    if not txt:
        return ""
    t = txt.strip()
    lower = t.lower()
    # Strip leading role scaffolding if present.
    for marker in ("assistant\n", "assistant:", "system\n", "system:", "user\n", "user:"):
        if lower.startswith(marker):
            t = t[len(marker):].strip()
            lower = t.lower()
    # Remove inline role tokens if model echoed conversation template.
    for token in ("\nsystem\n", "\nuser\n", "\nassistant\n", "system\nYou are a helpful assistant."):
        t = t.replace(token, "\n")
    # Detect degenerate punctuation spam.
    compact = t.replace("\n", "").replace(" ", "")
    if len(compact) >= 40 and len(set(compact)) <= 2 and any(ch in compact for ch in ("!", "?", ".", "-")):
        return ""
    return t.strip()


def _extract_section(md_text, heading, fallback=""):
    if not md_text:
        return fallback
    tag = f"## {heading}"
    i = md_text.find(tag)
    if i < 0:
        return fallback
    j = md_text.find("\n## ", i + len(tag))
    if j < 0:
        j = len(md_text)
    body = md_text[i + len(tag):j].strip()
    return body or fallback


def _qwen_infer_on_image(snapshot_path, language='English', mode='text', user_text='', odd_context='', alert_context='', task='analysis', _allow_cpu_retry=True):
    proc, model = _get_qwen25vl()
    if proc is None or model is None:
        return {
            'incident_snapshot': 'Qwen2.5-VL is not available in local cache.',
            'regular_commentary': 'No model response available.',
            'root_cause': 'Unavailable',
            'recommendations': 'Configure local Qwen2.5-VL cache and retry.',
            'full_response': 'Model unavailable.'
        }
    try:
        def _gen_text(text_in, image_obj=None, max_new_tokens=220):
            ins = None
            if image_obj is not None and hasattr(proc, 'apply_chat_template'):
                # Match the working Colab-style flow for Qwen image-text models.
                try:
                    msgs = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_obj},
                            {"type": "text", "text": text_in},
                        ],
                    }]
                    ins = proc.apply_chat_template(
                        msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except Exception:
                    ins = None
            if ins is None:
                if image_obj is not None:
                    ins = proc(text=text_in, images=image_obj, return_tensors='pt')
                else:
                    if hasattr(proc, 'apply_chat_template'):
                        try:
                            txt_msgs = [{"role": "user", "content": [{"type": "text", "text": text_in}]}]
                            ins = proc.apply_chat_template(
                                txt_msgs,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_dict=True,
                                return_tensors="pt",
                            )
                        except Exception:
                            ins = proc(text=text_in, return_tensors='pt')
                    else:
                        ins = proc(text=text_in, return_tensors='pt')
            if torch is not None and hasattr(model, 'parameters'):
                dvc = next(model.parameters()).device
                ins = {k: v.to(dvc) if hasattr(v, 'to') else v for k, v in ins.items()}
            with _qwen25vl_infer_lock:
                if torch is not None:
                    with torch.inference_mode():
                        oids = model.generate(
                            **ins,
                            max_new_tokens=max_new_tokens,
                            num_beams=int(os.environ.get('VISION_QWEN_NUM_BEAMS', '1')),
                            do_sample=False,
                            repetition_penalty=1.1,
                        )
                else:
                    oids = model.generate(
                        **ins,
                        max_new_tokens=max_new_tokens,
                        num_beams=int(os.environ.get('VISION_QWEN_NUM_BEAMS', '1')),
                        do_sample=False,
                        repetition_penalty=1.1,
                    )
            try:
                in_len = int(ins["input_ids"].shape[-1]) if "input_ids" in ins else 0
                gen_only = oids[:, in_len:] if in_len > 0 else oids
                out = proc.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
            except Exception:
                out = proc.batch_decode(oids, skip_special_tokens=True)[0].strip()
            return _clean_qwen_text(out)

        def _sections_ok(text):
            if not text or len(text) < 60:
                return False
            if text.count('!') > 60:
                return False
            # Accept plain useful answers too (Qwen3.5 often returns concise text without headings).
            if "## " not in text and len(text.strip()) >= 40:
                return True
            need = (
                "## Incident Snapshot Observations",
                "## Regular Detection Commentary",
                "## Root Cause Analysis",
                "## Corrective Action Recommendations",
            )
            return all(h in text for h in need)

        img = Image.open(snapshot_path).convert('RGB')
        # Hard cap image side length to control visual token explosion and VRAM spikes.
        max_side = max(256, int(VISION_QWEN_IMAGE_MAX_SIDE))
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
        if task == 'chat':
            prompt = _build_chat_prompt(language, user_text, odd_context=odd_context)
        else:
            prompt = _build_analysis_prompt(language, user_text, odd_context=odd_context, alert_context=alert_context)
        max_default = int(os.environ.get('VISION_QWEN_MAX_NEW_TOKENS', '220'))
        chat_max = int(os.environ.get('VISION_QWEN_CHAT_MAX_NEW_TOKENS', '220'))
        max_new = chat_max if task == 'chat' else max_default
        txt = _gen_text(prompt, image_obj=img, max_new_tokens=max_new)

        def _chat_json_answer(raw):
            obj = _extract_json_block(raw)
            if not obj:
                return ''
            a = obj.get('answer')
            if isinstance(a, str) and len(a.strip()) >= 4:
                return a.strip()
            return ''

        if task == 'chat':
            ans = _chat_json_answer(txt)
            if not ans:
                # If model still emitted JSON-like output, extract answer field best-effort.
                maybe_ans = _extract_jsonish_value(txt, 'answer')
                if maybe_ans:
                    ans = maybe_ans
            if not ans:
                ans = _clean_qwen_text(txt).strip()
            # Fallback pass in English only when answer is empty/very short.
            if not ans or len(ans) < 2:
                en_raw = _gen_text(_build_chat_prompt('English', user_text, odd_context=odd_context), image_obj=img, max_new_tokens=max_new)
                en_ans = _chat_json_answer(en_raw) or _extract_jsonish_value(en_raw, 'answer') or _clean_qwen_text(en_raw).strip()
                lang_low = (language or 'English').strip().lower()
                if lang_low not in ('english', 'en') and en_ans:
                    tr = _gen_text(
                        f"Translate the following to {language}. Output ONLY the translation. "
                        f"Preserve numbers and factual claims exactly. No JSON, no headings, no extra sentences.\n\n{en_ans}",
                        image_obj=None,
                        max_new_tokens=min(max_new, 300),
                    )
                    ans = (tr or '').strip() or en_ans
                else:
                    ans = en_ans
            if not ans:
                ans = (
                    f"Could not produce a reliable answer in {language} for this question. "
                    f"Please rephrase or try English."
                )
            return {
                'incident_snapshot': '',
                'regular_commentary': '',
                'root_cause': '',
                'recommendations': ans,
                'full_response': ans,
            }

        # If output is malformed for target language, do reliable English first then translate.
        if not _sections_ok(txt):
            en_prompt = _build_analysis_prompt('English', user_text, odd_context=odd_context, alert_context=alert_context)
            en_txt = _gen_text(en_prompt, image_obj=img, max_new_tokens=max_new)
            if _sections_ok(en_txt):
                if (language or 'English').strip().lower() != 'english':
                    tr_prompt = (
                        f"Translate the following answer to {language}. "
                        f"Do not mix languages. Keep structure unchanged.\n\n{en_txt}"
                    )
                    tr_txt = _gen_text(tr_prompt, image_obj=None, max_new_tokens=max_new)
                    txt = tr_txt if _sections_ok(tr_txt) else en_txt
                else:
                    txt = en_txt

        if not txt:
            txt = (
                f"## Incident Snapshot Observations\nNo reliable text could be generated in {language}.\n\n"
                f"## Regular Detection Commentary\nModel output was empty.\n\n"
                f"## Root Cause Analysis\nInsufficient reliable output.\n\n"
                f"## Corrective Action Recommendations\nRetry with a shorter prompt or switch language."
            )

        # Analysis: prefer strict JSON to avoid duplicated sections and mixed outputs.
        if task != 'chat':
            obj = _extract_json_block(txt)
            if obj:
                commentary_sec = str(obj.get('commentary') or '').strip()
                root_sec = str(obj.get('root_cause') or '').strip()
                reco_txt = str(obj.get('recommendations') or '').strip()
                next_act = str(obj.get('next_action') or '').strip()
                odd_dev = str(obj.get('odd_deviations') or '').strip()
                if next_act:
                    reco_txt = (reco_txt + ("\n\nNext Action: " + next_act if reco_txt else next_act)).strip()
                if odd_dev:
                    reco_txt = (reco_txt + ("\n\nODD Deviations: " + odd_dev if reco_txt else odd_dev)).strip()
                snapshot_sec = _extract_section(txt, "Incident Snapshot Observations", commentary_sec or txt)
                if not commentary_sec:
                    commentary_sec = snapshot_sec or txt
                if not root_sec:
                    root_sec = commentary_sec or txt
                if not reco_txt:
                    reco_txt = root_sec or txt
                return {
                    'incident_snapshot': snapshot_sec,
                    'regular_commentary': commentary_sec,
                    'root_cause': root_sec,
                    'recommendations': reco_txt,
                    'full_response': txt
                }
            # Fallback for partially malformed JSON output.
            commentary_sec = _extract_jsonish_value(txt, 'commentary')
            root_sec = _extract_jsonish_value(txt, 'root_cause')
            reco_txt = _extract_jsonish_value(txt, 'recommendations')
            next_act = _extract_jsonish_value(txt, 'next_action')
            odd_dev = _extract_jsonish_value(txt, 'odd_deviations')
            if next_act:
                reco_txt = (reco_txt + ("\n\nNext Action: " + next_act if reco_txt else next_act)).strip()
            if odd_dev:
                reco_txt = (reco_txt + ("\n\nODD Deviations: " + odd_dev if reco_txt else odd_dev)).strip()
            if commentary_sec or root_sec or reco_txt:
                if not commentary_sec:
                    commentary_sec = (root_sec or reco_txt or txt).strip()
                if not root_sec:
                    root_sec = (commentary_sec or reco_txt or txt).strip()
                if not reco_txt:
                    # Never return empty recommendations; derive from available anomaly-focused sections.
                    if next_act:
                        reco_txt = next_act
                    elif alert_context:
                        reco_txt = _derive_recommendations_from_alert_context(alert_context)
                    else:
                        reco_txt = (root_sec or commentary_sec or txt).strip()
                return {
                    'incident_snapshot': commentary_sec or root_sec or reco_txt,
                    'regular_commentary': commentary_sec,
                    'root_cause': root_sec,
                    'recommendations': reco_txt,
                    'full_response': txt
                }

        snapshot_sec = _extract_section(txt, "Incident Snapshot Observations", "")
        commentary_sec = _extract_section(txt, "Regular Detection Commentary", "")
        root_sec = _extract_section(txt, "Root Cause Analysis", "")
        reco_sec = _extract_section(txt, "Corrective Action Recommendations", "")

        # If model returned plain text (no headings), still populate all sections with useful output.
        if not snapshot_sec and not commentary_sec and not root_sec and not reco_sec:
            snapshot_sec = txt
            commentary_sec = txt
            root_sec = txt
            reco_sec = txt
        else:
            if not snapshot_sec:
                snapshot_sec = commentary_sec or txt
            if not commentary_sec:
                commentary_sec = snapshot_sec or txt
            if not root_sec:
                root_sec = commentary_sec or txt
            if not reco_sec:
                reco_sec = root_sec or txt
        return {
            'incident_snapshot': snapshot_sec,
            'regular_commentary': commentary_sec,
            'root_cause': root_sec,
            'recommendations': reco_sec,
            'full_response': txt
        }
    except Exception as e:
        msg = f"Qwen vision inference error: {e}"
        logger.error(msg)
        # GPU can occasionally hit device-side asserts; retry once on CPU for reliability.
        emsg = str(e).lower()
        if _allow_cpu_retry and ('cuda error' in emsg or 'device-side assert' in emsg):
            logger.warning("Qwen vision GPU failure; retrying on CPU for this request.")
            prev_dev = os.environ.get('VISION_QWEN_DEVICE')
            try:
                os.environ['VISION_QWEN_DEVICE'] = 'cpu'
                _unload_qwen25vl()
                return _qwen_infer_on_image(
                    snapshot_path, language=language, mode=mode, user_text=user_text, odd_context=odd_context, alert_context=alert_context, task=task, _allow_cpu_retry=False
                )
            finally:
                if prev_dev is None:
                    os.environ.pop('VISION_QWEN_DEVICE', None)
                else:
                    os.environ['VISION_QWEN_DEVICE'] = prev_dev
        return {
            'incident_snapshot': msg,
            'regular_commentary': msg,
            'root_cause': msg,
            'recommendations': msg,
            'full_response': msg
        }
    finally:
        maybe_unload_qwen_after_incident()
        _torch_free_cuda_memory()


# ---------- API: Cameras ----------

@app.route('/api/cameras', methods=['GET'])
def api_list_cameras():
    # Any camera can have multiple detection types enabled at once (headcount + entryexit + ppe etc.).
    cameras = []
    conn = get_db()
    rows = conn.execute('SELECT * FROM cameras ORDER BY added_time DESC').fetchall()
    conn.close()
    for row in rows:
        cid = row['camera_id']
        reader = camera_readers.get(cid)
        cameras.append({
            'camera_id': cid,
            'name': row['name'],
            'rtsp_url': row['rtsp_url'],
            'latitude': row['latitude'] if 'latitude' in row.keys() else None,
            'longitude': row['longitude'] if 'longitude' in row.keys() else None,
            'status': reader.status if reader else 'stopped',
            'headcount_active': cid in headcount_procs,
            'entryexit_active': cid in entryexit_procs,
            'flapgate_active': cid in flapgate_procs,
            'ppe_active': cid in ppe_procs,
            'fire_smoke_active': cid in fire_smoke_procs,
            'anpr_active': cid in anpr_procs,
            'beta_active': cid in beta_procs,
        })
    return jsonify(cameras)


@app.route('/api/cameras', methods=['POST'])
def api_add_camera():
    data = request.json
    rtsp_url = data.get('rtsp_url', '').strip()
    name = data.get('name', '').strip()
    if not rtsp_url:
        return jsonify({'error': 'RTSP URL is required'}), 400

    camera_id = data.get('camera_id', '').strip()
    if not camera_id:
        camera_id = f"cam_{len(camera_readers) + 1}_{int(time.time()) % 10000}"
    if not name:
        name = camera_id

    if camera_id in camera_readers:
        return jsonify({'error': 'Camera ID already exists'}), 400

    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO cameras (camera_id, name, rtsp_url, status) VALUES (?,?,?,?)',
                 (camera_id, name, rtsp_url, 'active'))
    conn.commit()
    conn.close()

    reader = CameraReader(camera_id, rtsp_url)
    camera_readers[camera_id] = reader
    reader.start()

    return jsonify({'success': True, 'camera_id': camera_id})


@app.route('/api/cameras/<camera_id>', methods=['PUT'])
def api_update_camera(camera_id):
    data = request.json or {}
    rtsp_url = (data.get('rtsp_url') or '').strip()
    name = (data.get('name') or '').strip()
    if not rtsp_url:
        return jsonify({'error': 'RTSP URL is required'}), 400

    conn = get_db()
    row = conn.execute('SELECT camera_id FROM cameras WHERE camera_id=?', (camera_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Camera not found'}), 404

    if not name:
        existing = conn.execute('SELECT name FROM cameras WHERE camera_id=?', (camera_id,)).fetchone()
        name = existing['name'] if existing else camera_id

    conn.execute(
        'UPDATE cameras SET name=?, rtsp_url=?, status=? WHERE camera_id=?',
        (name, rtsp_url, 'active', camera_id),
    )
    conn.commit()
    conn.close()

    if camera_id in camera_readers:
        camera_readers[camera_id].stop()
        del camera_readers[camera_id]
    reader = CameraReader(camera_id, rtsp_url)
    camera_readers[camera_id] = reader
    reader.start()

    return jsonify({'success': True, 'camera_id': camera_id})


@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def api_remove_camera(camera_id):
    if camera_id in headcount_procs:
        headcount_procs[camera_id].stop()
        del headcount_procs[camera_id]
    if camera_id in entryexit_procs:
        entryexit_procs[camera_id].stop()
        del entryexit_procs[camera_id]
    if camera_id in flapgate_procs:
        flapgate_procs[camera_id].stop()
        del flapgate_procs[camera_id]
    if camera_id in ppe_procs:
        ppe_procs[camera_id].stop()
        del ppe_procs[camera_id]
    if camera_id in fire_smoke_procs:
        fire_smoke_procs[camera_id].stop()
        del fire_smoke_procs[camera_id]
    if camera_id in anpr_procs:
        anpr_procs[camera_id].stop()
        del anpr_procs[camera_id]
    if camera_id in beta_procs:
        beta_procs[camera_id].stop()
        del beta_procs[camera_id]
        _maybe_unload_beta_vlms_if_idle()
    if camera_id in camera_readers:
        camera_readers[camera_id].stop()
        del camera_readers[camera_id]

    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=?', (camera_id,))
    conn.execute('DELETE FROM cameras WHERE camera_id=?', (camera_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/cameras/<camera_id>/snapshot')
def api_camera_snapshot(camera_id):
    reader = camera_readers.get(camera_id)
    if not reader:
        return jsonify({'error': 'Camera not found'}), 404
    frame = reader.get_frame()
    if frame is None:
        return jsonify({'error': 'No frame available yet'}), 503
    _, buf = cv2.imencode('.jpg', frame)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/feed_snapshot/<camera_id>')
def api_feed_snapshot(camera_id):
    """Single JPEG of the best available feed. When multiple detection types are active
    for this camera (e.g. headcount + PPE), composites all overlays on one frame."""
    active = []
    if camera_id in headcount_procs:
        active.append(('headcount', headcount_procs[camera_id]))
    if camera_id in entryexit_procs:
        active.append(('entryexit', entryexit_procs[camera_id]))
    if camera_id in flapgate_procs:
        active.append(('flapgate', flapgate_procs[camera_id]))
    if camera_id in ppe_procs:
        active.append(('ppe', ppe_procs[camera_id]))
    if camera_id in fire_smoke_procs:
        active.append(('fire_smoke', fire_smoke_procs[camera_id]))
    if camera_id in anpr_procs:
        active.append(('anpr', anpr_procs[camera_id]))
    if camera_id in beta_procs:
        active.append(('beta', beta_procs[camera_id]))
    if camera_id in fr_procs:
        active.append(('fr', fr_procs[camera_id]))

    frame = None
    if len(active) > 1:
        # Multiple detections: composite all on raw frame
        reader = camera_readers.get(camera_id)
        if reader:
            frame = reader.get_frame()
        if frame is not None:
            frame = frame.copy()
            for _name, proc in active:
                for (bbox, label, color) in proc.get_last_detections():
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    if _name in ('ppe', 'beta', 'fire_smoke', 'anpr'):
                        _draw_premium_ppe_box(frame, x1, y1, x2, y2, label, color)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_y = max(y1 - 6, 14)
                        cv2.putText(frame, label, (x1, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    elif len(active) == 1:
        frame = active[0][1].get_annotated_frame()
    if frame is None:
        reader = camera_readers.get(camera_id)
        if reader:
            frame = reader.get_frame()
    if frame is None:
        return '', 204
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    resp = Response(buf.tobytes(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


def _alert_event_to_api(r):
    try:
        meta = json.loads(r['meta_json']) if r['meta_json'] else {}
    except Exception:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    cam_name = r['camera_name'] if 'camera_name' in r.keys() else r['camera_id']
    status = (r['status'] if 'status' in r.keys() and r['status'] else 'open') or 'open'
    assigned = (r['assigned_to'] if 'assigned_to' in r.keys() else '') or ''
    eng_id = (r['assigned_engineer_id'] if 'assigned_engineer_id' in r.keys() else '') or ''
    eng = _engineer_by_id(eng_id) if eng_id else None
    return {
        'id': int(r['id']),
        'camera_id': r['camera_id'],
        'camera_name': cam_name or r['camera_id'],
        'detection_type': r['detection_type'],
        'alert_label': r['alert_label'],
        'severity': r['severity'] or 'medium',
        'status': status,
        'assigned_to': assigned,
        'assigned_engineer_id': eng_id,
        'engineer': eng,
        'created_time': r['created_time'],
        'has_snapshot': bool(r['snapshot_path']),
        'snapshot_url': f"/api/alerts/{int(r['id'])}/snapshot" if r['snapshot_path'] else None,
        'meta': meta,
    }


@app.route('/api/alerts/analytics', methods=['GET'])
def api_alerts_analytics():
    days = max(1, min(int(request.args.get('days', 30)), 90))
    conn = get_db()
    freq_rows = conn.execute(
        '''SELECT a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name, COUNT(*) AS cnt
           FROM alert_events a
           LEFT JOIN cameras c ON c.camera_id = a.camera_id
           WHERE a.created_time >= datetime('now', ?)
           GROUP BY a.camera_id
           ORDER BY cnt DESC''',
        (f'-{days} days',)
    ).fetchall()
    trend_rows = conn.execute(
        '''SELECT date(a.created_time) AS day, a.camera_id,
                  COALESCE(c.name, a.camera_id) AS camera_name, COUNT(*) AS cnt
           FROM alert_events a
           LEFT JOIN cameras c ON c.camera_id = a.camera_id
           WHERE a.created_time >= datetime('now', ?)
           GROUP BY day, a.camera_id
           ORDER BY day ASC''',
        (f'-{days} days',)
    ).fetchall()
    conn.close()
    frequency = [
        {'camera_id': r['camera_id'], 'camera_name': r['camera_name'], 'count': int(r['cnt'])}
        for r in freq_rows
    ]
    trend_map = {}
    camera_names = {}
    for r in trend_rows:
        day = r['day']
        cid = r['camera_id']
        camera_names[cid] = r['camera_name']
        trend_map.setdefault(day, {})[cid] = int(r['cnt'])
    days_sorted = sorted(trend_map.keys())
    trend = {
        'labels': days_sorted,
        'cameras': [
            {
                'camera_id': cid,
                'camera_name': camera_names.get(cid, cid),
                'counts': [trend_map.get(d, {}).get(cid, 0) for d in days_sorted],
            }
            for cid in camera_names
        ],
    }
    return jsonify({'frequency': frequency, 'trend': trend, 'days': days})


@app.route('/api/alerts', methods=['GET'])
def api_alerts_list():
    limit = int(request.args.get('limit', 200))
    detection_type = (request.args.get('detection_type') or '').strip()
    conn = get_db()
    base_sql = (
        'SELECT a.id, a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name, '
        'a.detection_type, a.alert_label, a.severity, a.snapshot_path, a.meta_json, '
        'a.created_time, a.status, a.assigned_to, a.assigned_engineer_id '
        'FROM alert_events a LEFT JOIN cameras c ON c.camera_id = a.camera_id '
    )
    if detection_type:
        rows = conn.execute(
            base_sql + 'WHERE a.detection_type=? ORDER BY a.id DESC LIMIT ?',
            (detection_type, max(1, min(limit, 500)))
        ).fetchall()
    else:
        rows = conn.execute(
            base_sql + 'ORDER BY a.id DESC LIMIT ?',
            (max(1, min(limit, 500)),)
        ).fetchall()
    conn.close()
    out = [_alert_event_to_api(r) for r in rows]
    return jsonify({'alerts': out, 'qwen_ready': _qwen25vl_ready()})


@app.route('/api/alerts/<int:alert_id>/ack', methods=['POST'])
def api_alerts_ack(alert_id):
    conn = get_db()
    row = conn.execute('SELECT id FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Alert not found'}), 404
    conn.execute(
        "UPDATE alert_events SET status='acknowledged' WHERE id=?",
        (alert_id,)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'status': 'acknowledged'})


@app.route('/api/alerts/<int:alert_id>/assign', methods=['POST'])
def api_alerts_assign(alert_id):
    data = request.get_json(silent=True) or {}
    engineer_id = str(data.get('engineer_id') or '').strip()
    eng = _engineer_by_id(engineer_id)
    if not eng:
        return jsonify({'error': 'Engineer not found'}), 400
    conn = get_db()
    row = conn.execute('SELECT id FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Alert not found'}), 404
    conn.execute(
        "UPDATE alert_events SET status='assigned', assigned_engineer_id=?, assigned_to=? WHERE id=?",
        (eng['id'], eng['name'], alert_id)
    )
    conn.commit()
    alert_row = conn.execute(
        '''SELECT a.id, a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name,
                  a.detection_type, a.alert_label, a.severity, a.snapshot_path, a.meta_json,
                  a.created_time, a.status, a.assigned_to, a.assigned_engineer_id
           FROM alert_events a LEFT JOIN cameras c ON c.camera_id = a.camera_id WHERE a.id=?''',
        (alert_id,)
    ).fetchone()
    conn.close()
    if eng.get('email'):
        try:
            cfg = _get_email_alert_settings()
            if cfg.get('enabled'):
                subject = _alert_assign_email_subject(alert_row)
                body = _alert_assign_email_body(alert_row, eng['name'], alert_id)

                def _worker():
                    try:
                        _send_smtp_email(
                            cfg.get('sender_email'), eng['email'],
                            subject, body, cfg.get('app_password'),
                            cfg.get('smtp_host') or 'smtp.gmail.com',
                            cfg.get('smtp_port') or 587,
                        )
                    except Exception as e:
                        logger.error(f"Engineer assign email failed: {e}")
                threading.Thread(target=_worker, daemon=True).start()
        except Exception as e:
            logger.error(f"Engineer assign email setup failed: {e}")
    return jsonify({'ok': True, 'status': 'assigned', 'engineer': eng, 'alert': _alert_event_to_api(alert_row)})


@app.route('/api/engineers', methods=['GET'])
def api_engineers_list():
    return jsonify({'engineers': _get_engineers_list()})


@app.route('/api/assigned-engineers', methods=['GET'])
def api_assigned_engineers():
    engineers = {e['id']: {**e, 'alerts': []} for e in _get_engineers_list()}
    conn = get_db()
    rows = conn.execute(
        '''SELECT a.id, a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name,
                  a.detection_type, a.alert_label, a.severity, a.snapshot_path, a.meta_json,
                  a.created_time, a.status, a.assigned_to, a.assigned_engineer_id
           FROM alert_events a
           LEFT JOIN cameras c ON c.camera_id = a.camera_id
           WHERE a.assigned_engineer_id != '' AND lower(COALESCE(a.status, 'open')) != 'resolved'
           ORDER BY a.created_time DESC'''
    ).fetchall()
    conn.close()
    unassigned_bucket = []
    for r in rows:
        item = _alert_event_to_api(r)
        eid = item.get('assigned_engineer_id') or ''
        if eid in engineers:
            engineers[eid]['alerts'].append(item)
        elif item.get('assigned_to'):
            unassigned_bucket.append(item)
    out = [v for v in engineers.values() if v['alerts']]
    return jsonify({
        'engineers': out,
        'total_assigned_alerts': sum(len(e['alerts']) for e in out),
        'orphan_alerts': unassigned_bucket,
    })


def _device_health_label(health):
    return {
        'normal': 'ONLINE',
        'warning': 'WARNING',
        'critical': 'CRITICAL',
        'offline': 'OFFLINE',
    }.get(health or 'offline', 'OFFLINE')


def _device_stream_quality(reader):
    if not reader:
        return {'label': 'POOR', 'feed': 'Feed Lost', 'class': 'stop'}
    st = (reader.status or '').lower()
    if st == 'active':
        return {'label': 'OK', 'feed': 'Feed Active', 'class': 'ok'}
    if st == 'reconnecting':
        return {'label': 'LOW', 'feed': 'Feed Degraded', 'class': 'low'}
    if st in ('initializing',):
        return {'label': 'LOW', 'feed': 'Connecting', 'class': 'low'}
    return {'label': 'POOR', 'feed': 'Feed Lost', 'class': 'stop'}


def _device_signal_estimate(reader):
    if not reader:
        return {'pct': 0, 'dbm': 'No Signal', 'label': '0% — No Signal'}
    st = (reader.status or '').lower()
    if st == 'active':
        return {'pct': 82, 'dbm': '-76dBm', 'label': '82% — -76dBm'}
    if st == 'reconnecting':
        return {'pct': 45, 'dbm': '-88dBm', 'label': '45% — -88dBm'}
    if st == 'initializing':
        return {'pct': 30, 'dbm': '-92dBm', 'label': '30% — -92dBm'}
    return {'pct': 0, 'dbm': 'No Signal', 'label': '0% — No Signal'}


def _device_recording_status():
    local = _system_kv_get('local_settings', _local_settings_default())
    rec = local.get('record_files') if isinstance(local.get('record_files'), dict) else {}
    path = (rec.get('record_path') or '').strip()
    if path:
        return {'label': 'REC', 'text': 'Recording Active', 'class': 'ok'}
    return {'label': 'STOP', 'text': 'Recording Stopped', 'class': 'stop'}


def _device_detection_summary(camera_id):
    """Primary live detection readout for device list/detail."""
    candidates = []
    fs = fire_smoke_procs.get(camera_id)
    if fs:
        fc = int(fs.stats.get('fire_count') or 0)
        sc = int(fs.stats.get('smoke_count') or 0)
        total = int(fs.stats.get('total') or fc + sc)
        if total > 0:
            candidates.append(('critical', 'Fire/Smoke', f'🔥 {total}', f'Fire/Smoke — ALARM', 'fire_smoke'))
        elif fs.stats.get('status') == 'Active':
            candidates.append(('ok', 'Fire/Smoke', '✔ CLEAR', 'Smoke Detection — SAFE', 'fire_smoke'))
    fg = flapgate_procs.get(camera_id)
    if fg:
        tp = int(fg.stats.get('trespassing_total') or 0)
        if tp > 0:
            candidates.append(('critical', 'Intrusion', '🚶 HIGH', 'Intrusion — RESTRICTED ZONE', 'flapgate'))
    ppe = ppe_procs.get(camera_id)
    if ppe:
        viol = int(ppe.stats.get('violation_count') or 0)
        if viol > 0:
            candidates.append(('critical', 'PPE Violation', '⛑ VIOLATION', f'PPE Violation — {viol} alert(s)', 'ppe'))
    hc = headcount_procs.get(camera_id)
    if hc:
        cnt = int(hc.stats.get('current_count') or 0)
        candidates.append(('info', 'Head Count', f'👤 {cnt}', f'Head Count — {cnt} in frame', 'headcount'))
    ee = entryexit_procs.get(camera_id)
    if ee:
        inside = int(ee.stats.get('current_inside') or 0)
        candidates.append(('info', 'Entry/Exit', f'↔ {inside}', f'Entry/Exit — {inside} inside', 'entryexit'))
    anpr = anpr_procs.get(camera_id)
    if anpr:
        plates = int(anpr.stats.get('plates') or 0)
        candidates.append(('info', 'ANPR', f'🚗 {plates}', f'ANPR — {plates} plate(s)', 'anpr'))
    if camera_id in beta_procs:
        candidates.append(('info', 'Beta AI', 'AI', 'Beta detection active', 'beta'))

    rank = {'critical': 3, 'warning': 2, 'info': 1, 'ok': 0}
    if not candidates:
        return {
            'value': '— N/A' if camera_id not in camera_readers else '✔ CLEAR',
            'detail': 'Feed Unavailable' if camera_id not in camera_readers else 'No Detection — SAFE',
            'type': 'none',
            'severity_class': 'na' if camera_id not in camera_readers else 'ok',
            'confidence': '—',
        }
    best = max(candidates, key=lambda c: rank.get(c[0], 0))
    sev, dtype, val, detail, key = best
    return {
        'value': val,
        'detail': detail,
        'type': dtype,
        'detection_key': key,
        'severity_class': 'stop' if sev == 'critical' else ('low' if sev == 'warning' else 'ok'),
        'confidence': val.split()[-1] if '%' in val or val.replace('.', '').isdigit() else '—',
    }


def _device_active_ai_models(camera_id):
    models = []
    mapping = [
        ('headcount', headcount_procs, 'Head Count YOLO'),
        ('entryexit', entryexit_procs, 'Entry/Exit Tracking'),
        ('flapgate', flapgate_procs, 'Flap Gate Trespass'),
        ('ppe', ppe_procs, 'PPE Compliance'),
        ('fire_smoke', fire_smoke_procs, 'Fire/Smoke Detection'),
        ('anpr', anpr_procs, 'ANPR / LibreYOLO'),
        ('beta', beta_procs, 'Beta OWL / VLM'),
    ]
    for key, proc_map, label in mapping:
        if camera_id in proc_map:
            models.append({'key': key, 'name': label, 'status': proc_map[camera_id].stats.get('status', 'Active')})
    return models


def _device_uptime_label(added_time, reader):
    if not reader or (reader.status or '').lower() not in ('active', 'reconnecting', 'initializing'):
        return 'Last seen: offline'
    if not added_time:
        return 'Active: —'
    try:
        ts = str(added_time).replace('T', ' ')[:19]
        dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        delta = datetime.now() - dt
        hours = int(delta.total_seconds() // 3600)
        mins = int((delta.total_seconds() % 3600) // 60)
        if hours > 48:
            return f'Active: {hours}h {mins}m'
        return f'Active: {hours}h {mins}m ago'
    except Exception:
        return f'Active since {added_time}'


def _build_device_row(row, index, center_lat, center_lng):
    cid = row['camera_id']
    name = row['name'] or cid
    reader = camera_readers.get(cid)
    health = _camera_map_status(cid)
    status = _device_health_label(health)
    stream = _device_stream_quality(reader)
    signal = _device_signal_estimate(reader)
    recording = _device_recording_status()
    detection = _device_detection_summary(cid)
    lat = row['latitude']
    lng = row['longitude']
    if lat is None or lng is None:
        lat, lng = _default_camera_coords(cid, index, center_lat, center_lng)
    conn = get_db()
    open_alerts = conn.execute(
        '''SELECT COUNT(*) AS c FROM alert_events
           WHERE camera_id=? AND lower(COALESCE(status,'open')) NOT IN ('resolved','acknowledged')''',
        (cid,)
    ).fetchone()['c']
    conn.close()
    alert_on = health == 'critical' or int(open_alerts or 0) > 0
    ai_active = _device_active_ai_models(cid)
    return {
        'camera_id': cid,
        'name': name,
        'tag': cid,
        'status': status,
        'health': health,
        'detection': detection,
        'stream_quality': stream,
        'recording': recording,
        'signal': signal,
        'alert_output': 'ON' if alert_on else 'OFF',
        'open_alerts': int(open_alerts or 0),
        'zone': name,
        'location': {
            'latitude': float(lat),
            'longitude': float(lng),
            'label': f'{lat:.4f}, {lng:.4f}',
        },
        'uptime_label': _device_uptime_label(row['added_time'] if 'added_time' in row.keys() else None, reader),
        'stream_status': reader.status if reader else 'stopped',
        'rtsp_host': _extract_ip_from_rtsp_url(row['rtsp_url']) if 'rtsp_url' in row.keys() else '',
        'ai_models': ai_active,
        'ai_tasks': ', '.join(m['name'] for m in ai_active) or 'None',
    }


@app.route('/api/system/devices', methods=['GET'])
def api_system_devices():
    local = _system_kv_get('local_settings', _local_settings_default())
    plant = local.get('plant_map') if isinstance(local.get('plant_map'), dict) else {}
    center_lat = float(plant.get('center_lat') or 28.6139)
    center_lng = float(plant.get('center_lng') or 77.2090)
    conn = get_db()
    rows = conn.execute(
        'SELECT camera_id, name, rtsp_url, latitude, longitude, added_time, status FROM cameras ORDER BY added_time'
    ).fetchall()
    unacked = conn.execute(
        '''SELECT COUNT(*) AS c FROM alert_events
           WHERE lower(COALESCE(status,'open')) NOT IN ('resolved','acknowledged','assigned')'''
    ).fetchone()['c']
    conn.close()
    devices = [_build_device_row(r, i, center_lat, center_lng) for i, r in enumerate(rows)]
    stats = {
        'total': len(devices),
        'online': sum(1 for d in devices if d['status'] == 'ONLINE'),
        'offline': sum(1 for d in devices if d['status'] == 'OFFLINE'),
        'warning': sum(1 for d in devices if d['status'] == 'WARNING'),
        'critical': sum(1 for d in devices if d['status'] == 'CRITICAL'),
        'unacked': int(unacked or 0),
    }
    zones = sorted({d['zone'] for d in devices if d.get('zone')})
    return jsonify({'stats': stats, 'devices': devices, 'zones': zones})


@app.route('/api/system/devices/<camera_id>', methods=['GET'])
def api_system_device_detail(camera_id):
    conn = get_db()
    row = conn.execute(
        'SELECT camera_id, name, rtsp_url, latitude, longitude, added_time, status FROM cameras WHERE camera_id=?',
        (camera_id,)
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Device not found'}), 404
    alerts = conn.execute(
        '''SELECT id, detection_type, alert_label, severity, status, created_time
           FROM alert_events WHERE camera_id=? ORDER BY id DESC LIMIT 20''',
        (camera_id,)
    ).fetchall()
    configs = conn.execute(
        'SELECT detection_type, enabled, config_json FROM detection_configs WHERE camera_id=?',
        (camera_id,)
    ).fetchall()
    conn.close()
    local = _system_kv_get('local_settings', _local_settings_default())
    plant = local.get('plant_map') if isinstance(local.get('plant_map'), dict) else {}
    center_lat = float(plant.get('center_lat') or 28.6139)
    center_lng = float(plant.get('center_lng') or 77.2090)
    device = _build_device_row(row, 0, center_lat, center_lng)
    device['recent_alerts'] = [
        {
            'id': int(a['id']),
            'time': a['created_time'],
            'event': a['alert_label'] or a['detection_type'],
            'severity': (a['severity'] or 'medium').upper(),
            'status': (a['status'] or 'open').upper(),
            'detection_type': a['detection_type'],
        }
        for a in alerts
    ]
    device['configs'] = [
        {'detection_type': c['detection_type'], 'enabled': bool(c['enabled']), 'config': json.loads(c['config_json'] or '{}')}
        for c in configs
    ]
    device['snapshot_url'] = f'/api/cameras/{camera_id}/snapshot'
    return jsonify(device)


def _camera_map_status(camera_id):
    """Return plant-map health: offline, critical, warning, or normal."""
    reader = camera_readers.get(camera_id)
    if not reader or (reader.status or '').lower() not in ('active', 'reconnecting'):
        return 'offline'
    if (reader.status or '').lower() == 'reconnecting':
        return 'warning'
    sev_rank = {'critical': 3, 'high': 3, 'medium': 2, 'low': 1, 'info': 1}
    max_sev = 0
    try:
        conn = get_db()
        rows = conn.execute(
            '''SELECT severity FROM alert_events
               WHERE camera_id=? AND lower(COALESCE(status, 'open')) NOT IN ('resolved')
               AND created_time >= datetime('now', '-24 hours')''',
            (camera_id,)
        ).fetchall()
        conn.close()
        for r in rows:
            max_sev = max(max_sev, sev_rank.get((r['severity'] or 'medium').lower(), 2))
    except Exception:
        pass
    fg = flapgate_procs.get(camera_id)
    if fg and (fg.stats.get('trespassing_total') or 0) > 0:
        return 'critical'
    fs = fire_smoke_procs.get(camera_id)
    if fs and (fs.stats.get('total') or fs.stats.get('fire_count') or 0) > 0:
        return 'critical'
    ppe = ppe_procs.get(camera_id)
    if ppe and (ppe.stats.get('violation_count') or 0) > 0:
        max_sev = max(max_sev, 2)
    if max_sev >= 3:
        return 'critical'
    if max_sev >= 2:
        return 'warning'
    return 'normal'


def _default_camera_coords(camera_id, index, center_lat, center_lng):
    h = abs(hash(camera_id)) % 997
    ring = index // 8
    slot = index % 8
    angle = (slot / 8.0) * 2 * math.pi
    radius = 0.00035 * (1 + ring * 0.6)
    lat = center_lat + radius * math.cos(angle) + (h % 11) * 0.00001
    lng = center_lng + radius * math.sin(angle) + (h % 13) * 0.00001
    return lat, lng


@app.route('/api/plant-map', methods=['GET'])
def api_plant_map():
    local = _system_kv_get('local_settings', _local_settings_default())
    plant = local.get('plant_map') if isinstance(local.get('plant_map'), dict) else {}
    center_lat = float(plant.get('center_lat') or 28.6139)
    center_lng = float(plant.get('center_lng') or 77.2090)
    default_zoom = int(plant.get('default_zoom') or 16)
    counts = {'normal': 0, 'warning': 0, 'critical': 0, 'offline': 0}
    cameras_out = []
    conn = get_db()
    rows = conn.execute('SELECT camera_id, name, latitude, longitude FROM cameras ORDER BY added_time').fetchall()
    conn.close()
    for i, row in enumerate(rows):
        cid = row['camera_id']
        health = _camera_map_status(cid)
        counts[health] = counts.get(health, 0) + 1
        lat = row['latitude']
        lng = row['longitude']
        position_default = False
        if lat is None or lng is None:
            lat, lng = _default_camera_coords(cid, i, center_lat, center_lng)
            position_default = True
        reader = camera_readers.get(cid)
        cameras_out.append({
            'camera_id': cid,
            'name': row['name'] or cid,
            'latitude': float(lat),
            'longitude': float(lng),
            'position_default': position_default,
            'health': health,
            'stream_status': reader.status if reader else 'stopped',
            'headcount_active': cid in headcount_procs,
            'entryexit_active': cid in entryexit_procs,
            'flapgate_active': cid in flapgate_procs,
            'ppe_active': cid in ppe_procs,
            'fire_smoke_active': cid in fire_smoke_procs,
        })
    return jsonify({
        'center': {'lat': center_lat, 'lng': center_lng, 'zoom': default_zoom},
        'counts': counts,
        'cameras': cameras_out,
        'total': len(cameras_out),
    })


@app.route('/api/cameras/<camera_id>/location', methods=['PUT'])
def api_camera_location(camera_id):
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data.get('latitude'))
        lng = float(data.get('longitude'))
    except (TypeError, ValueError):
        return jsonify({'error': 'latitude and longitude are required'}), 400
    conn = get_db()
    row = conn.execute('SELECT camera_id FROM cameras WHERE camera_id=?', (camera_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Camera not found'}), 404
    conn.execute('UPDATE cameras SET latitude=?, longitude=? WHERE camera_id=?', (lat, lng, camera_id))
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'camera_id': camera_id, 'latitude': lat, 'longitude': lng})


@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
def api_alerts_resolve(alert_id):
    conn = get_db()
    row = conn.execute('SELECT id FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Alert not found'}), 404
    conn.execute("UPDATE alert_events SET status='resolved' WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    return jsonify({'ok': True, 'status': 'resolved'})


@app.route('/api/alerts/<int:alert_id>', methods=['GET'])
def api_alerts_get(alert_id):
    conn = get_db()
    r = conn.execute(
        'SELECT id, camera_id, detection_type, alert_label, severity, snapshot_path, meta_json, created_time FROM alert_events WHERE id=?',
        (alert_id,)
    ).fetchone()
    conn.close()
    if not r:
        return jsonify({'error': 'Alert not found'}), 404
    try:
        meta = json.loads(r['meta_json']) if r['meta_json'] else {}
    except Exception:
        meta = {}
    return jsonify({
        'id': int(r['id']),
        'camera_id': r['camera_id'],
        'detection_type': r['detection_type'],
        'alert_label': r['alert_label'],
        'severity': r['severity'] or 'medium',
        'created_time': r['created_time'],
        'snapshot_url': f"/api/alerts/{int(r['id'])}/snapshot",
        'meta': meta,
        'qwen_ready': _qwen25vl_ready(),
        'qwen_device': _qwen_resolve_device(),
        'beta_vlm_device': _vlm_beta_device(),
        'qwen_unload_after_use': VISION_QWEN_UNLOAD_AFTER_USE,
    })


@app.route('/api/alerts/<int:alert_id>/snapshot', methods=['GET'])
def api_alerts_snapshot(alert_id):
    conn = get_db()
    r = conn.execute('SELECT snapshot_path FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    conn.close()
    if not r or not r['snapshot_path'] or not os.path.isfile(r['snapshot_path']):
        return '', 404
    try:
        with open(r['snapshot_path'], 'rb') as f:
            data = f.read()
        return Response(data, mimetype='image/jpeg')
    except Exception:
        return '', 500


@app.route('/api/alerts/<int:alert_id>/investigate', methods=['POST'])
def api_alerts_investigate(alert_id):
    data = request.json or {}
    language = (data.get('language') or 'English').strip()
    mode = (data.get('mode') or 'text').strip().lower()
    user_text = (data.get('message') or '').strip()
    odd_context = (data.get('odd_context') or '').strip()
    conn = get_db()
    r = conn.execute('SELECT snapshot_path, detection_type, alert_label, meta_json FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify({'error': 'Alert not found'}), 404
    snapshot_path = r['snapshot_path']
    if not snapshot_path or not os.path.isfile(snapshot_path):
        return jsonify({'error': 'Snapshot not found'}), 404
    # Make investigation anomaly-aware instead of generic scene commentary.
    try:
        meta = json.loads(r['meta_json']) if r['meta_json'] else {}
    except Exception:
        meta = {}
    det_type = (r['detection_type'] or '').strip()
    alert_label = (r['alert_label'] or '').strip()
    focus_parts = []
    if det_type:
        focus_parts.append(f"detection_type={det_type}")
    if alert_label:
        focus_parts.append(f"alert_label={alert_label}")
    if isinstance(meta, dict):
        instr = str(meta.get('instruction') or '').strip()
        if instr:
            focus_parts.append(f"instruction={instr}")
        new_lbls = meta.get('new_alert_labels')
        if isinstance(new_lbls, list) and new_lbls:
            focus_parts.append("new_alert_labels=" + ", ".join(str(x) for x in new_lbls))
    focus_context = " | ".join(focus_parts)
    if isinstance(meta, dict):
        # Add stronger, human-readable focus hint from structured anomaly labels.
        lbls = meta.get('new_alert_labels') or meta.get('alert_labels') or []
        if isinstance(lbls, list) and lbls:
            focus_context = (focus_context + " | " if focus_context else "") + "anomaly_labels=" + ", ".join(str(x) for x in lbls)
    if focus_context:
        user_text = (user_text + " " if user_text else "") + (
            "Focus analysis strictly on this alert anomaly context: " + focus_context + ". "
            "Explain why this specific alert likely triggered and provide targeted corrective action for this anomaly."
        )
    logger.info(f"Incident investigate request: alert_id={alert_id}, language={language}, mode={mode}")
    result = _qwen_infer_on_image(
        snapshot_path, language=language, mode=mode, user_text=user_text, odd_context=odd_context, alert_context=focus_context, task='analysis'
    )
    _save_incident_ai_output(
        alert_id, 'analysis', language, mode,
        prompt_text=user_text if user_text else 'Initial incident investigation',
        response_text=result.get('full_response', '')
    )
    return jsonify(result)


@app.route('/api/alerts/<int:alert_id>/chat', methods=['POST'])
def api_alerts_chat(alert_id):
    data = request.json or {}
    language = (data.get('language') or 'English').strip()
    mode = (data.get('mode') or 'text').strip().lower()
    message = (data.get('message') or '').strip()
    odd_context = (data.get('odd_context') or '').strip()
    logger.info(f"Incident chat request: alert_id={alert_id}, language={language}, mode={mode}, chars={len(message)}")
    conn = get_db()
    r = conn.execute('SELECT snapshot_path FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify({'error': 'Alert not found'}), 404
    snapshot_path = r['snapshot_path']
    if not snapshot_path or not os.path.isfile(snapshot_path):
        return jsonify({'error': 'Snapshot not found'}), 404
    result = _qwen_infer_on_image(
        snapshot_path, language=language, mode=mode, user_text=message, odd_context=odd_context, task='chat'
    )
    # Chat response should answer user question directly; avoid section duplication in UI.
    chat_text = result.get('full_response') or result.get('recommendations') or result.get('regular_commentary') or ''
    _save_incident_ai_output(alert_id, 'chat', language, mode, prompt_text=message, response_text=chat_text)
    return jsonify({'full_response': chat_text})


@app.route('/api/alerts/<int:alert_id>/workspace', methods=['GET'])
def api_alerts_workspace_get(alert_id):
    conn = get_db()
    alert = conn.execute('SELECT id FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    if not alert:
        conn.close()
        return jsonify({'error': 'Alert not found'}), 404
    row = conn.execute(
        'SELECT analysis_language, analysis_mode, chat_language, chat_mode, odd_context, commentary, rootcause, chat_history_json, updated_time '
        'FROM incident_workspaces WHERE alert_id=?',
        (alert_id,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({'saved': False})
    return jsonify({
        'saved': True,
        'analysis_language': row['analysis_language'] or 'English',
        'analysis_mode': row['analysis_mode'] or 'text',
        'chat_language': row['chat_language'] or 'English',
        'chat_mode': row['chat_mode'] or 'text',
        'odd_context': row['odd_context'] or '',
        'commentary': row['commentary'] or '',
        'rootcause': row['rootcause'] or '',
        'chat_history': _parse_chat_history(row['chat_history_json']),
        'updated_time': row['updated_time'],
    })


@app.route('/api/alerts/<int:alert_id>/workspace', methods=['POST'])
def api_alerts_workspace_save(alert_id):
    data = request.json or {}
    analysis_language = str(data.get('analysis_language') or 'English').strip()[:50]
    analysis_mode = str(data.get('analysis_mode') or 'text').strip().lower()
    chat_language = str(data.get('chat_language') or 'English').strip()[:50]
    chat_mode = str(data.get('chat_mode') or 'text').strip().lower()
    odd_context = str(data.get('odd_context') or '')[:20000]
    commentary = str(data.get('commentary') or '')[:50000]
    rootcause = str(data.get('rootcause') or '')[:50000]
    chat_history_json = _serialize_chat_history(data.get('chat_history'))
    if analysis_mode not in ('text', 'voice'):
        analysis_mode = 'text'
    if chat_mode not in ('text', 'voice'):
        chat_mode = 'text'

    conn = get_db()
    alert = conn.execute('SELECT id FROM alert_events WHERE id=?', (alert_id,)).fetchone()
    if not alert:
        conn.close()
        return jsonify({'error': 'Alert not found'}), 404
    conn.execute(
        'INSERT INTO incident_workspaces (alert_id, analysis_language, analysis_mode, chat_language, chat_mode, odd_context, commentary, rootcause, chat_history_json, updated_time) '
        'VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP) '
        'ON CONFLICT(alert_id) DO UPDATE SET '
        'analysis_language=excluded.analysis_language, '
        'analysis_mode=excluded.analysis_mode, '
        'chat_language=excluded.chat_language, '
        'chat_mode=excluded.chat_mode, '
        'odd_context=excluded.odd_context, '
        'commentary=excluded.commentary, '
        'rootcause=excluded.rootcause, '
        'chat_history_json=excluded.chat_history_json, '
        'updated_time=CURRENT_TIMESTAMP',
        (alert_id, analysis_language, analysis_mode, chat_language, chat_mode, odd_context, commentary, rootcause, chat_history_json)
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})


def _extract_odd_upload_text(file_bytes, ext):
    ext = (ext or '').lower()
    if ext in ('.txt', '.md', '.csv', '.log', ''):
        return file_bytes.decode('utf-8', errors='replace').strip()
    if ext == '.pdf':
        if not _PYPDF_AVAILABLE or _PdfReaderClass is None:
            raise ValueError('PDF uploads require the pypdf package in the container image.')
        reader = _PdfReaderClass(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or '')
        return '\n'.join(parts).strip()
    raise ValueError('Unsupported file type. Use .txt, .md, .csv, .log, or .pdf.')


@app.route('/api/alerts/<int:alert_id>/odd_upload', methods=['POST'])
def api_alerts_odd_upload(alert_id):
    """Extract text from an uploaded ODD document and optionally persist a copy under data/odd_uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file field "file" in multipart form.'}), 400
    up = request.files['file']
    if not up or not up.filename:
        return jsonify({'error': 'Empty filename.'}), 400
    raw_name = secure_filename(up.filename)
    ext = os.path.splitext(raw_name)[1]
    max_bytes = int(os.environ.get('VISION_ODD_UPLOAD_MAX_MB', '4')) * 1024 * 1024
    blob = up.read(max_bytes + 1)
    if len(blob) > max_bytes:
        return jsonify({'error': f'File too large (max {max_bytes // (1024 * 1024)} MB).'}), 400
    try:
        text = _extract_odd_upload_text(blob, ext)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f'ODD upload extract error: {e}')
        return jsonify({'error': 'Could not read file.'}), 400
    save_dir = os.path.join(ODD_UPLOADS_DIR, str(int(alert_id)))
    os.makedirs(save_dir, exist_ok=True)
    try:
        dest = os.path.join(save_dir, raw_name or f'odd{ext or ".txt"}')
        with open(dest, 'wb') as wf:
            wf.write(blob)
    except Exception as e:
        logger.warning(f'ODD file copy skipped: {e}')
    return jsonify({'text': text, 'filename': raw_name, 'chars': len(text)})


async def _edge_tts_synthesize_bytes(text, voice):
    communicate = _edge_tts_mod.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            chunks.append(chunk['data'])
    return b''.join(chunks)


@app.route('/api/tts', methods=['POST'])
def api_tts_multilingual():
    """Neural TTS for languages often missing from browser SpeechSynthesis (e.g. Hindi/Kannada on Linux)."""
    data = request.json or {}
    text = (data.get('text') or '').strip()
    language = (data.get('language') or 'English').strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400
    if not _EDGE_TTS_AVAILABLE or _edge_tts_mod is None:
        return jsonify({'error': 'edge-tts not installed', 'use_browser': True}), 501
    max_chars = int(os.environ.get('VISION_TTS_MAX_CHARS', '6000'))
    if len(text) > max_chars:
        text = text[:max_chars]
    voice = EDGE_TTS_VOICES.get(language, EDGE_TTS_VOICES['English'])
    try:
        audio = asyncio.run(_edge_tts_synthesize_bytes(text, voice))
    except Exception as e:
        logger.error(f'edge-tts failed: {e}')
        return jsonify({'error': str(e), 'use_browser': True}), 500
    return Response(audio, mimetype='audio/mpeg')


@app.route('/api/vlm/qwen/unload', methods=['POST'])
def api_unload_qwen():
    """Explicitly unload Qwen model to free resources from UI exit action."""
    _unload_qwen25vl()
    return jsonify({'success': True, 'qwen_ready': _qwen25vl_ready()})


# ---------- API: Detection Config ----------

@app.route('/api/detection/headcount/enable/<camera_id>', methods=['POST'])
def api_enable_headcount(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404

    data = request.json or {}
    conf = float(data.get('confidence', 0.1))

    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
                 (camera_id, 'headcount', 1, json.dumps({'confidence': conf})))
    conn.commit()
    conn.close()

    if camera_id in headcount_procs:
        headcount_procs[camera_id].stop()

    engine = get_inference_engine('headcount', HEADCOUNT_MODEL_PATH)
    proc = HeadCountProcessor(camera_id, camera_readers[camera_id], engine, conf)
    headcount_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/headcount/disable/<camera_id>', methods=['POST'])
def api_disable_headcount(camera_id):
    if camera_id in headcount_procs:
        headcount_procs[camera_id].stop()
        del headcount_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'headcount'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/entryexit/enable/<camera_id>', methods=['POST'])
def api_enable_entryexit(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404

    data = request.json or {}
    entry_zone = data.get('entry_zone', [])
    exit_zone = data.get('exit_zone', [])
    canvas_w = int(data.get('canvas_width', 800))
    canvas_h = int(data.get('canvas_height', 450))
    conf = float(data.get('confidence', 0.5))

    if len(entry_zone) < 3 or len(exit_zone) < 3:
        return jsonify({'error': 'Both zones need at least 3 points'}), 400

    entry_tuples = [(p['x'], p['y']) if isinstance(p, dict) else tuple(p) for p in entry_zone]
    exit_tuples = [(p['x'], p['y']) if isinstance(p, dict) else tuple(p) for p in exit_zone]

    config = {
        'entry_zone': [list(t) for t in entry_tuples],
        'exit_zone': [list(t) for t in exit_tuples],
        'canvas_width': canvas_w,
        'canvas_height': canvas_h,
        'confidence': conf
    }

    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
                 (camera_id, 'entryexit', 1, json.dumps(config)))
    conn.commit()
    conn.close()

    if camera_id in entryexit_procs:
        entryexit_procs[camera_id].stop()

    # Person-only (COCO class 0) so chairs etc. are not detected or assigned IDs.
    engine = get_inference_engine('entryexit', ENTRYEXIT_MODEL_PATH, classes=[0])
    proc = EntryExitProcessor(
        camera_id, camera_readers[camera_id],
        entry_tuples, exit_tuples,
        (canvas_w, canvas_h), engine, conf
    )
    entryexit_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/entryexit/disable/<camera_id>', methods=['POST'])
def api_disable_entryexit(camera_id):
    if camera_id in entryexit_procs:
        entryexit_procs[camera_id].stop()
        del entryexit_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'entryexit'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/flapgate/enable/<camera_id>', methods=['POST'])
def api_enable_flapgate(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404

    data = request.json or {}
    gate_zones_raw = data.get('gate_zones', {})
    canvas_w = int(data.get('canvas_width', 800))
    canvas_h = int(data.get('canvas_height', 450))
    conf = float(data.get('confidence', 0.65))

    gate_zones = {}
    for zid_str, pts in gate_zones_raw.items():
        zid = int(zid_str)
        if len(pts) < 3:
            return jsonify({'error': f'Gate {zid} zone needs at least 3 points'}), 400
        gate_zones[zid] = [(p['x'], p['y']) if isinstance(p, dict) else tuple(p) for p in pts]

    if len(gate_zones) < 3:
        return jsonify({'error': 'All 3 gate zones must be configured'}), 400

    config = {
        'gate_zones': {str(k): [list(p) for p in v] for k, v in gate_zones.items()},
        'canvas_width': canvas_w,
        'canvas_height': canvas_h,
        'confidence': conf
    }

    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
                 (camera_id, 'flapgate', 1, json.dumps(config)))
    conn.commit()
    conn.close()

    if camera_id in flapgate_procs:
        flapgate_procs[camera_id].stop()

    # Person-only (COCO class 0) so chairs etc. are not detected.
    engine = get_inference_engine('flapgate', FLAPGATE_MODEL_PATH, classes=[0])
    proc = FlapGateProcessor(
        camera_id, camera_readers[camera_id],
        gate_zones, (canvas_w, canvas_h), engine, conf
    )
    flapgate_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/flapgate/disable/<camera_id>', methods=['POST'])
def api_disable_flapgate(camera_id):
    if camera_id in flapgate_procs:
        flapgate_procs[camera_id].stop()
        del flapgate_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'flapgate'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/ppe/enable/<camera_id>', methods=['POST'])
def api_enable_ppe(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404

    data = request.json or {}
    conf = float(data.get('confidence', 0.4))

    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
                 (camera_id, 'ppe', 1, json.dumps({'confidence': conf})))
    conn.commit()
    conn.close()

    if camera_id in ppe_procs:
        ppe_procs[camera_id].stop()

    engine = get_inference_engine('ppe', PPE_MODEL_PATH, classes=None)
    proc = PPEDetectionProcessor(camera_id, camera_readers[camera_id], engine, conf)
    ppe_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/ppe/disable/<camera_id>', methods=['POST'])
def api_disable_ppe(camera_id):
    if camera_id in ppe_procs:
        ppe_procs[camera_id].stop()
        del ppe_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'ppe'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/firesmoke/enable/<camera_id>', methods=['POST'])
def api_enable_firesmoke(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404
    data = request.json or {}
    conf = float(data.get('confidence', 0.35))
    conn = get_db()
    conn.execute('INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
                 (camera_id, 'firesmoke', 1, json.dumps({'confidence': conf})))
    conn.commit()
    conn.close()
    if camera_id in fire_smoke_procs:
        fire_smoke_procs[camera_id].stop()
    engine = get_inference_engine('firesmoke', FIRE_SMOKE_MODEL_PATH, classes=None)
    proc = FireSmokeDetectionProcessor(camera_id, camera_readers[camera_id], engine, conf)
    fire_smoke_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/firesmoke/disable/<camera_id>', methods=['POST'])
def api_disable_firesmoke(camera_id):
    if camera_id in fire_smoke_procs:
        fire_smoke_procs[camera_id].stop()
        del fire_smoke_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'firesmoke'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/anpr/status', methods=['GET'])
def api_anpr_status():
    plate_path = _anpr_resolve_weight_path(ANPR_PLATE_MODEL_PATH)
    vehicle_path = _anpr_resolve_weight_path(ANPR_VEHICLE_MODEL_PATH)
    return jsonify({
        'active_cameras': list(anpr_procs.keys()),
        'stats': {cid: proc.stats for cid, proc in anpr_procs.items()},
        'libreyolo_available': _LIBREYOLO_AVAILABLE,
        'paddleocr_available': _PADDLEOCR_AVAILABLE,
        'plate_model_path': plate_path,
        'vehicle_model_path': vehicle_path,
        'plate_model_found': bool(plate_path and os.path.isfile(plate_path)),
        'vehicle_model_found': bool(vehicle_path and os.path.isfile(vehicle_path)),
    })


@app.route('/api/detection/anpr/enable/<camera_id>', methods=['POST'])
def api_enable_anpr(camera_id):
    if camera_id not in camera_readers:
        return jsonify({'error': 'Camera not found'}), 404
    if not _LIBREYOLO_AVAILABLE:
        return jsonify({'error': 'LibreYOLO is not available. Install/configure libreyolo for ANPR.'}), 400
    if not _PADDLEOCR_AVAILABLE:
        return jsonify({'error': 'PaddleOCR is not available. Install paddleocr for ANPR plate reading.'}), 400

    plate_path = _anpr_resolve_weight_path(ANPR_PLATE_MODEL_PATH)
    vehicle_path = _anpr_resolve_weight_path(ANPR_VEHICLE_MODEL_PATH)
    if not os.path.isfile(plate_path):
        return jsonify({'error': f'ANPR plate model not found: {plate_path}'}), 400
    if not os.path.isfile(vehicle_path):
        return jsonify({'error': f'ANPR vehicle model not found: {vehicle_path}'}), 400

    data = request.json or {}
    conf = float(data.get('confidence', ANPR_VEHICLE_CONF))
    speed_threshold = float(data.get('speed_threshold_kmh', 50))
    meters_per_pixel = float(data.get('meters_per_pixel', ANPR_METERS_PER_PIXEL))
    config = {
        'confidence': conf,
        'speed_threshold_kmh': speed_threshold,
        'meters_per_pixel': meters_per_pixel,
    }

    conn = get_db()
    conn.execute(
        'INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
        (camera_id, 'anpr', 1, json.dumps(config))
    )
    conn.commit()
    conn.close()

    if camera_id in anpr_procs:
        anpr_procs[camera_id].stop()

    proc = ANPRDetectionProcessor(
        camera_id, camera_readers[camera_id],
        conf=conf, speed_threshold_kmh=speed_threshold,
        meters_per_pixel=meters_per_pixel,
    )
    anpr_procs[camera_id] = proc
    proc.start()
    return jsonify({'success': True})


@app.route('/api/detection/anpr/disable/<camera_id>', methods=['POST'])
def api_disable_anpr(camera_id):
    if camera_id in anpr_procs:
        anpr_procs[camera_id].stop()
        del anpr_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?', (camera_id, 'anpr'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ---------- FR Detection API ----------

@app.route('/api/fr/status', methods=['GET'])
def api_fr_status():
    """Return FR availability flags and which cameras have FR active."""
    _act = list(fr_procs.keys())
    logger.info(f'FR GET /api/fr/status → active_cameras={_act!r} count={len(_act)}')
    return jsonify({
        'active_cameras': _act,
        'stats': {cid: proc.stats for cid, proc in fr_procs.items()},
        'insightface_available': _FR_INSIGHTFACE_AVAILABLE,
        'yolo_available': _FR_YOLO_AVAILABLE,
        'deepsort_available': _FR_DEEPSORT_AVAILABLE,
        'csv_path': FR_AUTOFACEDATA_CSV,
        'base_dir': FR_AUTOFACEDATA_BASE_DIR,
    })


@app.route('/api/fr/datasets', methods=['GET'])
def api_fr_datasets():
    """List all people registered in the CSV with folder info."""
    rows, _ = _fr_get_or_create_csv()
    result = []
    for row in rows:
        name = row.get('person_name', '')
        file_path = row.get('file_path', '')
        paths = [p.strip() for p in file_path.split(',') if p.strip()] if file_path else []
        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, name)
        image_count = 0
        if os.path.isdir(person_dir):
            image_count = len([f for f in os.listdir(person_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        result.append({
            'person_name': name,
            'image_count': image_count,
            'folder': person_dir,
            'paths_count': len(paths),
        })
    return jsonify({
        'datasets': result,
        'csv_path': FR_AUTOFACEDATA_CSV,
        'base_dir': FR_AUTOFACEDATA_BASE_DIR,
    })


@app.route('/api/fr/folder/add', methods=['POST'])
def api_fr_add_folder():
    """Create a new person folder and register in CSV.
    If the folder already contains images (manually placed), file_path is populated immediately."""
    data = request.json or {}
    person_name = data.get('person_name', '').strip()
    if not person_name:
        return jsonify({'error': 'person_name is required'}), 400
    if re.search(r'[^\w\-. ]', person_name):
        return jsonify({'error': 'person_name contains invalid characters'}), 400
    try:
        person_dir = _fr_ensure_dirs(person_name)
        # Scan folder for pre-existing images so file_path is never left empty
        _FR_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
        existing_images = []
        if os.path.isdir(person_dir):
            for fname in sorted(os.listdir(person_dir)):
                if os.path.splitext(fname)[1].lower() in _FR_IMG_EXTS:
                    existing_images.append(os.path.join(person_dir, fname))
        path_str = ','.join(existing_images)

        rows, fieldnames = _fr_get_or_create_csv()
        found = False
        for row in rows:
            if str(row.get('person_name', '')).strip() == person_name:
                # Always refresh file_path in case images were added manually
                row['file_path'] = path_str
                found = True
                break
        if not found:
            rows.append({'person_name': person_name, 'file_path': path_str})
        _fr_save_csv(rows, fieldnames)

        return jsonify({
            'success': True,
            'person_name': person_name,
            'folder': person_dir,
            'images_found': len(existing_images),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fr/scan_and_sync', methods=['POST'])
def api_fr_scan_and_sync():
    """Scan all subdirectories of FR_AUTOFACEDATA_BASE_DIR and rebuild autofacedata.csv.

    Covers employees whose folders were placed manually on the server (not via the UI).
    For each person folder found it:
      - lists every image file (jpg/jpeg/png/bmp) in sorted order
      - writes comma-separated relative paths into the person's file_path column
      - adds a new row if the person did not exist in the CSV yet
      - leaves rows for CSV-registered persons whose folder no longer exists intact
        (image_count will show 0, user can delete them manually)
    After writing, the embedding cache is invalidated and running FR processors reload.
    """
    _FR_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    try:
        logger.info(f'FR scan_and_sync: base_dir={FR_AUTOFACEDATA_BASE_DIR!r}  csv={FR_AUTOFACEDATA_CSV!r}')
        os.makedirs(FR_AUTOFACEDATA_BASE_DIR, exist_ok=True)

        rows, fieldnames = _fr_get_or_create_csv()
        # Build a mutable map: person_name -> row dict
        row_map = {}
        for r in rows:
            name = str(r.get('person_name', '')).strip()
            if name:
                row_map[name] = dict(r)

        new_persons = 0
        updated_persons = 0

        for entry in sorted(os.scandir(FR_AUTOFACEDATA_BASE_DIR), key=lambda e: e.name.lower()):
            if not entry.is_dir():
                continue
            person_name = entry.name
            # Skip hidden dirs and any internal cache-related dirs
            if person_name.startswith('.') or person_name.startswith('_'):
                continue

            images = []
            try:
                for fname in sorted(os.listdir(entry.path)):
                    if os.path.splitext(fname)[1].lower() in _FR_IMG_EXTS:
                        images.append(os.path.join(entry.path, fname))
            except OSError:
                continue

            logger.info(f'FR scan_and_sync: person={person_name!r}  images_found={len(images)}  '
                        f'first={images[0] if images else "none"}')
            path_str = ','.join(images)

            if person_name in row_map:
                row_map[person_name]['file_path'] = path_str
                updated_persons += 1
            else:
                row_map[person_name] = {'person_name': person_name, 'file_path': path_str}
                new_persons += 1

        # Preserve original row order then append new persons alphabetically
        ordered = list(row_map.values())
        _fr_save_csv(ordered, fieldnames)

        # Invalidate embedding cache
        import glob as _fr_glob
        for f in _fr_glob.glob(os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.*')):
            try:
                os.remove(f)
            except Exception:
                pass

        # Trigger embedding reload on any running FR processors
        for proc in list(fr_procs.values()):
            threading.Thread(target=proc.reload_embeddings, daemon=True).start()

        total = new_persons + updated_persons
        logger.info(f'FR scan_and_sync: {total} folder(s) synced ({new_persons} new, {updated_persons} updated)')
        return jsonify({
            'success': True,
            'total_synced': total,
            'new_persons': new_persons,
            'updated_persons': updated_persons,
            'message': (
                f'Synced {total} employee folder(s) into CSV '
                f'({new_persons} new, {updated_persons} updated).'
            ),
        })
    except Exception as e:
        logger.error(f'FR scan_and_sync error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/fr/folder/delete', methods=['POST'])
def api_fr_delete_folder():
    """Delete a person's folder and remove from CSV. Invalidates embedding cache."""
    import shutil
    data = request.json or {}
    person_name = data.get('person_name', '').strip()
    if not person_name:
        return jsonify({'error': 'person_name is required'}), 400
    try:
        person_dir = os.path.join(FR_AUTOFACEDATA_BASE_DIR, person_name)
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir)
        rows, fieldnames = _fr_get_or_create_csv()
        rows = [r for r in rows if str(r.get('person_name', '')).strip() != person_name]
        _fr_save_csv(rows, fieldnames)
        import glob as _fr_glob
        for f in _fr_glob.glob(os.path.join(FR_AUTOFACEDATA_BASE_DIR, f'{FR_CACHE_FILENAME}.*')):
            try:
                os.remove(f)
            except Exception:
                pass
        for proc in list(fr_procs.values()):
            threading.Thread(target=proc.reload_embeddings, daemon=True).start()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fr/collect/start', methods=['POST'])
def api_fr_collect_start():
    """Start face data collection from an RTSP stream for a named person."""
    data = request.json or {}
    rtsp_url = data.get('rtsp_url', '').strip()
    person_name = data.get('person_name', '').strip()
    if not rtsp_url or not person_name:
        return jsonify({'error': 'rtsp_url and person_name are required'}), 400
    if re.search(r'[^\w\-. ]', person_name):
        return jsonify({'error': 'person_name contains invalid characters'}), 400

    session_id = f'fr_collect_{int(time.time() * 1000)}'
    state = {
        'stop': False, 'frame': None, 'error': None,
        'latest_frame': None, 'latest_ok': False,
        'counts': {}, 'completed': [], 'completed_ids': set(),
        'csv_updated': False, 'person_name': person_name,
    }
    fr_collect_states[session_id] = state
    threading.Thread(target=_fr_reader_thread, args=(rtsp_url, state), daemon=True).start()
    threading.Thread(target=_fr_processor_thread, args=(state, person_name), daemon=True).start()
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/fr/collect/stop', methods=['POST'])
def api_fr_collect_stop():
    """Stop an active data collection session and trigger embedding reload."""
    data = request.json or {}
    session_id = data.get('session_id', '').strip()
    if session_id in fr_collect_states:
        fr_collect_states[session_id]['stop'] = True
        for proc in list(fr_procs.values()):
            threading.Thread(target=proc.reload_embeddings, daemon=True).start()
    return jsonify({'success': True})


@app.route('/api/fr/collect/status/<session_id>', methods=['GET'])
def api_fr_collect_status(session_id):
    """Get progress of a data collection session."""
    state = fr_collect_states.get(session_id)
    if not state:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({
        'error': state.get('error'),
        'counts': state.get('counts', {}),
        'completed': [int(x) for x in state.get('completed_ids', set())],
        'person_name': state.get('person_name', ''),
        'csv_updated': state.get('csv_updated', False),
        'stopped': state.get('stop', False),
    })


@app.route('/api/fr/detection/enable/<camera_id>', methods=['POST'])
def api_fr_enable(camera_id):
    """Enable face recognition on a camera."""
    # Some cameras may exist in DB but their reader is not currently instantiated
    # (e.g. after runtime hiccups/restarts). For FR, auto-recreate on demand.
    if camera_id not in camera_readers:
        conn = get_db()
        cam = conn.execute('SELECT camera_id, rtsp_url FROM cameras WHERE camera_id=?', (camera_id,)).fetchone()
        conn.close()
        if not cam:
            return jsonify({'error': 'Camera not found in DB'}), 404
        rtsp_url = str(cam['rtsp_url'] or '').strip()
        if not rtsp_url:
            return jsonify({'error': f'Camera {camera_id} has empty RTSP URL'}), 400
        try:
            reader = CameraReader(camera_id, rtsp_url)
            camera_readers[camera_id] = reader
            reader.start()
            logger.info(f'FR {camera_id}: Recreated missing camera reader')
        except Exception as e:
            return jsonify({'error': f'Failed to start camera reader for {camera_id}: {e}'}), 500
    data = request.json or {}
    threshold = float(data.get('threshold', FR_RECOGNITION_THRESHOLD))
    logger.info(f'FR enable requested: camera_id={camera_id} threshold={threshold}')

    conn = get_db()
    conn.execute(
        'INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)',
        (camera_id, 'fr_detection', 1, json.dumps({'threshold': threshold}))
    )
    conn.commit()
    conn.close()

    if camera_id in fr_procs:
        fr_procs[camera_id].stop()
    proc = FaceRecognitionProcessor(camera_id, camera_readers[camera_id], threshold=threshold)
    fr_procs[camera_id] = proc
    proc.start()
    logger.info(f'FR enabled: camera_id={camera_id} active_now={list(fr_procs.keys())}')
    return jsonify({'success': True})


@app.route('/api/fr/detection/disable/<camera_id>', methods=['POST'])
def api_fr_disable(camera_id):
    """Disable face recognition on a camera."""
    if camera_id in fr_procs:
        fr_procs[camera_id].stop()
        del fr_procs[camera_id]
    conn = get_db()
    conn.execute('DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?',
                 (camera_id, 'fr_detection'))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/detection/beta/config', methods=['GET'])
def api_beta_config_get():
    """Return all beta detection prompts and default confidence."""
    conn = get_db()
    rows = conn.execute(
        'SELECT id, prompt_text, camera_ids, confidence, enabled FROM beta_detection_prompts ORDER BY id'
    ).fetchall()
    beta_model = 'owlv2'
    settings_conf = None
    try:
        srow = conn.execute('SELECT model, confidence FROM beta_settings WHERE id=1').fetchone()
        if srow:
            if srow['model']:
                beta_model = str(srow['model']).strip().lower() or 'owlv2'
            if srow['confidence'] is not None:
                settings_conf = float(srow['confidence'])
    except sqlite3.OperationalError:
        pass
    conn.close()
    prompts = []
    default_conf = settings_conf if settings_conf is not None else 0.2
    for r in rows:
        try:
            camera_ids = json.loads(r['camera_ids']) if r['camera_ids'] else []
        except Exception:
            camera_ids = []
        prompts.append({
            'id': r['id'],
            'prompt_text': r['prompt_text'] or '',
            'camera_ids': camera_ids,
            'confidence': float(r['confidence']) if r['confidence'] is not None else 0.2,
            'enabled': bool(r['enabled']),
        })
        if r['confidence'] is not None and settings_conf is None:
            default_conf = float(r['confidence'])
    return jsonify({
        'prompts': prompts,
        'confidence': default_conf,
        'beta_model': beta_model,
        'owlv2_available': _OWLV2_AVAILABLE,
        'florence2_available': _florence2_ready(),
        'qwen_available': _qwen25vl_ready(),
    })


@app.route('/api/detection/beta/save', methods=['POST'])
def api_beta_config_save():
    """Save beta prompts and camera assignments; (re)start beta processors for affected cameras."""
    data = request.json or {}
    prompts_data = data.get('prompts', [])
    confidence = float(data.get('confidence', 0.2))
    backend = (data.get('model') or data.get('beta_model') or 'owlv2').strip().lower()
    if backend not in ('owlv2', 'florence2', 'qwen'):
        backend = 'owlv2'
    if backend == 'owlv2' and not _OWLV2_AVAILABLE:
        return jsonify({'error': 'OWLv2 not available. Install: pip install transformers pillow'}), 400
    if backend == 'florence2' and not _florence2_ready():
        return jsonify({
            'error': 'Florence-2 not available. Install transformers with Florence-2 support and place the model under florence2_model2/Florence-2-large (or set VISION_FLORENCE2_LOCAL_PATH).'
        }), 400
    if backend == 'qwen' and not _qwen25vl_ready():
        return jsonify({
            'error': 'Qwen monitor backend not available. Ensure Qwen local cache is mounted and VISION_QWEN25VL_LOCAL_PATH is correct.'
        }), 400

    conn = get_db()
    conn.execute('DELETE FROM beta_detection_prompts')
    for p in prompts_data:
        prompt_text = (p.get('prompt_text') or '').strip()
        camera_ids = p.get('camera_ids') or []
        if not isinstance(camera_ids, list):
            camera_ids = []
        if not prompt_text and not camera_ids:
            continue
        conn.execute(
            'INSERT INTO beta_detection_prompts (prompt_text, camera_ids, confidence, enabled) VALUES (?,?,?,?)',
            (prompt_text, json.dumps(camera_ids), confidence, 1)
        )
    try:
        conn.execute(
            'INSERT OR REPLACE INTO beta_settings (id, model, confidence) VALUES (1, ?, ?)',
            (backend, confidence)
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

    # Build camera_id -> list of prompt_texts for this camera
    camera_to_prompts = {}
    for p in prompts_data:
        prompt_text = (p.get('prompt_text') or '').strip()
        camera_ids = p.get('camera_ids') or []
        if not prompt_text:
            continue
        for cid in camera_ids:
            if cid not in camera_to_prompts:
                camera_to_prompts[cid] = []
            camera_to_prompts[cid].append(prompt_text)

    # Stop beta for cameras no longer in any prompt
    for cid in list(beta_procs.keys()):
        if cid not in camera_to_prompts:
            beta_procs[cid].stop()
            del beta_procs[cid]
    _maybe_unload_beta_vlms_if_idle()

    # Start or restart beta for each camera that has prompts
    started = []
    skipped = []
    start_errors = []
    for cid, prompt_texts in camera_to_prompts.items():
        if cid not in camera_readers:
            skipped.append({'camera_id': cid, 'reason': 'camera_not_found'})
            continue
        if cid in beta_procs:
            beta_procs[cid].stop()
        proc = _make_beta_processor(cid, camera_readers[cid], prompt_texts, conf=confidence, backend=backend)
        beta_procs[cid] = proc
        proc.start()
        # Give thread a short moment to fail fast on model-load issues.
        time.sleep(0.05)
        if not proc.running:
            start_errors.append({'camera_id': cid, 'status': proc.stats.get('status', 'failed')})
            try:
                proc.stop()
            except Exception:
                pass
            if cid in beta_procs:
                del beta_procs[cid]
        else:
            started.append(cid)

    if camera_to_prompts and not started:
        _maybe_unload_beta_vlms_if_idle()
        return jsonify({
            'error': 'Beta backend did not start on selected cameras.',
            'backend': backend,
            'start_errors': start_errors,
            'skipped': skipped,
        }), 400
    return jsonify({
        'success': True,
        'backend': backend,
        'started_cameras': started,
        'start_errors': start_errors,
        'skipped': skipped,
    })


@app.route('/api/detection/config/<camera_id>')
def api_get_detection_config(camera_id):
    conn = get_db()
    rows = conn.execute('SELECT * FROM detection_configs WHERE camera_id=?', (camera_id,)).fetchall()
    conn.close()
    configs = {}
    for row in rows:
        configs[row['detection_type']] = {
            'enabled': bool(row['enabled']),
            'config': json.loads(row['config_json']) if row['config_json'] else {}
        }
    return jsonify(configs)


# ---------- API: System stats (GPU, CPU, VRAM, threads) ----------

def _get_system_stats():
    """Gather CPU, memory, GPU, and process stats. No existing code changed."""
    out = {
        'cpu_percent': None,
        'cpu_count': None,
        'cpu_freq_mhz': None,
        'memory_percent': None,
        'memory_used_mb': None,
        'memory_available_mb': None,
        'memory_total_mb': None,
        'process_threads': None,
        'process_memory_mb': None,
        'process_cpu_percent': None,
        'gpus': [],
        'uptime_seconds': round(time.time() - startup_time, 1),
        'platform': platform.system(),
        'resource_tuning': {
            'imgsz': INFERENCE_IMGSZ,
            'target_fps': TARGET_FPS,
            'micro_batch': MICRO_BATCH,
            'callback_pool_workers': CALLBACK_POOL_WORKERS,
            'use_fp16': USE_FP16,
            'use_tensorrt_if_available': USE_TENSORRT_IF_AVAILABLE,
        },
    }
    if psutil:
        try:
            out['cpu_percent'] = round(psutil.cpu_percent(interval=0.1), 1)
            out['cpu_count'] = psutil.cpu_count(logical=True)
            try:
                f = psutil.cpu_freq()
                out['cpu_freq_mhz'] = round(f.current, 1) if f else None
            except Exception:
                pass
            v = psutil.virtual_memory()
            out['memory_percent'] = round(v.percent, 1)
            out['memory_used_mb'] = round(v.used / (1024 * 1024), 1)
            out['memory_available_mb'] = round(v.available / (1024 * 1024), 1)
            out['memory_total_mb'] = round(v.total / (1024 * 1024), 1)
            p = psutil.Process(os.getpid())
            out['process_threads'] = p.num_threads()
            out['process_memory_mb'] = round(p.memory_info().rss / (1024 * 1024), 2)
            out['process_cpu_percent'] = round(p.cpu_percent(interval=0.1), 1)
        except Exception as e:
            out['psutil_error'] = str(e)
    if _nvml_ok:
        try:
            dev_count = pynvml.nvmlDeviceGetCount()
            for i in range(dev_count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode('utf-8', errors='replace')
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpu_percent = util.gpu
                except Exception:
                    gpu_percent = None
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    vram_used_mb = round(mem.used / (1024 * 1024), 1)
                    vram_total_mb = round(mem.total / (1024 * 1024), 1)
                    vram_percent = round(100.0 * mem.used / mem.total, 1) if mem.total else None
                except Exception:
                    vram_used_mb = vram_total_mb = vram_percent = None
                out['gpus'].append({
                    'index': i,
                    'name': name,
                    'gpu_util_percent': gpu_percent,
                    'vram_used_mb': vram_used_mb,
                    'vram_total_mb': vram_total_mb,
                    'vram_percent': vram_percent,
                })
        except Exception as e:
            out['nvml_error'] = str(e)
    else:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        try:
                            total_mb = float(parts[4])
                            used_mb = float(parts[3])
                            vram_pct = round(100.0 * used_mb / total_mb, 1) if total_mb > 0 else None
                        except (ValueError, IndexError):
                            used_mb = total_mb = None
                            vram_pct = None
                        out['gpus'].append({
                            'index': int(parts[0]) if parts[0].isdigit() else len(out['gpus']),
                            'name': parts[1],
                            'gpu_util_percent': int(parts[2]) if parts[2].isdigit() else None,
                            'vram_used_mb': used_mb,
                            'vram_total_mb': total_mb,
                            'vram_percent': vram_pct,
                        })
        except Exception as e:
            out['nvidia_smi_error'] = str(e)
    return out


@app.route('/api/system-stats')
def api_system_stats():
    return jsonify(_get_system_stats())


# ---------- API: Health ----------

@app.route('/api/system/network-probe')
def api_system_network_probe():
    host = (request.args.get('host') or '').strip()
    port = int(request.args.get('port') or 554)
    if not host:
        return jsonify({'error': 'host query parameter is required'}), 400
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return jsonify({'error': f'Invalid host: {host}'}), 400
    probe = _probe_tcp_endpoint(host, port, timeout=float(request.args.get('timeout') or 3.0))
    return jsonify({
        'host': host,
        'port': port,
        'runtime': _runtime_network_info or _runtime_network_context(),
        'probe': probe,
    })


@app.route('/api/health')
def api_health():
    health = {
        'uptime_seconds': round(time.time() - startup_time, 1),
        'runtime_network': _runtime_network_info or _runtime_network_context(),
        'engines': {},
        'cameras': {},
        'streams_total': len(camera_readers),
    }
    for name, eng in _engines.items():
        health['engines'][name] = {
            'running': eng.running,
            'heartbeat_age_s': round(time.time() - eng.heartbeat, 1),
            'total_processed': eng.total_processed,
            'errors': eng.errors,
            'registered_streams': eng.stream_count,
        }
    for cid, reader in camera_readers.items():
        health['cameras'][cid] = {
            'capture_status': reader.status,
            'headcount': cid in headcount_procs,
            'entryexit': cid in entryexit_procs,
            'flapgate': cid in flapgate_procs,
            'ppe': cid in ppe_procs,
            'fire_smoke': cid in fire_smoke_procs,
            'anpr': cid in anpr_procs,
        }
    return jsonify(health)


# ---------- API: Stats & History ----------

@app.route('/api/stats')
def api_all_stats():
    result = {'headcount': {}, 'entryexit': {}, 'flapgate': {}, 'ppe': {}, 'firesmoke': {}, 'anpr': {}}
    for cid, proc in headcount_procs.items():
        result['headcount'][cid] = proc.stats
    for cid, proc in entryexit_procs.items():
        result['entryexit'][cid] = proc.stats
    for cid, proc in flapgate_procs.items():
        result['flapgate'][cid] = proc.stats
    for cid, proc in ppe_procs.items():
        result['ppe'][cid] = proc.stats
    for cid, proc in fire_smoke_procs.items():
        result['firesmoke'][cid] = proc.stats
    for cid, proc in anpr_procs.items():
        result['anpr'][cid] = proc.stats
    return jsonify(result)


@app.route('/api/history/headcount/<camera_id>')
def api_headcount_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT timestamp, current_count, total_entries FROM headcounts WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/history/entryexit/<camera_id>')
def api_entryexit_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT timestamp, entries, exits, current_inside FROM entryexit_counts WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/history/flapgate/<camera_id>')
def api_flapgate_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT timestamp, trespassing_count, persons_in_frame, gate1_status, gate2_status, gate3_status FROM flapgate_events WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/history/ppe/<camera_id>')
def api_ppe_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT timestamp, compliant_count, violation_count, total_persons FROM ppe_events WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/history/firesmoke/<camera_id>')
def api_firesmoke_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        'SELECT timestamp, fire_count, smoke_count, total_detections FROM fire_smoke_events WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/history/anpr/<camera_id>')
def api_anpr_history(camera_id):
    conn = get_db()
    rows = conn.execute(
        '''SELECT timestamp, track_id, vehicle_type, license_plate, speed_kmh, speed_threshold_kmh, overspeeding, confidence
           FROM anpr_events WHERE camera_id=? ORDER BY timestamp DESC LIMIT 100''',
        (camera_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


# ---------- CSV Export (sheet-friendly) ----------

def _rows_to_csv(headers, rows_of_lists):
    """Produce CSV string with headers; safe for Excel/Google Sheets."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(headers)
    for row in rows_of_lists:
        w.writerow([str(x) if x is not None else '' for x in row])
    return buf.getvalue()


def _csv_response(csv_str, filename):
    """Return Flask Response with UTF-8 BOM for Excel compatibility."""
    body = '\ufeff' + (csv_str if isinstance(csv_str, str) else csv_str.decode('utf-8'))
    return Response(body, mimetype='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename={filename}'})


@app.route('/api/export/headcount.csv')
def api_export_headcount_csv():
    camera_id = request.args.get('camera_id', '').strip()
    limit = min(int(request.args.get('limit', 10000)), 100000)
    conn = get_db()
    if camera_id:
        rows = conn.execute('''
            SELECT h.id, h.camera_id, COALESCE(c.name, h.camera_id) AS camera_name, h.timestamp, h.current_count, h.total_entries
            FROM headcounts h LEFT JOIN cameras c ON h.camera_id = c.camera_id
            WHERE h.camera_id = ? ORDER BY h.timestamp DESC LIMIT ?
        ''', (camera_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT h.id, h.camera_id, COALESCE(c.name, h.camera_id) AS camera_name, h.timestamp, h.current_count, h.total_entries
            FROM headcounts h LEFT JOIN cameras c ON h.camera_id = c.camera_id
            ORDER BY h.timestamp DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    headers = ['id', 'camera_id', 'camera_name', 'timestamp', 'current_count', 'total_entries']
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_headcount.csv')


@app.route('/api/export/entryexit.csv')
def api_export_entryexit_csv():
    camera_id = request.args.get('camera_id', '').strip()
    limit = min(int(request.args.get('limit', 10000)), 100000)
    conn = get_db()
    if camera_id:
        rows = conn.execute('''
            SELECT e.id, e.camera_id, COALESCE(c.name, e.camera_id) AS camera_name, e.timestamp, e.entries, e.exits, e.current_inside
            FROM entryexit_counts e LEFT JOIN cameras c ON e.camera_id = c.camera_id
            WHERE e.camera_id = ? ORDER BY e.timestamp DESC LIMIT ?
        ''', (camera_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT e.id, e.camera_id, COALESCE(c.name, e.camera_id) AS camera_name, e.timestamp, e.entries, e.exits, e.current_inside
            FROM entryexit_counts e LEFT JOIN cameras c ON e.camera_id = c.camera_id
            ORDER BY e.timestamp DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    headers = ['id', 'camera_id', 'camera_name', 'timestamp', 'entries', 'exits', 'current_inside']
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_entryexit.csv')


@app.route('/api/export/flapgate.csv')
def api_export_flapgate_csv():
    camera_id = request.args.get('camera_id', '').strip()
    limit = min(int(request.args.get('limit', 10000)), 100000)
    conn = get_db()
    if camera_id:
        rows = conn.execute('''
            SELECT f.id, f.camera_id, COALESCE(c.name, f.camera_id) AS camera_name, f.timestamp, f.trespassing_count, f.persons_in_frame, f.gate1_status, f.gate2_status, f.gate3_status
            FROM flapgate_events f LEFT JOIN cameras c ON f.camera_id = c.camera_id
            WHERE f.camera_id = ? ORDER BY f.timestamp DESC LIMIT ?
        ''', (camera_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT f.id, f.camera_id, COALESCE(c.name, f.camera_id) AS camera_name, f.timestamp, f.trespassing_count, f.persons_in_frame, f.gate1_status, f.gate2_status, f.gate3_status
            FROM flapgate_events f LEFT JOIN cameras c ON f.camera_id = c.camera_id
            ORDER BY f.timestamp DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    headers = ['id', 'camera_id', 'camera_name', 'timestamp', 'trespassing_count', 'persons_in_frame', 'gate1_status', 'gate2_status', 'gate3_status']
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_flapgate.csv')


@app.route('/api/export/ppe.csv')
def api_export_ppe_csv():
    camera_id = request.args.get('camera_id', '').strip()
    limit = min(int(request.args.get('limit', 10000)), 100000)
    conn = get_db()
    if camera_id:
        rows = conn.execute('''
            SELECT p.id, p.camera_id, COALESCE(c.name, p.camera_id) AS camera_name, p.timestamp, p.compliant_count, p.violation_count, p.total_persons
            FROM ppe_events p LEFT JOIN cameras c ON p.camera_id = c.camera_id
            WHERE p.camera_id = ? ORDER BY p.timestamp DESC LIMIT ?
        ''', (camera_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT p.id, p.camera_id, COALESCE(c.name, p.camera_id) AS camera_name, p.timestamp, p.compliant_count, p.violation_count, p.total_persons
            FROM ppe_events p LEFT JOIN cameras c ON p.camera_id = c.camera_id
            ORDER BY p.timestamp DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    headers = ['id', 'camera_id', 'camera_name', 'timestamp', 'compliant_count', 'violation_count', 'total_persons']
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_ppe.csv')


@app.route('/api/export/anpr.csv')
def api_export_anpr_csv():
    camera_id = request.args.get('camera_id', '').strip()
    limit = min(int(request.args.get('limit', 10000)), 100000)
    conn = get_db()
    if camera_id:
        rows = conn.execute('''
            SELECT a.id, a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name, a.timestamp,
                   a.track_id, a.vehicle_type, a.license_plate, a.speed_kmh,
                   a.speed_threshold_kmh, a.overspeeding, a.confidence
            FROM anpr_events a LEFT JOIN cameras c ON a.camera_id = c.camera_id
            WHERE a.camera_id = ? ORDER BY a.timestamp DESC LIMIT ?
        ''', (camera_id, limit)).fetchall()
    else:
        rows = conn.execute('''
            SELECT a.id, a.camera_id, COALESCE(c.name, a.camera_id) AS camera_name, a.timestamp,
                   a.track_id, a.vehicle_type, a.license_plate, a.speed_kmh,
                   a.speed_threshold_kmh, a.overspeeding, a.confidence
            FROM anpr_events a LEFT JOIN cameras c ON a.camera_id = c.camera_id
            ORDER BY a.timestamp DESC LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()
    headers = [
        'id', 'camera_id', 'camera_name', 'timestamp', 'track_id', 'vehicle_type',
        'license_plate', 'speed_kmh', 'speed_threshold_kmh', 'overspeeding', 'confidence'
    ]
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_anpr.csv')


@app.route('/api/export/cameras.csv')
def api_export_cameras_csv():
    conn = get_db()
    rows = conn.execute('''
        SELECT camera_id, name, rtsp_url, added_time, status FROM cameras ORDER BY added_time DESC
    ''').fetchall()
    conn.close()
    headers = ['camera_id', 'name', 'rtsp_url', 'added_time', 'status']
    data = [[r[h] for h in headers] for r in rows]
    return _csv_response(_rows_to_csv(headers, data), 'vision_ai_cameras.csv')


# ---------- Video Feeds ----------

@app.route('/video_feed/raw/<camera_id>')
def feed_raw(camera_id):
    return Response(_gen_raw(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/headcount/<camera_id>')
def feed_headcount(camera_id):
    return Response(_gen_headcount(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/entryexit/<camera_id>')
def feed_entryexit(camera_id):
    return Response(_gen_entryexit(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/flapgate/<camera_id>')
def feed_flapgate(camera_id):
    return Response(_gen_flapgate(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/ppe/<camera_id>')
def feed_ppe(camera_id):
    return Response(_gen_ppe(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/firesmoke/<camera_id>')
def feed_firesmoke(camera_id):
    return Response(_gen_fire_smoke(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/anpr/<camera_id>')
def feed_anpr(camera_id):
    return Response(_gen_anpr(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/beta/<camera_id>')
def feed_beta(camera_id):
    return Response(_gen_beta(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/fr/<camera_id>')
def feed_fr(camera_id):
    return Response(_gen_fr(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/fr_collect/<session_id>')
def feed_fr_collect(session_id):
    return Response(_gen_fr_collect(session_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/workforce/<machine_id>')
def feed_workforce(machine_id):
    if machine_id not in WORKFORCE_MACHINE_IDS:
        return jsonify({'error': 'Invalid machine id'}), 400
    ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
    return Response(_gen_workforce(machine_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/ups/<panel_id>')
def feed_ups(panel_id):
    return Response(_gen_ups(panel_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------- SocketIO: periodic stats push ----------

def _stats_emitter():
    while True:
        try:
            for cid, proc in list(headcount_procs.items()):
                socketio.emit('headcount_stats', {'camera_id': cid, 'stats': proc.stats})
            for cid, proc in list(entryexit_procs.items()):
                socketio.emit('entryexit_stats', {'camera_id': cid, 'stats': proc.stats})
            for cid, proc in list(flapgate_procs.items()):
                socketio.emit('flapgate_stats', {'camera_id': cid, 'stats': proc.stats})
            for cid, proc in list(ppe_procs.items()):
                socketio.emit('ppe_stats', {'camera_id': cid, 'stats': proc.stats})
            for cid, proc in list(fire_smoke_procs.items()):
                socketio.emit('firesmoke_stats', {'camera_id': cid, 'stats': proc.stats})
            for cid, proc in list(anpr_procs.items()):
                socketio.emit('anpr_stats', {'camera_id': cid, 'stats': proc.stats})
            socketio.emit('workforce_stats', get_aggregate_status())
            socketio.emit('ups_stats', {
                'panel': get_panel_status('UPS-PANEL-01'),
                'summary': ups_get_summary(),
            })
        except Exception:
            pass
        time.sleep(2)


# ---------- Startup: restore cameras from DB ----------

def restore_from_db():
    conn = get_db()
    cams = conn.execute("SELECT * FROM cameras WHERE status='active'").fetchall()
    for cam in cams:
        cid = cam['camera_id']
        reader = CameraReader(cid, cam['rtsp_url'])
        camera_readers[cid] = reader
        reader.start()
        logger.info(f"Restored camera: {cid}")

    time.sleep(2)

    configs = conn.execute("SELECT * FROM detection_configs WHERE enabled=1").fetchall()
    for cfg in configs:
        cid = cfg['camera_id']
        dtype = cfg['detection_type']
        config = json.loads(cfg['config_json']) if cfg['config_json'] else {}

        if cid not in camera_readers:
            continue

        if dtype == 'headcount':
            conf = config.get('confidence', 0.1)
            engine = get_inference_engine('headcount', HEADCOUNT_MODEL_PATH)
            proc = HeadCountProcessor(cid, camera_readers[cid], engine, conf)
            headcount_procs[cid] = proc
            proc.start()
            logger.info(f"Restored headcount for {cid}")

        elif dtype == 'entryexit':
            entry_z = [tuple(p) for p in config.get('entry_zone', [])]
            exit_z = [tuple(p) for p in config.get('exit_zone', [])]
            cw = config.get('canvas_width', 800)
            ch = config.get('canvas_height', 450)
            conf = config.get('confidence', 0.5)
            if len(entry_z) >= 3 and len(exit_z) >= 3:
                engine = get_inference_engine('entryexit', ENTRYEXIT_MODEL_PATH, classes=[0])
                proc = EntryExitProcessor(cid, camera_readers[cid], entry_z, exit_z, (cw, ch), engine, conf)
                entryexit_procs[cid] = proc
                proc.start()
                logger.info(f"Restored entryexit for {cid}")

        elif dtype == 'flapgate':
            gate_zones_raw = config.get('gate_zones', {})
            cw = config.get('canvas_width', 800)
            ch = config.get('canvas_height', 450)
            conf = config.get('confidence', 0.65)
            gate_zones = {}
            for zid_str, pts in gate_zones_raw.items():
                gate_zones[int(zid_str)] = [tuple(p) for p in pts]
            if len(gate_zones) >= 3 and all(len(pts) >= 3 for pts in gate_zones.values()):
                engine = get_inference_engine('flapgate', FLAPGATE_MODEL_PATH, classes=[0])
                proc = FlapGateProcessor(cid, camera_readers[cid], gate_zones, (cw, ch), engine, conf)
                flapgate_procs[cid] = proc
                proc.start()
                logger.info(f"Restored flapgate for {cid}")

        elif dtype == 'ppe':
            conf = config.get('confidence', 0.4)
            engine = get_inference_engine('ppe', PPE_MODEL_PATH, classes=None)
            proc = PPEDetectionProcessor(cid, camera_readers[cid], engine, conf)
            ppe_procs[cid] = proc
            proc.start()
            logger.info(f"Restored PPE detection for {cid}")

        elif dtype == 'firesmoke':
            conf = config.get('confidence', 0.35)
            engine = get_inference_engine('firesmoke', FIRE_SMOKE_MODEL_PATH, classes=None)
            proc = FireSmokeDetectionProcessor(cid, camera_readers[cid], engine, conf)
            fire_smoke_procs[cid] = proc
            proc.start()
            logger.info(f"Restored Fire/Smoke detection for {cid}")

        elif dtype == 'anpr':
            conf = config.get('confidence', ANPR_VEHICLE_CONF)
            speed_threshold = config.get('speed_threshold_kmh', 50)
            meters_per_pixel = config.get('meters_per_pixel', ANPR_METERS_PER_PIXEL)
            proc = ANPRDetectionProcessor(
                cid, camera_readers[cid],
                conf=conf, speed_threshold_kmh=speed_threshold,
                meters_per_pixel=meters_per_pixel,
            )
            anpr_procs[cid] = proc
            proc.start()
            logger.info(f"Restored ANPR detection for {cid}")

    # Restore Beta (OWLv2 or Florence-2) detection from beta_detection_prompts + beta_settings
    beta_backend = 'owlv2'
    default_conf = 0.2
    try:
        srow = conn.execute('SELECT model, confidence FROM beta_settings WHERE id=1').fetchone()
        if srow:
            if srow['model']:
                beta_backend = str(srow['model']).strip().lower() or 'owlv2'
            if srow['confidence'] is not None:
                default_conf = float(srow['confidence'])
    except sqlite3.OperationalError:
        pass
    beta_rows = conn.execute(
        'SELECT prompt_text, camera_ids, confidence FROM beta_detection_prompts WHERE enabled=1'
    ).fetchall()
    camera_to_prompts = {}
    for r in beta_rows:
        try:
            camera_ids = json.loads(r['camera_ids']) if r['camera_ids'] else []
        except Exception:
            camera_ids = []
        conf = float(r['confidence']) if r['confidence'] is not None else default_conf
        default_conf = conf
        pt = (r['prompt_text'] or '').strip()
        if not pt:
            continue
        for cid in camera_ids:
            if cid not in camera_to_prompts:
                camera_to_prompts[cid] = []
            camera_to_prompts[cid].append(pt)
    can_restore = (
        (beta_backend == 'owlv2' and _OWLV2_AVAILABLE)
        or (beta_backend == 'florence2' and _florence2_ready())
        or (beta_backend == 'qwen' and _qwen25vl_ready())
    )
    if can_restore and camera_to_prompts:
        for cid, prompt_texts in camera_to_prompts.items():
            if cid not in camera_readers:
                continue
            proc = _make_beta_processor(cid, camera_readers[cid], prompt_texts, conf=default_conf, backend=beta_backend)
            beta_procs[cid] = proc
            proc.start()
            logger.info(f"Restored Beta ({beta_backend}) detection for {cid}")

    # Workforce: bind bundled videos and start display-only playback (no detection)
    try:
        bind_workforce_videos_from_folder(get_db, WORKFORCE_VIDEOS_DIR)
        for mid in list(workforce_procs.keys()):
            wf_stop_machine(mid)
        ensure_workforce_playback(WORKFORCE_VIDEOS_DIR)
        logger.info('Workforce playback-only: file readers started for live view')
    except Exception as e:
        logger.error(f'Workforce bind error: {e}')

    try:
        restore_ups(get_db, _save_alert_event, UPS_VIDEOS_DIR)
    except Exception as e:
        logger.error(f'UPS panel restore error: {e}')

    try:
        start_dashboard_persistence(
            get_db,
            _workforce_dashboard_status,
            get_utilization_chart,
            get_ppe_stats,
            lambda: {'panel': get_panel_status('UPS-PANEL-01'), 'summary': ups_get_summary()},
            ups_get_trend_data,
        )
    except Exception as e:
        logger.error(f'Module dashboard persistence error: {e}')

    conn.close()


# Fail loud in logs if this process was started from an old build (common cause of 404 on /login, /system).
def _missing_auth_routes():
    need = ('/login', '/api/login', '/api/logout', '/system')
    rules = {r.rule for r in app.url_map.iter_rules()}
    return [p for p in need if p not in rules]


_AUTH_ROUTE_ISSUES = _missing_auth_routes()
if _AUTH_ROUTE_ISSUES:
    logger.error(
        'Missing auth routes %s — this Python process does not include the latest app.py routes. '
        'Rebuild the Docker image and restart (e.g. docker compose build --no-cache && docker compose up -d), '
        'or redeploy app.py + templates folder to the server.',
        _AUTH_ROUTE_ISSUES,
    )


# ==============================================================
# MAIN
# ==============================================================

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


if __name__ == '__main__':
    _log_runtime_network_bootstrap()
    restore_thread = threading.Thread(target=restore_from_db, daemon=True)
    restore_thread.start()

    threading.Thread(target=_preload_workforce_models, daemon=True, name='WF-ModelPreload').start()

    stats_thread = threading.Thread(target=_stats_emitter, daemon=True)
    stats_thread.start()

    CUDACacheSteward.start()

    def _start_watchdog():
        time.sleep(5)
        for eng in _engines.values():
            watchdog.register(eng)
        watchdog.start()

    threading.Thread(target=_start_watchdog, daemon=True).start()

    port = int(os.environ['PORT']) if os.environ.get('PORT') else find_free_port()
    host = get_local_ip()
    base = f"http://{host}:{port}"
    host_port = os.environ.get('HOST_PORT')
    host_name = os.environ.get('HOST_NAME', 'localhost')
    print(f"\n{'='*55}")
    print(f"  Resource tuning: imgsz={INFERENCE_IMGSZ} fps={TARGET_FPS} batch={MICRO_BATCH} fp16={USE_FP16} tensorrt={USE_TENSORRT_IF_AVAILABLE} workers={CALLBACK_POOL_WORKERS}")
    print(f"  Vision AI Platform running at: {base}")
    if host_port:
        external_base = f"http://{host_name}:{host_port}"
        print(f"  >>> On this server:  {external_base}/dashboard")
        print(f"  >>> From another PC: http://<SERVER_IP>:{host_port}/dashboard  (use this machine's IP)")
    else:
        print(f"  Dashboard:   {base}/dashboard")
    print(f"  AI Config:   {base}/ai-config")
    print(f"  Sign in:     {base}/login  (default admin user/password from env or admin/admin on first run)")
    print(f"  Health:      {base}/api/health")
    print(f"  Module chat: {base}/api/module-chat  (POST)")
    print(f"  System Stats (GPU/CPU/VRAM): {base}/api/system-stats")
    if _AUTH_ROUTE_ISSUES:
        print(f"  WARNING: Auth routes missing: {', '.join(_AUTH_ROUTE_ISSUES)} — rebuild image / redeploy app.py (see log above).")
    else:
        print(f"  Auth routes OK: /login, /system, /api/logout")
    print(f"{'='*55}\n")
    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True, use_reloader=False)
