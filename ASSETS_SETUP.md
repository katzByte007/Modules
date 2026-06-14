# Assets setup (after git clone)

The git repo contains **application code only** (~few MB) so it clones on machines with limited disk quota.

Large files (models, demo videos, database) are **not** in git. Copy them once from your Windows dev machine or another source.

## 1. Clone (should work now)

```bash
git clone https://github.com/katzByte007/Dashboardmodule.git
cd Dashboardmodule
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Required folders

```bash
mkdir -p data/workforce_videos data/ups_videos models videos
```

## 3. Copy assets from Windows (example with scp)

From your **Linux** machine, pull from the machine that has the full project:

```bash
# Workforce demo videos (6 files)
scp -r user@WINDOWS_IP:"/path/to/Dashboardmodule/data/workforce_videos/"* ./data/workforce_videos/

# Panel video (optional)
scp -r user@WINDOWS_IP:"/path/to/Dashboardmodule/data/ups_videos/"* ./data/ups_videos/

# YOLO / detection weights
scp user@WINDOWS_IP:"/path/to/Dashboardmodule/models/"*.pt ./models/

# Demo clips (optional, for other detection modules)
scp -r user@WINDOWS_IP:"/path/to/Dashboardmodule/videos/"* ./videos/
```

Replace `user@WINDOWS_IP` and paths with your values.

## 4. Minimum for Workforce + Panel modules

| Path | Purpose |
|------|---------|
| `data/workforce_videos/MACHINE-01_*.mp4` … `MACHINE-06_*.mp4` | Live view feeds |
| `data/ups_videos/*.mp4` | Panel live view (optional) |
| `models/best10xemphead.pt` | Headcount / workforce person model |
| `models/bestpharmappepro.pt` or `models/LibreYOLO9s.pt` | PPE model |
| `models/yolov8m.pt` | General YOLO (or auto-downloaded by ultralytics) |

Database `data/vision_ai.db` is created automatically on first run (`python app.py`).

## 5. Run

```bash
cp .env.example .env   # optional
python app.py
```

Open `http://localhost:8080` (or the port shown in the console).

## Disk space

- **Git clone:** ~5–15 MB  
- **Full assets (models + videos):** ~400+ MB additional — keep on disk or network share, not in git.
