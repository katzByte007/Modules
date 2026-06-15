# Modules

Lightweight **Sutek Vision** deployment for **PythonAnywhere** (low disk). Focus:

- **Login** — `/login`
- **Modules** — Workforce & Panel dashboards (`/modules`)
- **Alerts** — `/alerts`
- **System** — users, email, engineers (`/system`)
- **SQLite database** — auto-created under `data/vision_ai.db`

Other pages (dashboard, AI config, plant map, etc.) ship as **HTML templates only** — no heavy AI stack required.

## PythonAnywhere setup

1. Clone this repo into your PA project folder.
2. Create a virtualenv and install **minimal** deps (~100 MB):

```bash
pip install --user --no-cache-dir -r requirements_pa.txt
```

3. Set environment variables (PA **Web → WSGI** or `.env`):

```bash
VISION_LITE=1
VISION_SECRET_KEY=change-me-to-a-long-random-string
```

4. Upload workforce videos manually (optional):

```
data/workforce_videos/MACHINE-01_....mp4
...
```

5. WSGI entry (example):

```python
import sys
import os
path = '/home/YOURUSER/Modules'
if path not in sys.path:
    sys.path.insert(0, path)
os.environ['VISION_LITE'] = '1'
from app import app as application
```

6. Reload the web app. Sign in: **admin** / **admin** (change in System → Users).

## What is excluded from git (save disk)

- `requirements.txt`, Docker, model weights (`.pt`), `videos.zip`
- Runtime DB and uploaded videos

## Local run (lite)

```bash
pip install -r requirements_pa.txt
set VISION_LITE=1
python app.py
```

Open `http://127.0.0.1:PORT/login` then `/modules`.

## Full AI stack (optional, not for PA free tier)

Install `requirements.txt` + model weights locally or on a GPU server. Unset `VISION_LITE` or set `VISION_LITE=0`.
