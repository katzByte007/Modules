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
2. **Free disk first** (PA free tier fills up quickly):

```bash
pip cache purge
rm -rf ~/.cache/pip
du -ah ~ | sort -rh | head -15
```

Remove old virtualenvs or large uploads if needed. **Do not upload workforce MP4s until deps are installed** (videos use extra space).

3. Install deps — pick one:

```bash
# Full lite (~70 MB) — includes workforce video streaming
pip install --user --no-cache-dir -r requirements_pa.txt

# OR core only (~25 MB) — login/alerts/system; add opencv later for video
pip install --user --no-cache-dir -r requirements_pa_core.txt
```

If a previous install failed mid-way:

```bash
pip uninstall -y flask werkzeug numpy opencv-python-headless
pip cache purge
pip install --user --no-cache-dir -r requirements_pa.txt
```

4. Set environment variables (PA **Web → WSGI** or `.env`):

```bash
VISION_LITE=1
VISION_SECRET_KEY=change-me-to-a-long-random-string
```

4. Upload workforce videos manually **after** pip succeeds (optional):

```
data/workforce_videos/MACHINE-01_....mp4
...
```

5. **WSGI configuration** (Web → your Flask app → WSGI configuration file):

**Do not use** `from Modules import app` — `Modules` is only the folder name, not a Python module.

Replace the entire WSGI file with (change `kartik2025` if your PA username differs):

```python
import sys
import os

project_home = '/home/kartik2025/Modules'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)
os.environ['VISION_LITE'] = '1'

from app import app
application = app
```

Reload the web app. If it still fails, check **Web → Log files → error log**.

6. Sign in: **admin** / **admin** (change in System → Users).

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
