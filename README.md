# Dashboardmodule

Sutek Vision AI dashboard with Workforce Monitoring, Panel module, unified analytics, module assistant chatbot, and live video feeds.

## Quick start (Docker)

See **[DOCKER_README.md](DOCKER_README.md)** for GPU Docker build/run, volumes, and environment variables.

```bash
docker compose build
docker compose up -d
```

Open `http://<host>:8080` and sign in.

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # optional overrides
python app.py
```

## Project layout

| Path | Description |
|------|-------------|
| `app.py` | Main Flask application |
| `workforce_monitoring.py` | Workforce live view & playback |
| `ups_panel_monitoring.py` | Panel module |
| `module_dashboard_store.py` | Module assistant DB snapshots |
| `templates/` | Web UI |
| `static/` | Logo and shared CSS |
| `models/` | YOLO weights (copy after clone — see ASSETS_SETUP.md) |
| `data/workforce_videos/` | Workforce demo MP4 feeds (copy after clone) |
| `data/ups_videos/` | Panel demo videos (copy after clone) |
| `data/vision_ai.db` | SQLite database (auto-created on first run) |

## Assets (models & videos)

The git repo is **code-only** (~few MB) so it clones on machines with limited disk quota.  
After clone, copy models and videos from your dev machine — see **[ASSETS_SETUP.md](ASSETS_SETUP.md)**.

## License

Proprietary — internal use unless otherwise specified.
