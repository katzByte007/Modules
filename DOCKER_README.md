# Vision AI Platform – Docker deploy (GPU)

Run the app in a container on your Linux AI server (e.g. with NVIDIA RTX 6000 and Jupyter). The container uses host networking so RTSP cameras reachable from the AI server are reachable from OpenCV/FFmpeg inside the app with the same routes.

## Prerequisites on the Linux server

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU):

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## What gets copied into the image

- **`static/`** – Copied into the image so the dashboard can serve assets (e.g. **`static/logo.jpeg`** for the Pixecore logo). Ensure `static/logo.jpeg` exists in the project before building.
- **`models/`** – Contents are copied to the app root in the image so the app finds `best10xemphead.pt`, `yolov8m.pt`, and any `.engine` files. Put all model files in the `models/` folder:
  - `models/best10xemphead.pt` (and optionally `models/best10xemphead.engine`)
  - `models/yolov8m.pt` (and optionally `models/yolov8m.engine`)
  - `models/firensmoke.pt`
  - `models/ANPRlib.pt`
  - `models/LibreYOLO9s.pt`
- **`data/`** – Not copied into the image. The app creates **`vision_ai.db`** inside the container at `/app/data` on first run. That directory is persisted via the Docker volume `vision_ai_data`, so the DB is kept across restarts. To start with an existing DB, copy it into the volume after the first run (e.g. `docker cp vision_ai.db vision-ai:/app/data/`).

## Build and run

From the project root (where `Dockerfile` and `docker-compose.yml` are):

```bash
# Build
docker compose build

# Run in background (GPU, host networking, app listens on port 8080)
docker compose up -d

# Logs
docker compose logs -f vision-ai
```

**Open in your browser:** With host networking, the app listens directly on **host port 8080**.

- **If you're on the same Linux server** (browser on the AI server):  
  - **Dashboard:** http://localhost:8080/dashboard  
- **If you're on another PC** (Windows/Mac/Laptop): use the **AI server’s IP**, not localhost:  
  - **Dashboard:** http://\<AI_SERVER_IP\>:8080/dashboard  
  - Example: if the server IP is 192.168.1.10 → http://192.168.1.10:8080/dashboard  

The app uses Docker host networking so RTSP routing matches the AI server host. Always use **port 8080** on the host.

### Links not working?

1. **From another PC:** Do **not** use `localhost` — that is your own machine. Use the Linux server’s IP (e.g. `192.168.x.x` or the server hostname). Find it on the server with: `hostname -I | awk '{print $1}'`.
2. **Firewall:** Allow port 8080 on the AI server:
   ```bash
   sudo ufw allow 8080/tcp
   sudo ufw status
   ```
3. **Test from the server:** `curl -I http://127.0.0.1:8080` should return `HTTP/1.1 200` or `301`. If that works but the browser doesn’t, use the server IP in the browser.

## Run without Compose (docker run)

```bash
docker build -t vision-ai-platform:latest .
docker run -d --gpus all --network host \
  -e PORT=8080 \
  -v vision_ai_data:/app/data \
  --name vision-ai \
  vision-ai-platform:latest
```

## RTSP cameras on a different subnet (e.g. 192.168.8.x lab cameras)

If `172.30.121.x` cameras stream but `192.168.8.x` fails, this is a **network reachability** issue on the AI server, not RTSP auth or password encoding.

### Step 1 — confirm Docker host networking

```bash
docker inspect vision-ai --format 'NetworkMode={{.HostConfig.NetworkMode}}'
```

Expected: `NetworkMode=host`

If not host mode:

```bash
docker compose down
docker compose up -d --force-recreate
```

### Step 2 — test from the AI server itself (most important)

Run this **on the AI server shell**, not inside your laptop:

```bash
python3 -c "import socket; socket.create_connection(('192.168.8.5',554),5); print('host OK')"
ip route get 192.168.8.5
ping -c 3 192.168.8.5
nc -zv -w 5 192.168.8.5 554
```

If these fail on the host, the Vision app on `:8080` cannot work either — even though a notebook test on `:7860` may have worked earlier when routing/VPN was up. Ask your network team to allow/route TCP `554` from the AI server (`172.30.105.x`) to the lab camera subnet (`192.168.8.x`).

Compare with the app probe:

```bash
curl "http://127.0.0.1:8080/api/system/network-probe?host=192.168.8.5&port=554"
```

If host networking is active, host test and app probe should give the same result.

### Step 3 — verify the camera IP

Make sure the saved RTSP URL uses the correct camera IP (`192.168.8.5` vs `192.168.8.9`). A wrong IP will also show `No route to host` or timeout.

### Notes

- Do not use `docker stack deploy` for this service. Swarm ignores `network_mode: host`.
- Startup logs should show `likely_docker_bridge=False` and outbound IP like `172.30.105.x`.

## Tuning (optional)

Override via environment (e.g. in `docker-compose.yml` or `.env`):

- `VISION_INFERENCE_IMGSZ` (default 480)
- `VISION_TARGET_FPS` (default 9)
- `VISION_MICRO_BATCH` (default 8)
- `VISION_USE_FP16` (default 1)
- `VISION_USE_TENSORRT` (default 1)
- `VISION_CALLBACK_POOL_WORKERS` (default 12)

## Data

- SQLite DB and app data are stored in the Docker volume `vision_ai_data`. They persist across container restarts.
- To backup: `docker run --rm -v vision_ai_data:/data -v $(pwd):/backup alpine tar czf /backup/vision_ai_data.tar.gz -C /data .`
