# Vision AI Platform - GPU (NVIDIA RTX 6000) with TensorRT support
# Base: NVIDIA PyTorch container (CUDA 12.4, PyTorch, TensorRT, Python 3.10)
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /app

# Install system deps for OpenCV and other pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Remove base image's OpenCV completely to avoid cv2.dnn.DictValue conflict.
# Base image may install cv2 in a way pip uninstall doesn't remove, so delete it.
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.10/dist-packages/cv2* \
    && rm -rf /usr/local/lib/python3.10/site-packages/cv2* 2>/dev/null || true

# Install OpenCV first (version known to avoid DictValue bug), then rest of deps
RUN pip install --no-cache-dir "opencv-python-headless==4.8.0.74" \
    && pip install --no-cache-dir -r requirements.txt

# Optional Qwen fast-path deps (CUDA kernels).
# If these build/install successfully, Qwen3.5 can use faster kernels instead of torch fallback.
RUN pip install --no-cache-dir causal-conv1d flash-linear-attention || true

# Copy application code, templates, and static assets (logo, etc.)
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Copy models from project models/ into /app (app expects .pt/.engine in BASE_DIR)
COPY models/ /app/

# Data dir: bind-mount host project `data` to /app/data (vision_ai.db, alerts/, odd_uploads/)
RUN mkdir -p /app/data

# Environment: fixed port for Docker, tuning defaults (override via docker-compose or -e)
ENV VISION_DATA_DIR=/app/data
ENV PORT=5000
ENV VISION_INFERENCE_IMGSZ=480
ENV VISION_TARGET_FPS=9
ENV VISION_MICRO_BATCH=8
ENV VISION_USE_FP16=1
ENV VISION_USE_TENSORRT=1
ENV VISION_CALLBACK_POOL_WORKERS=12
ENV VISION_OWLV2_LOCAL_PATH=/models/owlv2/
ENV VISION_FIRE_SMOKE_MODEL_PATH=/app/firensmoke.pt
ENV VISION_ANPR_PLATE_MODEL_PATH=/app/ANPRlib.pt
ENV VISION_ANPR_VEHICLE_MODEL_PATH=/app/LibreYOLO9s.pt
ENV VISION_FLORENCE2_LOCAL_PATH=/models/florence2large/Florence-2-large
ENV VISION_QWEN25VL_LOCAL_PATH=/models/qwen25vl/Qwen2.5-VL-3B-Instruct
ENV VISION_QWEN_DEVICE=cuda
ENV VISION_QWEN_UNLOAD_AFTER_USE=0
ENV VISION_QWEN_MIN_PIXELS=200704
ENV VISION_QWEN_MAX_PIXELS=1003520
ENV VISION_QWEN_IMAGE_MAX_SIDE=1024
ENV VISION_QWEN_GPU_MIN_FREE_GB=20
ENV VISION_QWEN_CHAT_MAX_NEW_TOKENS=220
ENV VISION_QWEN_USE_4BIT=0
ENV VISION_VLM_EVICT_OTHERS=1
ENV VISION_QWEN_ALLOW_GPU_WITH_BETA=1
ENV VISION_CAMERA_SCAN_SUBNET=172.30.121.0/24
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 5000

CMD ["python", "app.py"]
