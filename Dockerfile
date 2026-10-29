# GPU-enabled base image; confirm this tag is available in RunPod before building.
FROM runpod/pytorch:2.4.0-py3.10-cuda12.1.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    VOICECHATGLM_MODEL_ID=themodelhq/voicechatglm-qwen3-8b \
    VOICECHATGLM_MODEL_REVISION=9f25d05035bfda0de9b36dfc5911ddcdb378f3c8 \
    MAX_NEW_TOKENS=1200 \
    MAX_ALLOWED_NEW_TOKENS=1200

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY handler.py inference.py ./

# RunPod Serverless worker protocol (not a public FastAPI web server).
CMD ["python", "handler.py"]
