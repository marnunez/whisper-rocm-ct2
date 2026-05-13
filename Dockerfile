# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ghcr.io/marnunez/whisper-rocm-ct2-base:rocm7.2.3-ct2v4.7.1-gfx1100
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV WHISPER_MODEL=large-v3-turbo
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV WHISPER_TASK=transcribe
ENV WHISPER_BEAM_SIZE=5
ENV WHISPER_BEST_OF=5
ENV WHISPER_TEMPERATURE=0.0
ENV HOST=0.0.0.0
ENV PORT=8080

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

WORKDIR /app
COPY app /app/app
EXPOSE 8080
CMD ["python3", "-m", "app"]
