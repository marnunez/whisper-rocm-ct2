# whisper-rocm-ct2

OpenAI-compatible Whisper transcription server for AMD ROCm using CTranslate2/faster-whisper.

This image exists because the stock faster-whisper/CTranslate2 wheels target CPU/CUDA, while AMD iGPU/APU inference needs a HIP/ROCm CTranslate2 build compiled against the same ROCm runtime used in the container.

Public image:

```bash
docker pull ghcr.io/marnunez/whisper-rocm-ct2:latest
```

## Goals

- AMD ROCm/HIP acceleration on `/dev/kfd` + `/dev/dri`
- CTranslate2 built with upstream `WITH_HIP=ON`
- `faster-whisper` service API
- OpenAI-compatible `POST /v1/audio/transcriptions`
- Simple `POST /transcribe` endpoint for `whisper-to-me`
- Runtime tunables through environment variables

## ROCm target

This image is intended for AMD ROCm/HIP-capable hosts where the GPU is exposed to the container, typically through `/dev/kfd` and `/dev/dri`.

Defaults:

- GPU visible as CTranslate2/faster-whisper `device="cuda"` when using the ROCm/HIP backend
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`, useful for some unsupported or partially supported AMD APUs
- model: `large-v3-turbo`
- compute type: `float16`

The word `cuda` in CTranslate2/faster-whisper is API compatibility naming. In this ROCm build it maps to HIP/ROCm, not NVIDIA CUDA.

## API

### Health

```bash
curl http://localhost:8080/health
```

### Simple transcription

```bash
curl http://localhost:8080/transcribe -F file=@audio.wav
```

Response:

```json
{
  "filename": "audio.wav",
  "duration_seconds": 7.56,
  "language": "en",
  "text": "..."
}
```

### OpenAI-compatible transcription

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-1 \
  -F response_format=json
```

## Runtime configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name/path |
| `WHISPER_DEVICE` | `cuda` | CTranslate2 device (`cuda` for ROCm HIP build) |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8`, etc. |
| `WHISPER_LANGUAGE` | empty | Force language, e.g. `en`/`es`; empty = auto |
| `WHISPER_TASK` | `transcribe` | `transcribe` or `translate` |
| `WHISPER_BEAM_SIZE` | `5` | Beam size |
| `WHISPER_BEST_OF` | `5` | Number of candidates for non-zero-temperature sampling |
| `WHISPER_TEMPERATURE` | `0.0` | Decoding temperature |
| `WHISPER_VAD_FILTER` | `true` | Enable faster-whisper VAD |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | faster-whisper condition-on-previous-text |
| `WHISPER_NO_SPEECH_THRESHOLD` | `0.6` | Whisper no-speech threshold |
| `WHISPER_LOG_PROB_THRESHOLD` | `-1.0` | Average log-probability failure threshold |
| `WHISPER_COMPRESSION_RATIO_THRESHOLD` | `2.4` | Compression-ratio failure threshold |
| `WHISPER_HALLUCINATION_SILENCE_THRESHOLD` | unset | Optional silence threshold for hallucination suppression |
| `WHISPER_HOTWORDS` | empty | Optional context/hotwords string |
| `HOST` | `0.0.0.0` | Uvicorn bind host |
| `PORT` | `8080` | Uvicorn port |

## Docker compose sketch

```yaml
services:
  whisper-rocm-ct2:
    image: ghcr.io/marnunez/whisper-rocm-ct2:latest
    container_name: whisper-rocm-ct2
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - WHISPER_MODEL=large-v3-turbo
      - WHISPER_DEVICE=cuda
      - WHISPER_COMPUTE_TYPE=float16
      - WHISPER_TASK=transcribe
      - WHISPER_BEAM_SIZE=5
      - WHISPER_BEST_OF=5
      - WHISPER_TEMPERATURE=0.0
      - HF_HOME=/root/.cache/huggingface
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    security_opt:
      - seccomp=unconfined
    volumes:
      - whisper-cache:/root/.cache/huggingface

volumes:
  whisper-cache:
```

## Build and publish

GitHub Actions builds and publishes the public image to GHCR on pushes to `main` using GitHub's built-in `GITHUB_TOKEN`; no personal access token is needed.

The Dockerfile builds upstream CTranslate2 `v4.7.1` from `OpenNMT/CTranslate2` against `rocm/dev-ubuntu-24.04:7.2.3`. This is expected to be slow and storage-hungry.

To build locally:

```bash
docker build \
  --build-arg ROCM_ARCHS=gfx1100 \
  -t whisper-rocm-ct2:dev .
```

Build arguments:

| Argument | Default | Meaning |
|---|---|---|
| `CT2_REPO` | `https://github.com/OpenNMT/CTranslate2.git` | CTranslate2 source repo |
| `CT2_REF` | `v4.7.1` | Git ref to build |
| `ROCM_ARCHS` | `gfx1100` | HIP architectures to compile |

For unsupported or partially supported APUs, `gfx1100` plus `HSA_OVERRIDE_GFX_VERSION=11.0.0` can be a practical first target. If your hardware has native ROCm support for a different architecture, set `ROCM_ARCHS` accordingly.

## Notes

This image avoids the `libhipblas` / `libamdhip64` ABI mismatch that can happen when mixing random CTranslate2 ROCm wheels with unrelated ROCm images.
