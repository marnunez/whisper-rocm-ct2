from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger("whisper-rocm-ct2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="Whisper ROCm CTranslate2 API",
    description="Whisper transcription on AMD ROCm via CTranslate2/faster-whisper",
    version="0.1.0",
)

AUDIO_FILE = File(...)
OPENAI_MODEL = Form("whisper-1")
OPENAI_RESPONSE_FORMAT = Form("json")
OPENAI_LANGUAGE = Form(None)

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(
            "Loading model=%s device=%s compute_type=%s",
            settings.model,
            settings.device,
            settings.compute_type,
        )
        start = time.perf_counter()
        _model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        logger.info("Model loaded in %.2fs", time.perf_counter() - start)
    return _model


@app.on_event("startup")
def load_on_startup() -> None:
    get_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model is not None else "loading",
        "model": settings.model,
        "device": settings.device,
        "compute_type": settings.compute_type,
    }


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        tmp.write(contents)
    return Path(tmp.name)


def _transcribe_path(path: Path, language: str | None = None) -> dict[str, Any]:
    model = get_model()
    kwargs: dict[str, Any] = {
        "beam_size": settings.beam_size,
        "vad_filter": settings.vad_filter,
        "condition_on_previous_text": settings.condition_on_previous_text,
    }
    effective_language = language or settings.language or None
    if effective_language:
        kwargs["language"] = effective_language

    start = time.perf_counter()
    segments, info = model.transcribe(str(path), **kwargs)
    text_parts: list[str] = []
    duration = 0.0
    for segment in segments:
        text_parts.append(segment.text.strip())
        duration = max(duration, float(segment.end))

    elapsed = time.perf_counter() - start
    text = " ".join(part for part in text_parts if part).strip()
    logger.info(
        "Transcribed %s in %.2fs audio=%.2fs language=%s",
        path.name,
        elapsed,
        duration,
        info.language,
    )
    return {
        "duration_seconds": round(duration, 2),
        "language": info.language,
        "language_probability": float(info.language_probability or 0.0),
        "text": text,
        "transcription_seconds": round(elapsed, 3),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = AUDIO_FILE) -> dict[str, Any]:
    path = await _save_upload(file)
    try:
        result = _transcribe_path(path)
        return {"filename": file.filename, **result}
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = AUDIO_FILE,
    model: str = OPENAI_MODEL,
    response_format: str = OPENAI_RESPONSE_FORMAT,
    language: str | None = OPENAI_LANGUAGE,
) -> Any:
    del model  # Model is configured server-side.
    path = await _save_upload(file)
    try:
        result = _transcribe_path(path, language=language)
        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format in {"json", "verbose_json"}:
            return result
        raise HTTPException(
            status_code=400,
            detail="response_format must be one of: json, verbose_json, text",
        )
    finally:
        path.unlink(missing_ok=True)
