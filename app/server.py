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
OPENAI_PROMPT = Form(None)
OPENAI_TEMPERATURE = Form(None)

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
        "task": settings.task,
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "temperature": settings.temperature,
        "vad_filter": settings.vad_filter,
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


def _transcribe_path(path: Path, **overrides: Any) -> dict[str, Any]:
    model = get_model()
    kwargs: dict[str, Any] = {
        "task": overrides.get("task") or settings.task,
        "beam_size": overrides.get("beam_size") or settings.beam_size,
        "best_of": overrides.get("best_of") or settings.best_of,
        "temperature": overrides.get("temperature")
        if overrides.get("temperature") is not None
        else settings.temperature,
        "vad_filter": overrides.get("vad_filter")
        if overrides.get("vad_filter") is not None
        else settings.vad_filter,
        "condition_on_previous_text": overrides.get("condition_on_previous_text")
        if overrides.get("condition_on_previous_text") is not None
        else settings.condition_on_previous_text,
        "no_speech_threshold": overrides.get("no_speech_threshold")
        if overrides.get("no_speech_threshold") is not None
        else settings.no_speech_threshold,
        "log_prob_threshold": overrides.get("log_prob_threshold")
        if overrides.get("log_prob_threshold") is not None
        else settings.log_prob_threshold,
        "compression_ratio_threshold": overrides.get("compression_ratio_threshold")
        if overrides.get("compression_ratio_threshold") is not None
        else settings.compression_ratio_threshold,
    }

    language = overrides.get("language") or settings.language or None
    if language:
        kwargs["language"] = language

    prompt = overrides.get("initial_prompt") or overrides.get("prompt")
    if prompt:
        kwargs["initial_prompt"] = prompt

    hotwords = overrides.get("hotwords") or settings.hotwords
    if hotwords:
        kwargs["hotwords"] = hotwords

    hallucination_silence_threshold = overrides.get(
        "hallucination_silence_threshold"
    )
    if hallucination_silence_threshold is None:
        hallucination_silence_threshold = settings.hallucination_silence_threshold
    if hallucination_silence_threshold is not None:
        kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold

    if kwargs["vad_filter"]:
        min_silence_duration_ms = overrides.get("min_silence_duration_ms")
        speech_pad_ms = overrides.get("speech_pad_ms")
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": min_silence_duration_ms
            if min_silence_duration_ms is not None
            else 2000,
            "speech_pad_ms": speech_pad_ms if speech_pad_ms is not None else 400,
        }

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
        "Transcribed %s in %.2fs audio=%.2fs language=%s task=%s",
        path.name,
        elapsed,
        duration,
        info.language,
        kwargs["task"],
    )
    return {
        "duration_seconds": round(duration, 2),
        "language": info.language,
        "language_probability": float(info.language_probability or 0.0),
        "text": text,
        "transcription_seconds": round(elapsed, 3),
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = AUDIO_FILE,
    model: str = OPENAI_MODEL,
    task: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    initial_prompt: str | None = Form(None),
    beam_size: int | None = Form(None),
    best_of: int | None = Form(None),
    temperature: float | None = Form(None),
    condition_on_previous_text: bool | None = Form(None),
    vad_filter: bool | None = Form(None),
    min_silence_duration_ms: int | None = Form(None),
    speech_pad_ms: int | None = Form(None),
    no_speech_threshold: float | None = Form(None),
    log_prob_threshold: float | None = Form(None),
    compression_ratio_threshold: float | None = Form(None),
    hallucination_silence_threshold: float | None = Form(None),
    hotwords: str | None = Form(None),
) -> dict[str, Any]:
    del model  # Model is configured server-side.
    path = await _save_upload(file)
    try:
        result = _transcribe_path(
            path,
            task=task,
            language=language,
            prompt=prompt,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )
        return {"filename": file.filename, **result}
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = AUDIO_FILE,
    model: str = OPENAI_MODEL,
    response_format: str = OPENAI_RESPONSE_FORMAT,
    language: str | None = OPENAI_LANGUAGE,
    prompt: str | None = OPENAI_PROMPT,
    temperature: float | None = OPENAI_TEMPERATURE,
    task: str | None = Form(None),
    beam_size: int | None = Form(None),
    best_of: int | None = Form(None),
    condition_on_previous_text: bool | None = Form(None),
    vad_filter: bool | None = Form(None),
    min_silence_duration_ms: int | None = Form(None),
    speech_pad_ms: int | None = Form(None),
    no_speech_threshold: float | None = Form(None),
    log_prob_threshold: float | None = Form(None),
    compression_ratio_threshold: float | None = Form(None),
    hallucination_silence_threshold: float | None = Form(None),
    hotwords: str | None = Form(None),
) -> Any:
    del model  # Model is configured server-side.
    path = await _save_upload(file)
    try:
        result = _transcribe_path(
            path,
            task=task or "transcribe",
            language=language,
            prompt=prompt,
            temperature=temperature,
            beam_size=beam_size,
            best_of=best_of,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )
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


@app.post("/v1/audio/translations")
async def openai_translations(
    file: UploadFile = AUDIO_FILE,
    model: str = OPENAI_MODEL,
    response_format: str = OPENAI_RESPONSE_FORMAT,
    prompt: str | None = OPENAI_PROMPT,
    temperature: float | None = OPENAI_TEMPERATURE,
) -> Any:
    del model
    path = await _save_upload(file)
    try:
        result = _transcribe_path(
            path, task="translate", prompt=prompt, temperature=temperature
        )
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
