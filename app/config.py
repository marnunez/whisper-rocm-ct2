from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    device: str = os.getenv("WHISPER_DEVICE", "cuda")
    compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    language: str = os.getenv("WHISPER_LANGUAGE", "")
    task: str = os.getenv("WHISPER_TASK", "transcribe")
    beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    best_of: int = int(os.getenv("WHISPER_BEST_OF", "5"))
    temperature: float = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
    vad_filter: bool = _bool_env("WHISPER_VAD_FILTER", True)
    condition_on_previous_text: bool = _bool_env(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
    )
    no_speech_threshold: float = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))
    log_prob_threshold: float = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0"))
    compression_ratio_threshold: float = float(
        os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
    )
    hallucination_silence_threshold: float | None = (
        float(os.environ["WHISPER_HALLUCINATION_SILENCE_THRESHOLD"])
        if "WHISPER_HALLUCINATION_SILENCE_THRESHOLD" in os.environ
        else None
    )
    hotwords: str = os.getenv("WHISPER_HOTWORDS", "")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))


settings = Settings()
