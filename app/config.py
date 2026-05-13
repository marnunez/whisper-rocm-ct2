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
    beam_size: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    vad_filter: bool = _bool_env("WHISPER_VAD_FILTER", True)
    condition_on_previous_text: bool = _bool_env(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False
    )
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))


settings = Settings()
