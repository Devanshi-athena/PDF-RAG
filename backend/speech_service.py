from __future__ import annotations

import requests

from .errors import BackendError


def transcribe(settings, audio_bytes: bytes, content_type: str | None) -> str:
    if not settings.hf_token:
        raise BackendError("HF_TOKEN is required for voice transcription.", 503, "asr_not_configured")
    try:
        response = requests.post(
            f"{settings.asr_url.rstrip('/')}/models/{settings.asr_model}",
            headers={"Authorization": f"Bearer {settings.hf_token}", "Content-Type": content_type or "audio/webm"},
            data=audio_bytes,
            timeout=120,
        )
    except requests.RequestException as error:
        raise BackendError("Could not reach Hugging Face for voice transcription.", 502, "asr_unavailable") from error
    if not response.ok:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(f"Hugging Face transcription failed ({response.status_code}): {detail}", 502, "asr_error")
    try:
        text = response.json().get("text", "").strip()
    except ValueError as error:
        raise BackendError("Hugging Face returned an invalid transcription response.", 502, "asr_error") from error
    if not text:
        raise BackendError("No speech was detected in the recording.", 422, "empty_transcription")
    return text
