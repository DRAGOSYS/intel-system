"""
Layer 4 Intelligence - LLM Router
Routes queries to the appropriate model based on token count, modality, and task type.

Routing logic:
  - Quick queries < 2000 tokens  → phi3:mini  (local Ollama)
  - Heavy analysis               → Groq llama-3.3-70b-versatile (free tier)
  - Images / video / PDF         → Gemini 1.5 Flash (free tier)
  - Audio                        → faster-whisper tiny (local)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import tiktoken  # pip install tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
QUICK_QUERY_TOKEN_THRESHOLD = 2000  # tokens


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class ModelBackend(str, Enum):
    PHI3_MINI = "phi3:mini"                         # Local Ollama
    GROQ_LLAMA = "llama-3.3-70b-versatile"          # Groq API
    GEMINI_FLASH = "gemini-1.5-flash"               # Google Gemini API
    FASTER_WHISPER = "faster-whisper-tiny"          # Local audio


class TaskType(str, Enum):
    QUICK_QA = "quick_qa"
    HEAVY_ANALYSIS = "heavy_analysis"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"


@dataclass
class RoutingDecision:
    backend: ModelBackend
    task_type: TaskType
    estimated_tokens: int
    reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str
    backend: ModelBackend
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_response: Any = None


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count. Uses tiktoken with cl100k_base as a universal
    approximation (accurate within ~10% for most open models).
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Backend clients (lazy-initialised singletons)
# ---------------------------------------------------------------------------

class _OllamaClient:
    """Thin wrapper around the Ollama REST API (local)."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def complete(
        self,
        system: str,
        user: str,
        model: str = "phi3:mini",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        import requests  # pip install requests

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class _GroqClient:
    """Wrapper around the Groq API."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set — Groq calls will fail.")

    def complete(
        self,
        system: str,
        user: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        from groq import Groq  # pip install groq

        client = Groq(api_key=self.api_key)
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat.choices[0].message.content


class _GeminiClient:
    """Wrapper around the Google Gemini API (supports multimodal)."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set — Gemini calls will fail.")

    def complete_text(
        self,
        system: str,
        user: str,
        model: str = "gemini-1.5-flash",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        import google.generativeai as genai  # pip install google-generativeai

        genai.configure(api_key=self.api_key)
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        resp = m.generate_content(user)
        return resp.text

    def complete_multimodal(
        self,
        system: str,
        parts: list,  # list of genai.Part or str
        model: str = "gemini-1.5-flash",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        resp = m.generate_content(parts)
        return resp.text


class _WhisperClient:
    """Wrapper around faster-whisper for local audio transcription."""

    _model = None  # lazy singleton

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        from faster_whisper import WhisperModel  # pip install faster-whisper

        if self._model is None:
            _WhisperClient._model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
            )
        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            temperature=0.0,
        )
        transcript = " ".join(seg.text.strip() for seg in segments)
        logger.info(
            "Whisper transcribed %.1f s of audio (detected lang=%s)",
            info.duration,
            info.language,
        )
        return transcript


# Singletons
_ollama = _OllamaClient()
_groq = _GroqClient()
_gemini = _GeminiClient()
_whisper = _WhisperClient()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Central routing layer. Decides which backend handles each request
    and returns a normalised LLMResponse.
    """

    def route(
        self,
        system: str,
        user: str,
        task_type: Optional[TaskType] = None,
        force_backend: Optional[ModelBackend] = None,
        # Multimodal extras
        audio_path: Optional[str] = None,
        media_parts: Optional[list] = None,
    ) -> LLMResponse:
        """
        Route and execute an LLM call.

        Args:
            system:        System prompt string.
            user:          User prompt string.
            task_type:     Override auto-detection of task type.
            force_backend: Skip routing and use this backend directly.
            audio_path:    Path to audio file (for AUDIO task type).
            media_parts:   Gemini multimodal parts list (for IMAGE/VIDEO/PDF).

        Returns:
            LLMResponse with content and metadata.
        """
        decision = self._decide(system, user, task_type, force_backend)
        logger.info(
            "Router → %s | task=%s | ~%d tokens | reason: %s",
            decision.backend.value,
            decision.task_type.value,
            decision.estimated_tokens,
            decision.reason,
        )

        start = time.perf_counter()
        content = self._execute(decision, system, user, audio_path, media_parts)
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = count_tokens(system + user)
        completion_tokens = count_tokens(content)

        return LLMResponse(
            content=content,
            backend=decision.backend,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decide(
        self,
        system: str,
        user: str,
        task_type: Optional[TaskType],
        force_backend: Optional[ModelBackend],
    ) -> RoutingDecision:
        if force_backend:
            return RoutingDecision(
                backend=force_backend,
                task_type=task_type or TaskType.QUICK_QA,
                estimated_tokens=0,
                reason="Force override by caller.",
            )

        total_text = (system or "") + (user or "")
        token_count = count_tokens(total_text)

        # Explicit task type takes priority
        if task_type == TaskType.AUDIO:
            return RoutingDecision(
                backend=ModelBackend.FASTER_WHISPER,
                task_type=TaskType.AUDIO,
                estimated_tokens=0,
                reason="Audio task → faster-whisper local.",
            )

        if task_type in (TaskType.IMAGE, TaskType.VIDEO, TaskType.PDF):
            return RoutingDecision(
                backend=ModelBackend.GEMINI_FLASH,
                task_type=task_type,
                estimated_tokens=token_count,
                reason=f"Multimodal task ({task_type.value}) → Gemini 1.5 Flash.",
            )

        if task_type == TaskType.HEAVY_ANALYSIS or token_count >= QUICK_QUERY_TOKEN_THRESHOLD:
            return RoutingDecision(
                backend=ModelBackend.GROQ_LLAMA,
                task_type=TaskType.HEAVY_ANALYSIS,
                estimated_tokens=token_count,
                reason=(
                    f"Token count {token_count} ≥ {QUICK_QUERY_TOKEN_THRESHOLD} "
                    "or heavy analysis task → Groq LLaMA."
                ),
            )

        # Default: quick local model
        return RoutingDecision(
            backend=ModelBackend.PHI3_MINI,
            task_type=TaskType.QUICK_QA,
            estimated_tokens=token_count,
            reason=f"Token count {token_count} < {QUICK_QUERY_TOKEN_THRESHOLD} → phi3:mini local.",
        )

    def _execute(
        self,
        decision: RoutingDecision,
        system: str,
        user: str,
        audio_path: Optional[str],
        media_parts: Optional[list],
    ) -> str:
        backend = decision.backend

        if backend == ModelBackend.FASTER_WHISPER:
            if not audio_path:
                raise ValueError("audio_path must be provided for AUDIO task type.")
            return _whisper.transcribe(audio_path)

        if backend == ModelBackend.GEMINI_FLASH:
            if media_parts:
                return _gemini.complete_multimodal(system, media_parts)
            return _gemini.complete_text(system, user)

        if backend == ModelBackend.GROQ_LLAMA:
            return _groq.complete(system, user)

        # Default: PHI3_MINI via Ollama
        return _ollama.complete(system, user)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

router = LLMRouter()


def route_query(
    system: str,
    user: str,
    task_type: Optional[TaskType] = None,
    force_backend: Optional[ModelBackend] = None,
    audio_path: Optional[str] = None,
    media_parts: Optional[list] = None,
) -> LLMResponse:
    """Convenience function wrapping LLMRouter.route()."""
    return router.route(
        system=system,
        user=user,
        task_type=task_type,
        force_backend=force_backend,
        audio_path=audio_path,
        media_parts=media_parts,
    )