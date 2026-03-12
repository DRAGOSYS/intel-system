"""
Layer 4 Intelligence - Media Agent
Handles analysis of images, video, audio, and PDF files.

Routing:
  Images / PDFs / Video frames  → Gemini 1.5 Flash (multimodal)
  Audio                         → faster-whisper tiny (local ASR) + optional cleanup
  PDF text extraction           → PyMuPDF locally, then Gemini or RAG for Q&A
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported media types
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
SUPPORTED_PDF_TYPE = ".pdf"

# Video frame sampling
DEFAULT_FRAME_SAMPLE_INTERVAL = 5   # seconds between sampled frames


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MediaAnalysisResult:
    """Unified result from any media analysis."""
    media_type: str                 # image | video | audio | pdf
    file_path: str
    question: Optional[str]
    answer: str
    passed_guard: bool
    confidence_score: float
    raw_transcript: Optional[str] = None   # Audio only
    extracted_text: Optional[str] = None   # PDF only
    frame_count: int = 0                   # Video only
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Media Agent
# ---------------------------------------------------------------------------

class MediaAgent:
    """
    Unified media analysis agent.

    Usage:
        agent = MediaAgent()
        result = agent.analyse("path/to/image.jpg", question="What text is visible?")
        result = agent.analyse("path/to/report.pdf", question="What are the conclusions?")
        result = agent.analyse("path/to/audio.mp3")   # transcription
    """

    def __init__(
        self,
        use_hallucination_guard: bool = True,
        frame_interval: int = DEFAULT_FRAME_SAMPLE_INTERVAL,
        audio_language: Optional[str] = None,
    ):
        self.use_guard = use_hallucination_guard
        self.frame_interval = frame_interval
        self.audio_language = audio_language

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        file_path: str,
        question: Optional[str] = None,
    ) -> MediaAnalysisResult:
        """
        Auto-detect media type and dispatch to the correct handler.

        Args:
            file_path: Path to the media file.
            question:  Optional question about the media content.

        Returns:
            MediaAnalysisResult with answer and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        ext = path.suffix.lower()

        if ext in SUPPORTED_IMAGE_TYPES:
            return self._analyse_image(file_path, question)
        if ext in SUPPORTED_VIDEO_TYPES:
            return self._analyse_video(file_path, question)
        if ext in SUPPORTED_AUDIO_TYPES:
            return self._analyse_audio(file_path, question)
        if ext == SUPPORTED_PDF_TYPE:
            return self._analyse_pdf(file_path, question)

        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: images={SUPPORTED_IMAGE_TYPES}, "
            f"video={SUPPORTED_VIDEO_TYPES}, "
            f"audio={SUPPORTED_AUDIO_TYPES}, "
            f"pdf={{'.pdf'}}"
        )

    # ------------------------------------------------------------------
    # Image analysis
    # ------------------------------------------------------------------

    def _analyse_image(self, file_path: str, question: Optional[str]) -> MediaAnalysisResult:
        t0 = time.perf_counter()
        logger.info("MediaAgent: image analysis — %s", file_path)

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            "image_analysis",
            question=question or "Describe all visible content in detail.",
        )

        # Build Gemini multimodal parts
        image_part = self._load_image_as_gemini_part(file_path)
        parts = [image_part, user]

        resp = route_query(
            system=system,
            user=user,
            task_type=TaskType.IMAGE,
            media_parts=parts,
        )
        answer = resp.content

        passed, confidence = self._guard(
            question=question or "Image analysis",
            context=f"[Image file: {Path(file_path).name}]\n{answer}",
            answer=answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return MediaAnalysisResult(
            media_type="image",
            file_path=file_path,
            question=question,
            answer=answer,
            passed_guard=passed,
            confidence_score=confidence,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Video analysis
    # ------------------------------------------------------------------

    def _analyse_video(self, file_path: str, question: Optional[str]) -> MediaAnalysisResult:
        t0 = time.perf_counter()
        logger.info("MediaAgent: video analysis — %s", file_path)

        frames, timestamps, duration = self._extract_video_frames(file_path)
        logger.info("Extracted %d frames from %.1f s video.", len(frames), duration)

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            "video_frame_analysis",
            metadata=f"file={Path(file_path).name}, duration={duration:.1f}s, frames={len(frames)}",
            timestamps=", ".join(f"{t:.1f}s" for t in timestamps),
            question=question or "Describe the video content and key events.",
        )

        # Build multimodal parts: interleave timestamp text + frame image
        parts = [user]
        for ts, frame_bytes in zip(timestamps, frames):
            parts.append(f"\n--- Frame at {ts:.1f}s ---")
            parts.append(self._bytes_to_gemini_part(frame_bytes, "image/jpeg"))

        resp = route_query(
            system=system,
            user=user,
            task_type=TaskType.VIDEO,
            media_parts=parts,
        )
        answer = resp.content

        passed, confidence = self._guard(
            question=question or "Video analysis",
            context=f"[Video: {Path(file_path).name}, {len(frames)} frames]\n{answer}",
            answer=answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return MediaAnalysisResult(
            media_type="video",
            file_path=file_path,
            question=question,
            answer=answer,
            passed_guard=passed,
            confidence_score=confidence,
            frame_count=len(frames),
            latency_ms=latency_ms,
            metadata={"duration_seconds": duration},
        )

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    def _analyse_audio(self, file_path: str, question: Optional[str]) -> MediaAnalysisResult:
        t0 = time.perf_counter()
        logger.info("MediaAgent: audio transcription — %s", file_path)

        from layer4_intelligence.llm_router import route_query, TaskType, ModelBackend

        # Step 1: Transcribe with faster-whisper
        resp = route_query(
            system="",
            user="",
            task_type=TaskType.AUDIO,
            force_backend=ModelBackend.FASTER_WHISPER,
            audio_path=file_path,
        )
        raw_transcript = resp.content

        # Step 2: Clean up the transcript
        from layer4_intelligence.prompt_templates import render_prompt

        system, user = render_prompt(
            "audio_transcription_cleanup",
            raw_transcript=raw_transcript,
        )
        cleanup_resp = route_query(system=system, user=user, task_type=TaskType.QUICK_QA)
        clean_transcript = cleanup_resp.content

        # Step 3: If a question is asked, answer it from the transcript
        if question:
            qa_system, qa_user = render_prompt(
                "rag_answer",
                context=f"[Source: audio_transcript | file: {Path(file_path).name}]\n{clean_transcript}",
                question=question,
            )
            qa_resp = route_query(system=qa_system, user=qa_user, task_type=TaskType.QUICK_QA)
            final_answer = qa_resp.content
        else:
            final_answer = clean_transcript

        passed, confidence = self._guard(
            question=question or "Audio transcription",
            context=clean_transcript,
            answer=final_answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return MediaAnalysisResult(
            media_type="audio",
            file_path=file_path,
            question=question,
            answer=final_answer,
            passed_guard=passed,
            confidence_score=confidence,
            raw_transcript=raw_transcript,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # PDF analysis
    # ------------------------------------------------------------------

    def _analyse_pdf(self, file_path: str, question: Optional[str]) -> MediaAnalysisResult:
        t0 = time.perf_counter()
        logger.info("MediaAgent: PDF analysis — %s", file_path)

        extracted_text, page_count = self._extract_pdf_text(file_path)

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        if question:
            system, user = render_prompt(
                "pdf_extraction_qa",
                page_range=f"1-{page_count}",
                pdf_text=extracted_text[:20_000],
                question=question,
            )
            # Route as multimodal task (Gemini) for long PDFs, else Groq
            task = TaskType.PDF if len(extracted_text) > 8000 else TaskType.HEAVY_ANALYSIS
            resp = route_query(system=system, user=user, task_type=task)
            answer = resp.content
        else:
            # Default: summarise
            system, user = render_prompt(
                "rag_summarise",
                source_id=Path(file_path).name,
                document_text=extracted_text[:20_000],
            )
            resp = route_query(system=system, user=user, task_type=TaskType.HEAVY_ANALYSIS)
            answer = resp.content

        passed, confidence = self._guard(
            question=question or "PDF summarisation",
            context=f"[Source: {Path(file_path).name}]\n{extracted_text[:8000]}",
            answer=answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return MediaAnalysisResult(
            media_type="pdf",
            file_path=file_path,
            question=question,
            answer=answer,
            passed_guard=passed,
            confidence_score=confidence,
            extracted_text=extracted_text,
            latency_ms=latency_ms,
            metadata={"page_count": page_count},
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image_as_gemini_part(file_path: str):
        """Load image file and return a Gemini-compatible Part object."""
        import google.generativeai as genai  # pip install google-generativeai

        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "image/jpeg"

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        return genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    @staticmethod
    def _bytes_to_gemini_part(image_bytes: bytes, mime_type: str = "image/jpeg"):
        import google.generativeai as genai

        return genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def _extract_video_frames(
        self,
        file_path: str,
    ) -> tuple[list[bytes], list[float], float]:
        """
        Extract sampled JPEG frames from a video file using OpenCV.

        Returns:
            (frame_bytes_list, timestamps_list, duration_seconds)
        """
        try:
            import cv2  # pip install opencv-python-headless
        except ImportError:
            raise ImportError(
                "OpenCV is required for video analysis: pip install opencv-python-headless"
            )

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        frame_bytes_list: list[bytes] = []
        timestamps: list[float] = []

        sample_every_n = max(1, int(fps * self.frame_interval))
        frame_idx = 0
        max_frames = 20  # Hard cap to avoid Gemini token overflow

        while cap.isOpened() and len(frame_bytes_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n == 0:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    frame_bytes_list.append(bytes(buf))
                    timestamps.append(frame_idx / fps)
            frame_idx += 1

        cap.release()
        return frame_bytes_list, timestamps, duration

    @staticmethod
    def _extract_pdf_text(file_path: str) -> tuple[str, int]:
        """
        Extract full text from a PDF using PyMuPDF.

        Returns:
            (full_text, page_count)
        """
        try:
            import fitz  # pip install pymupdf
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF analysis: pip install pymupdf")

        doc = fitz.open(file_path)
        pages_text = []
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                pages_text.append(f"[Page {i + 1}]\n{page_text.strip()}")

        full_text = "\n\n".join(pages_text)
        page_count = len(doc)
        doc.close()
        return full_text, page_count

    def _guard(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> tuple[bool, float]:
        """Run hallucination guard and return (passed, confidence)."""
        if not self.use_guard:
            return True, 1.0
        from layer4_intelligence.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        result = guard.verify(
            question=question,
            context=context,
            answer=answer,
            avg_similarity=0.9,  # Media: answer is generated from the file itself
        )
        return result.passed, result.confidence_score