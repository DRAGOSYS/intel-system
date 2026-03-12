"""
Layer 4 Intelligence - Hallucination Guard
Ensures every LLM response is grounded in retrieved context.

Strategy:
  1. Pre-generation: similarity threshold gate (0.75) — only pass context
     that is genuinely relevant.
  2. Post-generation: LLM-based fact verification against source context.
  3. Confidence scoring: composite score from similarity + citation coverage.
  4. Hard rejection: answers below threshold are replaced with NO_DATA_FOUND.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.75       # Minimum cosine similarity for context chunks
CONFIDENCE_THRESHOLD = 0.6        # Minimum composite confidence to pass
NO_DATA_FOUND_MARKER = "NO_DATA_FOUND"
CITATION_PATTERN = re.compile(r"\[Source:\s*[^\]]+\]|\[p\.\d+\]|\[doc_id:[^\]]+\]")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContextChunk:
    """A single retrieved chunk of context."""
    chunk_id: str
    source_id: str
    text: str
    similarity_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class GuardResult:
    """Result from the hallucination guard pipeline."""
    passed: bool                            # True = safe to return to user
    final_answer: str                       # Possibly cleaned or replaced
    confidence_score: float                 # 0.0 – 1.0
    verdict: str                            # SUPPORTED | PARTIAL | UNSUPPORTED | NO_CONTEXT
    unsupported_claims: list[str] = field(default_factory=list)
    filtered_chunks: int = 0               # Chunks removed by similarity gate
    citation_coverage: float = 0.0         # Fraction of claims with citations


# ---------------------------------------------------------------------------
# Similarity gate
# ---------------------------------------------------------------------------

def filter_chunks_by_similarity(
    chunks: list[ContextChunk],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[ContextChunk], int]:
    """
    Remove chunks whose similarity score falls below the threshold.

    Returns:
        (passing_chunks, rejected_count)
    """
    passing = [c for c in chunks if c.similarity_score >= threshold]
    rejected = len(chunks) - len(passing)
    if rejected:
        logger.info(
            "Similarity gate: rejected %d/%d chunks (threshold=%.2f)",
            rejected, len(chunks), threshold,
        )
    return passing, rejected


def build_context_string(chunks: list[ContextChunk]) -> str:
    """Format passing chunks into a single context block with source tags."""
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c.source_id} | chunk: {c.chunk_id} | sim: {c.similarity_score:.2f}]\n"
            f"{c.text.strip()}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Citation analysis
# ---------------------------------------------------------------------------

def compute_citation_coverage(answer: str) -> float:
    """
    Estimate what fraction of answer sentences contain a citation.
    Simple heuristic: count sentences, count cited sentences.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if CITATION_PATTERN.search(s))
    return cited / len(sentences)


def extract_citations(answer: str) -> list[str]:
    """Extract all citation strings from an answer."""
    return CITATION_PATTERN.findall(answer)


# ---------------------------------------------------------------------------
# LLM-based fact verification
# ---------------------------------------------------------------------------

def _verify_with_llm(context: str, answer: str) -> dict:
    """
    Call the hallucination-check LLM prompt and parse the JSON result.
    Returns dict with keys: verdict, unsupported_claims, confidence.
    Falls back to a safe default on parse failure.
    """
    from layer4_intelligence.prompt_templates import render_prompt
    from layer4_intelligence.llm_router import route_query, TaskType

    system, user = render_prompt(
        "hallucination_check",
        context=context[:6000],   # Trim to avoid token overflow
        answer=answer,
    )

    try:
        resp = route_query(system=system, user=user, task_type=TaskType.QUICK_QA)
        raw = resp.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        return {
            "verdict": result.get("verdict", "UNSUPPORTED"),
            "unsupported_claims": result.get("unsupported_claims", []),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except Exception as exc:
        logger.warning("LLM hallucination check failed: %s", exc)
        # Conservative fallback — treat as partial
        return {"verdict": "PARTIAL", "unsupported_claims": [], "confidence": 0.5}


def _score_confidence_with_llm(
    question: str,
    similarity_score: float,
    answer: str,
) -> float:
    """Use the confidence_scoring prompt to get a calibrated score."""
    from layer4_intelligence.prompt_templates import render_prompt
    from layer4_intelligence.llm_router import route_query, TaskType

    system, user = render_prompt(
        "confidence_scoring",
        question=question,
        similarity_score=f"{similarity_score:.2f}",
        answer=answer,
    )

    try:
        resp = route_query(system=system, user=user, task_type=TaskType.QUICK_QA)
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        return float(result.get("confidence_score", 0.0))
    except Exception as exc:
        logger.warning("Confidence scoring LLM call failed: %s", exc)
        return similarity_score  # Fallback to raw similarity


# ---------------------------------------------------------------------------
# Main guard class
# ---------------------------------------------------------------------------

class HallucinationGuard:
    """
    End-to-end hallucination prevention pipeline.

    Usage:
        guard = HallucinationGuard()

        # Step 1: filter chunks before generation
        clean_chunks, rejected = guard.filter_context(raw_chunks)
        context_str = build_context_string(clean_chunks)

        # Step 2: after generation, verify the answer
        result = guard.verify(
            question="...",
            context=context_str,
            answer=generated_answer,
            avg_similarity=avg_score,
        )
        if result.passed:
            return result.final_answer
        else:
            return "I could not find a reliable answer in the source documents."
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_llm_verification: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.use_llm_verification = use_llm_verification

    # ------------------------------------------------------------------
    # Pre-generation
    # ------------------------------------------------------------------

    def filter_context(
        self,
        chunks: list[ContextChunk],
    ) -> tuple[list[ContextChunk], int]:
        """Filter chunks by similarity threshold. Returns (clean_chunks, rejected_count)."""
        return filter_chunks_by_similarity(chunks, self.similarity_threshold)

    # ------------------------------------------------------------------
    # Post-generation
    # ------------------------------------------------------------------

    def verify(
        self,
        question: str,
        context: str,
        answer: str,
        avg_similarity: float = 0.0,
    ) -> GuardResult:
        """
        Full verification pipeline.

        Args:
            question:       The original user question.
            context:        The context string that was fed to the LLM.
            answer:         The LLM-generated answer.
            avg_similarity: Average similarity of the retrieved chunks.

        Returns:
            GuardResult with pass/fail decision and metadata.
        """
        # --- 1. Check for empty context ---
        if not context.strip():
            logger.warning("No context available — returning NO_DATA_FOUND.")
            return GuardResult(
                passed=False,
                final_answer=self._no_data_response(question),
                confidence_score=0.0,
                verdict="NO_CONTEXT",
            )

        # --- 2. Check if model already admitted no data ---
        if NO_DATA_FOUND_MARKER in answer:
            logger.info("Model returned NO_DATA_FOUND — propagating.")
            return GuardResult(
                passed=False,
                final_answer=self._no_data_response(question),
                confidence_score=0.0,
                verdict="NO_CONTEXT",
            )

        # --- 3. Citation coverage heuristic ---
        citation_coverage = compute_citation_coverage(answer)

        # --- 4. LLM-based fact verification ---
        if self.use_llm_verification:
            llm_check = _verify_with_llm(context, answer)
            verdict = llm_check["verdict"]
            unsupported = llm_check["unsupported_claims"]
            llm_confidence = llm_check["confidence"]
        else:
            verdict = "PARTIAL"
            unsupported = []
            llm_confidence = avg_similarity

        # --- 5. Composite confidence score ---
        confidence = self._compute_composite_confidence(
            avg_similarity=avg_similarity,
            citation_coverage=citation_coverage,
            llm_confidence=llm_confidence,
            verdict=verdict,
        )

        # --- 6. Decision ---
        if verdict == "UNSUPPORTED" or confidence < self.confidence_threshold:
            logger.warning(
                "Answer rejected: verdict=%s, confidence=%.2f",
                verdict, confidence,
            )
            return GuardResult(
                passed=False,
                final_answer=self._low_confidence_response(question, unsupported),
                confidence_score=confidence,
                verdict=verdict,
                unsupported_claims=unsupported,
                citation_coverage=citation_coverage,
            )

        # --- 7. Pass with optional warning for PARTIAL ---
        final_answer = answer
        if verdict == "PARTIAL" and unsupported:
            disclaimer = (
                "\n\n⚠️ *Note: Some claims could not be fully verified against "
                "source documents and have been flagged for review.*"
            )
            final_answer = answer + disclaimer

        logger.info(
            "Answer passed guard: verdict=%s, confidence=%.2f, citations=%.0f%%",
            verdict, confidence, citation_coverage * 100,
        )

        return GuardResult(
            passed=True,
            final_answer=final_answer,
            confidence_score=confidence,
            verdict=verdict,
            unsupported_claims=unsupported,
            citation_coverage=citation_coverage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_composite_confidence(
        avg_similarity: float,
        citation_coverage: float,
        llm_confidence: float,
        verdict: str,
    ) -> float:
        """Weighted composite confidence score."""
        verdict_weight = {"SUPPORTED": 1.0, "PARTIAL": 0.65, "UNSUPPORTED": 0.1}.get(verdict, 0.5)
        score = (
            0.35 * avg_similarity
            + 0.25 * citation_coverage
            + 0.25 * llm_confidence
            + 0.15 * verdict_weight
        )
        return round(min(max(score, 0.0), 1.0), 4)

    @staticmethod
    def _no_data_response(question: str) -> str:
        return (
            f"NO_DATA_FOUND: The retrieved documents do not contain sufficient "
            f"information to answer the question: \"{question}\". "
            "Please refine your query or provide additional source documents."
        )

    @staticmethod
    def _low_confidence_response(question: str, unsupported: list[str]) -> str:
        lines = [
            "LOW_CONFIDENCE: The generated answer could not be fully verified "
            "against the source context.",
            f"Question: \"{question}\"",
        ]
        if unsupported:
            lines.append("Unverified claims detected:")
            for claim in unsupported[:5]:
                lines.append(f"  • {claim}")
        lines.append(
            "Please provide more relevant source documents or rephrase the question."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

guard = HallucinationGuard()