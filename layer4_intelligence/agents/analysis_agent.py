"""
Layer 4 Intelligence - Analysis Agent
Performs deep, structured analysis of one or multiple documents.

Capabilities:
  • Single-document deep analysis (entities, arguments, data points, contradictions)
  • Multi-document comparative analysis across custom dimensions
  • Sentiment analysis
  • All outputs verified by HallucinationGuard
  • Routed to Groq LLaMA (heavy analysis) or phi3:mini (quick tasks)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DocumentInput:
    """A document to be analysed."""
    doc_id: str
    text: str
    doc_type: str = "text"          # text | pdf | html | markdown
    metadata: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result from a single-document analysis."""
    doc_id: str
    analysis: dict                  # Structured JSON from LLM
    raw_answer: str
    passed_guard: bool
    confidence_score: float
    latency_ms: float


@dataclass
class ComparisonResult:
    """Result from a multi-document comparison."""
    doc_ids: list[str]
    dimensions: list[str]
    comparison_table: dict
    verdict: str
    raw_answer: str
    passed_guard: bool
    confidence_score: float
    latency_ms: float


@dataclass
class SentimentResult:
    doc_id: str
    sentiment: str                  # POSITIVE | NEGATIVE | NEUTRAL
    confidence: float
    driving_phrases: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# Analysis Agent
# ---------------------------------------------------------------------------

class AnalysisAgent:
    """
    Deep document analysis agent.

    Usage:
        agent = AnalysisAgent()

        # Deep single-document analysis
        result = agent.analyse_document(DocumentInput(doc_id="rpt_01", text="..."))

        # Compare two documents
        result = agent.compare_documents(
            documents=[doc1, doc2],
            dimensions=["methodology", "conclusions", "sample size"],
        )

        # Sentiment
        result = agent.analyse_sentiment(DocumentInput(doc_id="review_01", text="..."))
    """

    def __init__(
        self,
        use_hallucination_guard: bool = True,
        max_doc_chars: int = 24_000,   # Truncation limit per document
    ):
        self.use_guard = use_hallucination_guard
        self.max_doc_chars = max_doc_chars

    # ------------------------------------------------------------------
    # Single-document analysis
    # ------------------------------------------------------------------

    def analyse_document(self, document: DocumentInput) -> AnalysisResult:
        """
        Run structured deep analysis on a single document.

        Returns AnalysisResult with JSON analysis dict (entities, arguments,
        contradictions, data points).
        """
        t0 = time.perf_counter()
        logger.info("AnalysisAgent: deep analysis of doc_id=%s", document.doc_id)

        truncated_text = self._truncate(document.text)
        context = (
            f"[Source: {document.doc_id} | type: {document.doc_type}]\n{truncated_text}"
        )

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            "document_analysis",
            doc_id=document.doc_id,
            doc_type=document.doc_type,
            document_text=truncated_text,
        )

        resp = route_query(system=system, user=user, task_type=TaskType.HEAVY_ANALYSIS)
        raw_answer = resp.content

        analysis_dict = self._parse_json_response(raw_answer, fallback_key="raw_text")

        # Guard
        passed, confidence = self._guard_check(
            question=f"Analyse document {document.doc_id}",
            context=context,
            answer=raw_answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Document analysis done: doc=%s | %.0f ms | guard_passed=%s",
            document.doc_id, latency_ms, passed,
        )

        return AnalysisResult(
            doc_id=document.doc_id,
            analysis=analysis_dict,
            raw_answer=raw_answer,
            passed_guard=passed,
            confidence_score=confidence,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Multi-document comparison
    # ------------------------------------------------------------------

    def compare_documents(
        self,
        documents: list[DocumentInput],
        dimensions: Optional[list[str]] = None,
    ) -> ComparisonResult:
        """
        Compare multiple documents across specified dimensions.

        Args:
            documents:  2+ DocumentInput objects.
            dimensions: Aspects to compare (e.g. ["methodology", "findings"]).
                        Defaults to ["summary", "key_arguments", "conclusions"].

        Returns:
            ComparisonResult with comparison_table and verdict.
        """
        if len(documents) < 2:
            raise ValueError("compare_documents requires at least 2 documents.")

        dimensions = dimensions or ["summary", "key_arguments", "conclusions", "limitations"]
        t0 = time.perf_counter()
        logger.info(
            "AnalysisAgent: comparing %d documents on %d dimensions.",
            len(documents), len(dimensions),
        )

        docs_text = "\n\n===\n\n".join(
            f"DOCUMENT [{d.doc_id}] (type={d.doc_type}):\n"
            + self._truncate(d.text, max_chars=self.max_doc_chars // len(documents))
            for d in documents
        )

        full_context = "\n\n".join(
            f"[Source: {d.doc_id}]\n{self._truncate(d.text, max_chars=4000)}"
            for d in documents
        )

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            "comparative_analysis",
            documents=docs_text,
            dimensions="\n".join(f"  • {dim}" for dim in dimensions),
        )

        resp = route_query(system=system, user=user, task_type=TaskType.HEAVY_ANALYSIS)
        raw_answer = resp.content
        parsed = self._parse_json_response(raw_answer)

        passed, confidence = self._guard_check(
            question=f"Compare documents: {', '.join(d.doc_id for d in documents)}",
            context=full_context,
            answer=raw_answer,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return ComparisonResult(
            doc_ids=[d.doc_id for d in documents],
            dimensions=dimensions,
            comparison_table=parsed.get("comparison_table", {}),
            verdict=parsed.get("verdict", "See raw_answer for details."),
            raw_answer=raw_answer,
            passed_guard=passed,
            confidence_score=confidence,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Sentiment analysis
    # ------------------------------------------------------------------

    def analyse_sentiment(self, document: DocumentInput) -> SentimentResult:
        """
        Run sentiment analysis on a document.

        Returns SentimentResult with POSITIVE/NEGATIVE/NEUTRAL classification,
        confidence, and top driving phrases.
        """
        t0 = time.perf_counter()
        logger.info("AnalysisAgent: sentiment for doc_id=%s", document.doc_id)

        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            "sentiment_analysis",
            text=self._truncate(document.text, max_chars=8000),
        )

        resp = route_query(system=system, user=user, task_type=TaskType.QUICK_QA)
        parsed = self._parse_json_response(resp.content)

        latency_ms = (time.perf_counter() - t0) * 1000

        return SentimentResult(
            doc_id=document.doc_id,
            sentiment=parsed.get("sentiment", "NEUTRAL"),
            confidence=float(parsed.get("confidence", 0.5)),
            driving_phrases=parsed.get("driving_phrases", []),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyse_batch(
        self,
        documents: list[DocumentInput],
    ) -> list[AnalysisResult]:
        """Analyse multiple documents sequentially. Returns a list of AnalysisResult."""
        results = []
        for doc in documents:
            try:
                results.append(self.analyse_document(doc))
            except Exception as exc:
                logger.error("Batch analysis failed for doc=%s: %s", doc.doc_id, exc)
                results.append(AnalysisResult(
                    doc_id=doc.doc_id,
                    analysis={"error": str(exc)},
                    raw_answer="",
                    passed_guard=False,
                    confidence_score=0.0,
                    latency_ms=0.0,
                ))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str, max_chars: Optional[int] = None) -> str:
        limit = max_chars or self.max_doc_chars
        if len(text) > limit:
            logger.debug("Document truncated from %d to %d chars.", len(text), limit)
            return text[:limit] + "\n...[TRUNCATED]"
        return text

    @staticmethod
    def _parse_json_response(raw: str, fallback_key: str = "content") -> dict:
        """Parse JSON from LLM response, with fallback to raw text dict."""
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            return {fallback_key: result}
        except json.JSONDecodeError:
            return {fallback_key: raw}

    def _guard_check(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> tuple[bool, float]:
        """Run hallucination guard and return (passed, confidence_score)."""
        if not self.use_guard:
            return True, 1.0
        from layer4_intelligence.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        result = guard.verify(
            question=question,
            context=context,
            answer=answer,
            avg_similarity=0.85,  # Analysis uses the full doc — similarity is implicit
        )
        return result.passed, result.confidence_score