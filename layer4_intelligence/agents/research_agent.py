"""
Layer 4 Intelligence - Research Agent
Performs multi-step research by decomposing a question into sub-queries,
answering each via the RAG pipeline, then synthesising a final cited answer.

Steps:
  1. PLAN   — decompose question into 3-5 sub-queries (phi3:mini)
  2. FETCH  — run each sub-query through RAGPipeline
  3. GUARD  — each sub-answer already guarded by RAGPipeline
  4. SYNTH  — synthesise sub-answers into final answer (Groq LLaMA)
  5. VERIFY — final answer re-checked by HallucinationGuard
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
class SubQueryResult:
    sub_query: str
    answer: str
    passed_guard: bool
    confidence_score: float
    sources_used: list[str]


@dataclass
class ResearchResult:
    original_question: str
    sub_queries: list[str]
    sub_results: list[SubQueryResult]
    final_answer: str
    overall_confidence: float
    passed_guard: bool
    latency_ms: float
    total_chunks_retrieved: int


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Multi-step research agent powered by RAGPipeline and LLMRouter.

    Args:
        rag_pipeline:    Configured RAGPipeline instance.
        max_sub_queries: Maximum number of sub-queries to generate (default 5).
        filters:         Optional vector store metadata filters applied to all sub-queries.
    """

    def __init__(
        self,
        rag_pipeline,               # RAGPipeline — avoid circular import with string hint
        max_sub_queries: int = 5,
        filters: Optional[dict] = None,
    ):
        self.rag = rag_pipeline
        self.max_sub_queries = max_sub_queries
        self.filters = filters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, question: str) -> ResearchResult:
        """
        Run the full multi-step research pipeline.

        Args:
            question: The user's research question.

        Returns:
            ResearchResult with final answer and full audit trail.
        """
        t0 = time.perf_counter()
        logger.info("ResearchAgent starting: %s", question[:120])

        # Step 1: Plan
        sub_queries = self._plan(question)
        logger.info("Research plan: %d sub-queries generated.", len(sub_queries))

        # Step 2 & 3: Fetch + guard (via RAG pipeline)
        sub_results: list[SubQueryResult] = []
        total_chunks = 0

        for i, sq in enumerate(sub_queries):
            logger.info("  Sub-query %d/%d: %s", i + 1, len(sub_queries), sq[:80])
            try:
                rag_result = self.rag.query(question=sq, filters=self.filters)
                sources = list({c.source_id for c in rag_result.chunks_used})
                total_chunks += rag_result.chunks_retrieved

                sub_results.append(SubQueryResult(
                    sub_query=sq,
                    answer=rag_result.answer,
                    passed_guard=rag_result.passed_guard,
                    confidence_score=rag_result.confidence_score,
                    sources_used=sources,
                ))
            except Exception as exc:
                logger.error("Sub-query %d failed: %s", i + 1, exc)
                sub_results.append(SubQueryResult(
                    sub_query=sq,
                    answer=f"NO_DATA_FOUND: Sub-query failed — {exc}",
                    passed_guard=False,
                    confidence_score=0.0,
                    sources_used=[],
                ))

        # Step 4: Synthesise
        final_answer = self._synthesise(question, sub_results)

        # Step 5: Final guard check
        all_context = self._merge_sub_answers_as_context(sub_results)
        from layer4_intelligence.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        guard_result = guard.verify(
            question=question,
            context=all_context,
            answer=final_answer,
            avg_similarity=self._avg_confidence(sub_results),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ResearchAgent done in %.0f ms | confidence=%.2f | guard_passed=%s",
            latency_ms, guard_result.confidence_score, guard_result.passed,
        )

        return ResearchResult(
            original_question=question,
            sub_queries=sub_queries,
            sub_results=sub_results,
            final_answer=guard_result.final_answer,
            overall_confidence=guard_result.confidence_score,
            passed_guard=guard_result.passed,
            latency_ms=latency_ms,
            total_chunks_retrieved=total_chunks,
        )

    # ------------------------------------------------------------------
    # Step 1: Planning
    # ------------------------------------------------------------------

    def _plan(self, question: str) -> list[str]:
        """Decompose the question into sub-queries using phi3:mini."""
        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt("research_plan", question=question)

        try:
            resp = route_query(system=system, user=user, task_type=TaskType.QUICK_QA)
            raw = resp.content.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            sub_queries = json.loads(raw)

            if not isinstance(sub_queries, list):
                raise ValueError("Response is not a JSON array.")

            # Enforce max_sub_queries and non-empty strings
            sub_queries = [
                sq.strip() for sq in sub_queries
                if isinstance(sq, str) and sq.strip()
            ][: self.max_sub_queries]

            if not sub_queries:
                raise ValueError("Empty sub-query list.")

            return sub_queries

        except Exception as exc:
            logger.warning("Planning LLM failed (%s) — falling back to original question.", exc)
            return [question]

    # ------------------------------------------------------------------
    # Step 4: Synthesis
    # ------------------------------------------------------------------

    def _synthesise(self, question: str, sub_results: list[SubQueryResult]) -> str:
        """Synthesise sub-answers into one final answer using Groq LLaMA."""
        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        sub_answers_text = "\n\n".join(
            f"[Sub-query {i + 1}]: {sr.sub_query}\n"
            f"[Answer]: {sr.answer}\n"
            f"[Confidence]: {sr.confidence_score:.2f} | [Guard passed]: {sr.passed_guard}"
            for i, sr in enumerate(sub_results)
        )

        system, user = render_prompt(
            "research_synthesise",
            question=question,
            sub_answers=sub_answers_text,
        )

        try:
            resp = route_query(system=system, user=user, task_type=TaskType.HEAVY_ANALYSIS)
            return resp.content
        except Exception as exc:
            logger.error("Synthesis LLM failed: %s", exc)
            # Fallback: concatenate sub-answers
            return "SYNTHESIS FAILED. Raw sub-answers:\n\n" + sub_answers_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_sub_answers_as_context(sub_results: list[SubQueryResult]) -> str:
        """Build a pseudo-context string from all sub-answers for final guard check."""
        parts = []
        for i, sr in enumerate(sub_results):
            parts.append(
                f"[Source: sub_query_{i + 1} | query: {sr.sub_query[:60]}]\n{sr.answer}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _avg_confidence(sub_results: list[SubQueryResult]) -> float:
        if not sub_results:
            return 0.0
        return sum(sr.confidence_score for sr in sub_results) / len(sub_results)