"""
Layer 4 Intelligence - RAG Pipeline
Retrieves relevant context chunks from the vector store, reranks them,
generates a grounded answer via the LLM router, and passes the result
through the hallucination guard.

Key parameters:
  TOP_K_RESULTS       = 5    (initial retrieval)
  RERANK_TOP_N        = 3    (after cross-encoder reranking)
  SIMILARITY_THRESHOLD= 0.75 (enforced by hallucination guard)
  LLM_TEMPERATURE     = 0.1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_RESULTS = 5
RERANK_TOP_N = 3
SIMILARITY_THRESHOLD = 0.75
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""
    chunk_id: str
    source_id: str
    text: str
    similarity_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    """Full result from a RAG pipeline invocation."""
    question: str
    answer: str
    passed_guard: bool
    confidence_score: float
    verdict: str
    chunks_used: list[RetrievedChunk]
    chunks_retrieved: int
    chunks_after_filter: int
    latency_ms: float
    backend_used: str
    citation_coverage: float = 0.0


# ---------------------------------------------------------------------------
# Abstract vector store interface
# ---------------------------------------------------------------------------

class VectorStoreAdapter:
    """
    Base class for vector store adapters.
    Subclass this for ChromaDB, Qdrant, Weaviate, FAISS, etc.
    """

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Search the vector store and return top_k chunks.

        Args:
            query:   Natural language query (will be embedded internally).
            top_k:   Number of results to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of RetrievedChunk sorted by descending similarity.
        """
        raise NotImplementedError


class ChromaAdapter(VectorStoreAdapter):
    """ChromaDB adapter (default vector store)."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_fn: Optional[Callable] = None,
    ):
        import chromadb  # pip install chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=persist_directory)
        ef = embedding_fn or embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance → similarity: 1 - distance
            similarity = round(1.0 - distance, 4)
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            chunk_id = results["ids"][0][i]
            source_id = meta.get("source_id", meta.get("source", chunk_id))

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    text=doc,
                    similarity_score=similarity,
                    metadata=meta,
                )
            )

        logger.debug("ChromaDB returned %d chunks for query.", len(chunks))
        return chunks


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Falls back to similarity-based ordering if model unavailable.
    """

    _model = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = RERANK_TOP_N,
    ) -> list[RetrievedChunk]:
        try:
            from sentence_transformers import CrossEncoder  # pip install sentence-transformers

            if self._model is None:
                CrossEncoderReranker._model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )

            pairs = [(query, c.text) for c in chunks]
            scores = self._model.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk.metadata["rerank_score"] = float(score)

            reranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"], reverse=True)
            logger.debug("Reranker selected top %d from %d chunks.", top_n, len(chunks))
            return reranked[:top_n]

        except Exception as exc:
            logger.warning("Cross-encoder reranking failed (%s) — using similarity order.", exc)
            return sorted(chunks, key=lambda c: c.similarity_score, reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Full RAG pipeline:
      retrieve → similarity_gate → rerank → build_context → generate → guard
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreAdapter] = None,
        top_k: int = TOP_K_RESULTS,
        rerank_top_n: int = RERANK_TOP_N,
        use_reranker: bool = True,
        use_hallucination_guard: bool = True,
    ):
        self.vector_store = vector_store  # Injected externally
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.use_guard = use_hallucination_guard

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: Optional[dict] = None,
        task_hint: Optional[str] = None,  # "summarise" | "analyse" | "qa"
    ) -> RAGResult:
        """
        End-to-end RAG query.

        Args:
            question:   The user's natural language question.
            filters:    Optional metadata filters for retrieval.
            task_hint:  Hint to select the right prompt template.

        Returns:
            RAGResult with answer and full metadata.
        """
        t0 = time.perf_counter()

        if not self.vector_store:
            raise RuntimeError("RAGPipeline.vector_store is not configured.")

        # 1. Retrieve
        raw_chunks = self.vector_store.search(question, top_k=self.top_k, filters=filters)
        logger.info("Retrieved %d chunks from vector store.", len(raw_chunks))

        # 2. Similarity gate
        from layer4_intelligence.hallucination_guard import (
            ContextChunk,
            HallucinationGuard,
            build_context_string,
        )

        context_chunks = [
            ContextChunk(
                chunk_id=c.chunk_id,
                source_id=c.source_id,
                text=c.text,
                similarity_score=c.similarity_score,
                metadata=c.metadata,
            )
            for c in raw_chunks
        ]

        guard_instance = HallucinationGuard()
        passing_ctx, filtered_count = guard_instance.filter_context(context_chunks)
        logger.info(
            "Similarity gate: %d passed, %d filtered.",
            len(passing_ctx), filtered_count,
        )

        # 3. Rerank
        if self.reranker and passing_ctx:
            # Convert back to RetrievedChunk for reranker
            passing_retrieved = [
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    source_id=c.source_id,
                    text=c.text,
                    similarity_score=c.similarity_score,
                    metadata=c.metadata,
                )
                for c in passing_ctx
            ]
            top_retrieved = self.reranker.rerank(question, passing_retrieved, top_n=self.rerank_top_n)
            # Sync back
            passing_ctx = [
                ContextChunk(
                    chunk_id=r.chunk_id,
                    source_id=r.source_id,
                    text=r.text,
                    similarity_score=r.similarity_score,
                    metadata=r.metadata,
                )
                for r in top_retrieved
            ]

        # 4. Build context
        context_str = build_context_string(passing_ctx)
        avg_similarity = (
            sum(c.similarity_score for c in passing_ctx) / len(passing_ctx)
            if passing_ctx else 0.0
        )

        # 5. Generate answer
        template_name = self._select_template(task_hint)
        from layer4_intelligence.prompt_templates import render_prompt
        from layer4_intelligence.llm_router import route_query, TaskType

        system, user = render_prompt(
            template_name,
            context=context_str,
            question=question,
        )

        llm_resp = route_query(
            system=system,
            user=user,
            task_type=TaskType.HEAVY_ANALYSIS if len(passing_ctx) > 2 else TaskType.QUICK_QA,
        )
        raw_answer = llm_resp.content

        # 6. Hallucination guard
        if self.use_guard:
            guard_result = guard_instance.verify(
                question=question,
                context=context_str,
                answer=raw_answer,
                avg_similarity=avg_similarity,
            )
            final_answer = guard_result.final_answer
            passed = guard_result.passed
            confidence = guard_result.confidence_score
            verdict = guard_result.verdict
            citation_cov = guard_result.citation_coverage
        else:
            final_answer = raw_answer
            passed = True
            confidence = avg_similarity
            verdict = "UNVERIFIED"
            citation_cov = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "RAG query completed in %.0f ms | guard_passed=%s | confidence=%.2f",
            latency_ms, passed, confidence,
        )

        # Convert passing_ctx back to RetrievedChunk for result
        chunks_used = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                source_id=c.source_id,
                text=c.text,
                similarity_score=c.similarity_score,
                metadata=c.metadata,
            )
            for c in passing_ctx
        ]

        return RAGResult(
            question=question,
            answer=final_answer,
            passed_guard=passed,
            confidence_score=confidence,
            verdict=verdict,
            chunks_used=chunks_used,
            chunks_retrieved=len(raw_chunks),
            chunks_after_filter=len(passing_ctx),
            latency_ms=latency_ms,
            backend_used=llm_resp.backend.value,
            citation_coverage=citation_cov,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_template(task_hint: Optional[str]) -> str:
        mapping = {
            "summarise": "rag_summarise",
            "summarize": "rag_summarise",
            "analyse": "document_analysis",
            "analyze": "document_analysis",
        }
        return mapping.get((task_hint or "").lower(), "rag_answer")

    def add_documents(
        self,
        documents: list[dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """
        Chunk and index documents into the vector store.

        Args:
            documents:     List of dicts with 'text', 'source_id', and optional 'metadata'.
            chunk_size:    Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            Number of chunks indexed.
        """
        if not hasattr(self.vector_store, "_collection"):
            raise RuntimeError("add_documents requires a ChromaAdapter vector store.")

        chunks_data = []
        for doc in documents:
            text = doc["text"]
            source_id = doc.get("source_id", "unknown")
            meta = doc.get("metadata", {})

            # Simple character-level chunking with overlap
            starts = range(0, len(text), chunk_size - chunk_overlap)
            for i, start in enumerate(starts):
                chunk_text = text[start: start + chunk_size].strip()
                if not chunk_text:
                    continue
                chunk_id = f"{source_id}_chunk_{i}"
                chunks_data.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {**meta, "source_id": source_id, "chunk_index": i},
                })

        if not chunks_data:
            logger.warning("No chunks generated — documents may be empty.")
            return 0

        self.vector_store._collection.upsert(
            ids=[c["id"] for c in chunks_data],
            documents=[c["text"] for c in chunks_data],
            metadatas=[c["metadata"] for c in chunks_data],
        )
        logger.info("Indexed %d chunks into ChromaDB.", len(chunks_data))
        return len(chunks_data)