# layer3_storage/hybrid_search.py
# PURPOSE: Advanced search with scoring, ranking, and result merging
# Builds on top of storage_manager's basic cascade search
# Adds: score normalization, result re-ranking, deduplication, and filtering

from loguru import logger
from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from layer3_storage import qdrant_store as qdrant
from layer3_storage import sqlite_store as sqlite


# ── Score Normalization ───────────────────────────────────────────────────────

def _normalize_scores(results: list[dict], score_field: str = "score") -> list[dict]:
    """
    Normalizes scores to a 0.0–1.0 range.
    This lets us fairly compare scores from different search methods.

    For example:
    - Semantic search returns scores like 0.82, 0.75, 0.61
    - We normalize so the best result = 1.0, worst = 0.0
    """
    scores = [r.get(score_field, 0.0) for r in results]
    if not scores:
        return results

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    for r in results:
        raw = r.get(score_field, 0.0)
        # Avoid division by zero if all scores are equal
        r[score_field] = (raw - min_score) / score_range if score_range > 0 else 1.0

    return results


# ── Result Re-ranking ─────────────────────────────────────────────────────────

def _rerank(results: list[dict]) -> list[dict]:
    """
    Re-ranks results using a weighted combination of:
    - Search score     (how relevant the result is)       weight: 0.5
    - Quality score    (how good the source content is)   weight: 0.3
    - Recency boost    (newer content ranks higher)        weight: 0.2

    This gives you smarter ranking than pure similarity score.
    """
    from datetime import datetime

    # Get current year for recency calculation
    current_year = datetime.now().year

    for r in results:
        search_score = r.get("score", 0.5)
        quality_score = r.get("quality_score", 0.5)

        # Calculate recency score from date field
        recency_score = 0.5  # Default if no date
        date_str = r.get("date", "")
        if date_str:
            try:
                # Parse year from date string (handles "2024-01-15" format)
                year = int(date_str[:4])
                # Score: current year = 1.0, 5 years ago = 0.0
                recency_score = max(0.0, min(1.0, (year - (current_year - 5)) / 5))
            except Exception:
                pass

        # Weighted final score
        r["final_score"] = (
            0.5 * search_score +
            0.3 * quality_score +
            0.2 * recency_score
        )

    # Sort by final score descending
    return sorted(results, key=lambda x: x.get("final_score", 0.0), reverse=True)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(results: list[dict]) -> list[dict]:
    """
    Removes duplicate results based on URL or text similarity.
    Keeps the highest-scoring version of each duplicate.

    Two results are considered duplicates if they share the same URL.
    """
    seen_urls = set()
    seen_ids = set()
    unique = []

    for r in results:
        rid = r.get("id", "")
        url = r.get("url", "")

        # Skip if we've seen this ID or URL before
        if rid in seen_ids:
            continue
        if url and url in seen_urls:
            continue

        seen_ids.add(rid)
        if url:
            seen_urls.add(url)
        unique.append(r)

    return unique


# ── Main Hybrid Search ────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    query_embedding: Optional[list[float]] = None,  # For semantic fallback
    source: Optional[str] = None,                   # Filter: "rss", "pdf", etc.
    language: Optional[str] = None,                 # Filter: "en", "fr", etc.
    date_from: Optional[str] = None,                # Filter: "2024-01-01"
    date_to: Optional[str] = None,                  # Filter: "2024-12-31"
    min_quality: float = 0.0,                       # Minimum quality score
    top_k: int = 10,                                # Max results to return
    use_rerank: bool = True                         # Whether to re-rank results
) -> list[dict]:
    """
    The main search function for your intel system.
    Combines ALL search strategies and returns the best results.

    Search pipeline:
    1. Keyword search  → fast, exact text matching
    2. Entity search   → finds named entities (people, orgs, places)
    3. Semantic search → meaning-based fallback (needs query_embedding)
    4. Metadata filter → applies date/source/language/quality filters
    5. Deduplicate     → removes duplicate results
    6. Re-rank         → scores by relevance + quality + recency
    7. Return top_k    → returns best results

    Parameters:
    - query           : search term or natural language question
    - query_embedding : vector of query text (from embedder.py)
    - source          : optional source filter
    - language        : optional language filter
    - date_from       : optional start date filter
    - date_to         : optional end date filter
    - min_quality     : minimum quality score filter
    - top_k           : number of results to return
    - use_rerank      : whether to apply re-ranking (default True)

    Returns list of result dicts, sorted by final_score descending.
    """
    all_results = []
    seen_ids = set()

    logger.info(f"[HybridSearch] Query: '{query}'")

    # ── Step 1: Keyword Search ───────────────────────────────────────────────
    qdrant_filters = {}
    if source:
        qdrant_filters["source"] = source
    if language:
        qdrant_filters["language"] = language

    keyword_hits = qdrant.keyword_search(
        keyword=query,
        field="text",
        top_k=top_k * 2,    # Fetch extra — we'll trim after merging
        filters=qdrant_filters if qdrant_filters else None
    )
    for r in keyword_hits:
        if r["id"] not in seen_ids:
            r["search_strategy"] = "keyword"
            r.setdefault("score", 0.7)   # Default score for keyword hits
            all_results.append(r)
            seen_ids.add(r["id"])

    logger.info(f"[HybridSearch] Keyword: {len(keyword_hits)} hits")

    # ── Step 2: Entity Search ────────────────────────────────────────────────
    entity_hits = sqlite.search_by_entity(query, limit=top_k * 2)
    for r in entity_hits:
        if r["id"] not in seen_ids:
            r["search_strategy"] = "entity"
            r.setdefault("score", 0.6)
            all_results.append(r)
            seen_ids.add(r["id"])

    logger.info(f"[HybridSearch] Entity: {len(entity_hits)} hits")

    # ── Step 3: Semantic Search (fallback) ───────────────────────────────────
    if query_embedding:
        semantic_hits = qdrant.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            score_threshold=0.4,    # Slightly lower threshold for more recall
            filters=qdrant_filters if qdrant_filters else None
        )
        for r in semantic_hits:
            if r["id"] not in seen_ids:
                r["search_strategy"] = "semantic"
                all_results.append(r)
                seen_ids.add(r["id"])

        logger.info(f"[HybridSearch] Semantic: {len(semantic_hits)} hits")

    # ── Step 4: Metadata Filtering ───────────────────────────────────────────
    # Apply date and quality filters to all collected results
    if date_from or date_to or min_quality > 0:
        filtered = []
        for r in all_results:
            # Date filter
            doc_date = r.get("date", "")
            if date_from and doc_date and doc_date < date_from:
                continue
            if date_to and doc_date and doc_date > date_to:
                continue
            # Quality filter
            if r.get("quality_score", 0.0) < min_quality:
                continue
            filtered.append(r)
        all_results = filtered
        logger.info(f"[HybridSearch] After filters: {len(all_results)} results")

    # ── Step 5: Deduplicate ───────────────────────────────────────────────────
    all_results = _deduplicate(all_results)
    logger.info(f"[HybridSearch] After dedup: {len(all_results)} results")

    # ── Step 6: Normalize scores ──────────────────────────────────────────────
    all_results = _normalize_scores(all_results)

    # ── Step 7: Re-rank ───────────────────────────────────────────────────────
    if use_rerank:
        all_results = _rerank(all_results)

    # ── Step 8: Return top_k ─────────────────────────────────────────────────
    final = all_results[:top_k]
    logger.success(f"[HybridSearch] Returning {len(final)} results for '{query}'")
    return final


# ── Specialized Search Functions ──────────────────────────────────────────────

def search_by_topic(
    topic: str,
    query_embedding: Optional[list[float]] = None,
    top_k: int = 10
) -> list[dict]:
    """
    Searches across all sources for a specific topic.
    Good for broad research queries like "climate change" or "AI regulation".
    """
    return hybrid_search(
        query=topic,
        query_embedding=query_embedding,
        top_k=top_k,
        use_rerank=True
    )


def search_recent(
    query: str,
    query_embedding: Optional[list[float]] = None,
    days_back: int = 7,
    top_k: int = 10
) -> list[dict]:
    """
    Searches only recent documents from the last N days.
    Good for "latest news on X" type queries.
    """
    from datetime import datetime, timedelta
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    return hybrid_search(
        query=query,
        query_embedding=query_embedding,
        date_from=date_from,
        top_k=top_k,
        use_rerank=True
    )


def search_by_source(
    query: str,
    source: str,
    query_embedding: Optional[list[float]] = None,
    top_k: int = 10
) -> list[dict]:
    """
    Searches only within a specific source type.
    Example: search_by_source("Tesla earnings", source="pdf")
    """
    return hybrid_search(
        query=query,
        query_embedding=query_embedding,
        source=source,
        top_k=top_k
    )


def search_high_quality(
    query: str,
    query_embedding: Optional[list[float]] = None,
    min_quality: float = 0.75,
    top_k: int = 10
) -> list[dict]:
    """
    Returns only high-quality results above a quality threshold.
    Good for research tasks where accuracy matters more than recall.
    """
    return hybrid_search(
        query=query,
        query_embedding=query_embedding,
        min_quality=min_quality,
        top_k=top_k
    )


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test hybrid search.
    Command: python layer3_storage/hybrid_search.py
    Make sure Docker + Qdrant are running and storage_manager test
    has been run first (so there's data to search)!
    """
    logger.info("Testing Hybrid Search...")

    # Fake embedding — in real use this comes from embedder.py
    fake_embedding = [0.1] * 768

    # Test 1: Full hybrid search
    logger.info("--- Test 1: Full hybrid search ---")
    results = hybrid_search(
        query="SpaceX",
        query_embedding=fake_embedding,
        top_k=5
    )
    for r in results:
        logger.info(
            f"  [{r.get('search_strategy')}] "
            f"score={r.get('final_score', 0):.3f} | "
            f"{r.get('title', r.get('text', '')[:50])}"
        )

    # Test 2: Recent search
    logger.info("--- Test 2: Recent search (last 30 days) ---")
    recent = search_recent("Tesla", fake_embedding, days_back=30)
    logger.info(f"Recent results: {len(recent)}")

    # Test 3: High quality search
    logger.info("--- Test 3: High quality search ---")
    quality = search_high_quality("Elon Musk", fake_embedding, min_quality=0.5)
    logger.info(f"High quality results: {len(quality)}")
    for r in quality:
        logger.info(
            f"  quality={r.get('quality_score', 0):.2f} | "
            f"{r.get('title', r.get('text', '')[:50])}"
        )

    logger.success("Hybrid search test complete!")