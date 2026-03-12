# layer3_storage/storage_manager.py
# PURPOSE: The single entry point for all storage operations
# Instead of calling qdrant_store and sqlite_store separately,
# everything in your system calls THIS file only.
# Think of it as a manager that delegates work to two assistants.

from loguru import logger
from typing import Optional
import uuid

# Import our two storage layers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # So Python can find our modules

from layer3_storage import qdrant_store as qdrant
from layer3_storage import sqlite_store as sqlite


# ── Initialization ────────────────────────────────────────────────────────────

def initialize() -> bool:
    """
    Sets up both storage systems on first run.
    Call this ONCE when your application starts.

    - Creates Qdrant collection (if not exists)
    - Creates SQLite tables (if not exists)

    Returns True if both are ready.
    """
    logger.info("Initializing storage systems...")

    qdrant_ok = qdrant.create_collection()
    sqlite_ok = sqlite.create_tables()

    if qdrant_ok and sqlite_ok:
        logger.success("Both storage systems ready.")
        return True
    else:
        logger.error(f"Storage init failed — Qdrant: {qdrant_ok}, SQLite: {sqlite_ok}")
        return False


# ── Storing a Full Document ───────────────────────────────────────────────────

def store_document_with_chunks(
    title: str,
    source: str,
    chunks: list[dict],             # List of chunk dicts from Layer 2
    url: Optional[str] = None,
    date: Optional[str] = None,
    language: str = "en",
    author: Optional[str] = None,
    entities: Optional[list] = None,
    summary: Optional[str] = None,
    quality_score: float = 0.0,
    doc_id: Optional[str] = None
) -> Optional[str]:
    """
    The MAIN function you will call from Layer 2 output.
    Saves a full document AND all its chunks in one call.

    Each item in chunks must be a dict with:
    - "embedding"     : list[float] — vector from embedder.py
    - "text"          : str         — chunk text
    - "chunk_index"   : int         — position (0, 1, 2...)
    - "token_count"   : int         — size of chunk
    - "entities"      : list        — named entities in this chunk
    - "quality_score" : float       — quality of this chunk

    Example usage:
        storage_manager.store_document_with_chunks(
            title="Tesla Berlin News",
            source="rss",
            url="https://example.com/article",
            chunks=[
                {
                    "embedding": [0.1, 0.2, ...],  # 768 floats
                    "text": "Elon Musk announced...",
                    "chunk_index": 0,
                    "token_count": 45,
                    "entities": ["Elon Musk", "Tesla"],
                    "quality_score": 0.88
                },
                ...
            ],
            entities=["Elon Musk", "Tesla", "Berlin"],
            quality_score=0.88
        )

    Returns the document ID string if successful, None if failed.
    """

    # ── Step 1: Deduplication check ──────────────────────────────────────────
    # Don't re-store something we already have
    if url:
        existing = sqlite.get_document_by_url(url)
        if existing:
            logger.info(f"Document already exists: '{title[:50]}' — skipping.")
            return existing["id"]

    # ── Step 2: Generate document ID ─────────────────────────────────────────
    doc_id = doc_id or str(uuid.uuid4())

    # ── Step 3: Save document metadata to SQLite ─────────────────────────────
    sqlite_ok = sqlite.store_document(
        doc_id=doc_id,
        title=title,
        source=source,
        url=url,
        date=date,
        language=language,
        author=author,
        entities=entities,
        summary=summary,
        quality_score=quality_score,
        chunk_count=len(chunks)
    )

    if not sqlite_ok:
        logger.error(f"Failed to save document metadata for '{title}'")
        return None

    # ── Step 4: Save each chunk to BOTH Qdrant and SQLite ────────────────────
    qdrant_chunks = []      # Batch list for Qdrant
    saved_chunk_ids = []

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())

        # Prepare chunk metadata for Qdrant payload
        chunk_metadata = {
            "document_id": doc_id,
            "source": source,
            "title": title,
            "url": url or "",
            "date": date or "",
            "language": language,
            "chunk_index": chunk.get("chunk_index", 0),
            "entities": chunk.get("entities", []),
            "quality_score": chunk.get("quality_score", 0.0)
        }

        # Add to Qdrant batch list
        qdrant_chunks.append({
            "chunk_id": chunk_id,
            "embedding": chunk["embedding"],
            "text": chunk["text"],
            "metadata": chunk_metadata
        })

        # Save chunk metadata to SQLite immediately
        sqlite.store_chunk_metadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=chunk["text"],
            chunk_index=chunk.get("chunk_index", 0),
            token_count=chunk.get("token_count", 0),
            entities=chunk.get("entities", []),
            quality_score=chunk.get("quality_score", 0.0)
        )

        saved_chunk_ids.append(chunk_id)

    # ── Step 5: Batch store all chunks in Qdrant at once (faster) ────────────
    if qdrant_chunks:
        stored_ids = qdrant.store_chunks_batch(qdrant_chunks)
        if not stored_ids:
            logger.warning(f"Qdrant batch store failed for document '{title}'")
        else:
            logger.success(
                f"Stored '{title[:50]}' | "
                f"{len(stored_ids)} chunks | source={source}"
            )

    return doc_id


# ── Retrieving a Document ─────────────────────────────────────────────────────

def get_document(doc_id: str) -> Optional[dict]:
    """
    Retrieves a document's metadata from SQLite by ID.
    Returns a dict or None if not found.
    """
    return sqlite.get_document(doc_id)


def get_document_chunks(doc_id: str) -> list[dict]:
    """
    Retrieves all chunks for a document from SQLite, in order.
    Useful for reconstructing the full text of a document.
    """
    return sqlite.get_chunks_for_document(doc_id)


def document_exists(url: str) -> bool:
    """
    Quick check: have we already ingested this URL?
    Use this before fetching a page to avoid duplicate work.

    Example:
        if storage_manager.document_exists(url):
            skip this article
    """
    return sqlite.get_document_by_url(url) is not None


# ── Search Routing ────────────────────────────────────────────────────────────

def search(
    query: str,
    query_embedding: Optional[list[float]] = None,  # Required for semantic search
    source: Optional[str] = None,                   # Filter by source
    language: Optional[str] = None,                 # Filter by language
    top_k: int = 10
) -> list[dict]:
    """
    Smart search that tries multiple strategies in order:

    1. Keyword search in SQLite (fast, exact)
    2. Entity search in SQLite (if keyword finds nothing)
    3. Semantic search in Qdrant (if both above find nothing)

    This is the cascade strategy — precise first, broad as fallback.

    Parameters:
    - query           : the search term or question
    - query_embedding : vector of the query (needed for semantic fallback)
    - source          : optional filter e.g. "rss", "pdf"
    - language        : optional filter e.g. "en"
    - top_k           : max results to return

    Returns combined, deduplicated list of result dicts.
    """
    results = []
    seen_ids = set()    # Track IDs to avoid duplicates across strategies

    # ── Strategy 1: Keyword search ───────────────────────────────────────────
    logger.info(f"[Search] Trying keyword search for: '{query}'")
    filters = {}
    if source:
        filters["source"] = source
    if language:
        filters["language"] = language

    keyword_results = qdrant.keyword_search(
        keyword=query,
        field="text",
        top_k=top_k,
        filters=filters if filters else None
    )

    for r in keyword_results:
        if r["id"] not in seen_ids:
            r["search_strategy"] = "keyword"    # Tag how it was found
            results.append(r)
            seen_ids.add(r["id"])

    logger.info(f"[Search] Keyword found: {len(keyword_results)} results")

    # ── Strategy 2: Entity search ────────────────────────────────────────────
    # Run in parallel with keyword — catches entity mentions keyword might miss
    logger.info(f"[Search] Trying entity search for: '{query}'")
    entity_results = sqlite.search_by_entity(query, limit=top_k)

    # Entity results come from SQLite — need to enrich format slightly
    for r in entity_results:
        if r["id"] not in seen_ids:
            r["search_strategy"] = "entity"
            results.append(r)
            seen_ids.add(r["id"])

    logger.info(f"[Search] Entity found: {len(entity_results)} results")

    # ── Strategy 3: Semantic search (fallback) ───────────────────────────────
    # Only run if we have an embedding AND results so far are sparse
    if query_embedding and len(results) < top_k:
        logger.info(f"[Search] Falling back to semantic search...")

        semantic_results = qdrant.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=0.5,
            filters=filters if filters else None
        )

        for r in semantic_results:
            if r["id"] not in seen_ids:
                r["search_strategy"] = "semantic"
                results.append(r)
                seen_ids.add(r["id"])

        logger.info(f"[Search] Semantic found: {len(semantic_results)} results")

    logger.success(f"[Search] Total results: {len(results)} for query: '{query}'")
    return results[:top_k]  # Cap at top_k total


# ── Stats & Health ────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Returns combined stats from both storage systems.
    Use this for health checks and dashboard displays.
    """
    sqlite_stats = sqlite.get_stats()
    qdrant_info = qdrant.get_collection_info()

    return {
        "sqlite": sqlite_stats,
        "qdrant": qdrant_info
    }


def health_check() -> bool:
    """
    Checks if both storage systems are reachable and working.
    Returns True if everything is healthy.
    """
    try:
        # Check Qdrant
        qdrant_info = qdrant.get_collection_info()
        qdrant_ok = bool(qdrant_info)

        # Check SQLite
        sqlite_stats = sqlite.get_stats()
        sqlite_ok = isinstance(sqlite_stats, dict)

        if qdrant_ok and sqlite_ok:
            logger.success("Health check passed — both stores are healthy.")
        else:
            logger.warning(f"Health check issues — Qdrant: {qdrant_ok}, SQLite: {sqlite_ok}")

        return qdrant_ok and sqlite_ok

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test the full storage pipeline.
    Command: python layer3_storage/storage_manager.py
    Make sure Docker + Qdrant are running!
    """
    logger.info("Testing Storage Manager...")

    # Step 1: Initialize both stores
    initialize()

    # Step 2: Health check
    health_check()

    # Step 3: Store a full document with chunks
    # Fake embeddings — in real use these come from embedder.py
    fake_embedding = [0.1] * 768

    doc_id = store_document_with_chunks(
        title="SpaceX Launches New Starship Mission",
        source="rss",
        url="https://example.com/spacex-starship-2024",
        date="2024-03-01",
        language="en",
        author="Jane Smith",
        entities=["Elon Musk", "SpaceX", "Starship", "NASA"],
        summary="SpaceX successfully launched Starship on its latest test flight.",
        quality_score=0.92,
        chunks=[
            {
                "embedding": fake_embedding,
                "text": "SpaceX successfully launched the Starship rocket from Boca Chica.",
                "chunk_index": 0,
                "token_count": 12,
                "entities": ["SpaceX", "Starship", "Boca Chica"],
                "quality_score": 0.92
            },
            {
                "embedding": fake_embedding,
                "text": "Elon Musk called it a milestone for human spaceflight and Mars missions.",
                "chunk_index": 1,
                "token_count": 13,
                "entities": ["Elon Musk", "Mars"],
                "quality_score": 0.90
            }
        ]
    )
    logger.info(f"Stored document ID: {doc_id}")

    # Step 4: Test deduplication — storing same URL again should skip
    logger.info("Testing deduplication...")
    dup_id = store_document_with_chunks(
        title="SpaceX Launches New Starship Mission",
        source="rss",
        url="https://example.com/spacex-starship-2024",  # Same URL
        chunks=[{"embedding": fake_embedding, "text": "duplicate", "chunk_index": 0}]
    )
    logger.info(f"Duplicate store returned: {dup_id} (should match original)")

    # Step 5: Test search cascade
    logger.info("Testing search cascade...")
    results = search(
        query="SpaceX",
        query_embedding=fake_embedding,  # For semantic fallback
        top_k=5
    )
    for r in results:
        logger.info(
            f"  → [{r.get('search_strategy')}] "
            f"{r.get('title', r.get('text', '')[:50])}"
        )

    # Step 6: Check combined stats
    stats = get_stats()
    logger.info(f"Combined stats: {stats}")

    logger.success("Storage Manager test complete!")