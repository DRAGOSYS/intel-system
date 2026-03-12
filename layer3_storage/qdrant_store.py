# layer3_storage/qdrant_store.py
# PURPOSE: Saves and retrieves vector embeddings in Qdrant
# Qdrant is a vector database running locally via Docker

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,          # How to measure similarity between vectors
    VectorParams,      # Settings for our vector collection
    PointStruct,       # A single item to store (vector + metadata)
    Filter,            # For filtering search results
    FieldCondition,    # A single filter condition (e.g. source == "rss")
    MatchValue,        # The value to match in a filter
    MatchText,         # For keyword/text matching inside filters
    SearchRequest,     # Used when doing batch searches
)
from loguru import logger
from typing import Optional
import uuid


# ── Constants ─────────────────────────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"       # Where Qdrant is running (Docker)
COLLECTION_NAME = "intel_documents"        # Name of our vector collection
VECTOR_SIZE = 768                          # nomic-embed-text produces 768-dim vectors
DISTANCE_METRIC = Distance.COSINE          # Cosine similarity (best for text)


# ── Client Setup ──────────────────────────────────────────────────────────────

def get_client() -> QdrantClient:
    """
    Creates and returns a connection to the local Qdrant instance.
    Call this at the start of any function that needs Qdrant.
    """
    return QdrantClient(url=QDRANT_URL)


# ── Collection Management ─────────────────────────────────────────────────────

def create_collection(client: Optional[QdrantClient] = None) -> bool:
    """
    Creates the intel_documents collection in Qdrant if it doesn't exist.
    Safe to call multiple times — won't overwrite existing data.

    Returns True if created or already exists, False on error.
    """
    client = client or get_client()
    try:
        # Get list of existing collections
        existing = [c.name for c in client.get_collections().collections]

        if COLLECTION_NAME in existing:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
            return True

        # Create new collection with our vector settings
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,          # Must match your embedding model output
                distance=DISTANCE_METRIC   # Cosine is standard for semantic text search
            )
        )
        logger.success(f"Created Qdrant collection: '{COLLECTION_NAME}'")
        return True

    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


def get_collection_info(client: Optional[QdrantClient] = None) -> dict:
    """
    Returns basic stats about the collection:
    - How many vectors are stored
    - Collection status
    """
    client = client or get_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        # vectors_count was renamed in newer qdrant-client versions
        total = (
            info.vectors_count
            if hasattr(info, "vectors_count")
            else info.points_count
        )
        return {
            "name": COLLECTION_NAME,
            "total_vectors": total,
            "status": str(info.status)
        }
    except Exception as e:
        logger.error(f"Could not get collection info: {e}")
        return {}


# ── Storing Vectors ───────────────────────────────────────────────────────────

def store_chunk(
    embedding: list[float],     # The vector from your embedder.py
    text: str,                  # The actual chunk text
    metadata: dict,             # Any extra info: source, date, entities, etc.
    chunk_id: Optional[str] = None,
    client: Optional[QdrantClient] = None
) -> Optional[str]:
    """
    Stores a single text chunk + its vector embedding into Qdrant.

    Parameters:
    - embedding : list of floats from nomic-embed-text (must be 768 values)
    - text      : the raw chunk text
    - metadata  : dict with keys like source, url, date, entities, language, etc.
    - chunk_id  : optional string ID (auto-generated UUID if not provided)

    Returns the chunk_id string if successful, None if failed.

    Example metadata:
    {
        "source": "rss",
        "url": "https://example.com/article",
        "title": "Breaking News",
        "date": "2024-01-15",
        "entities": ["Elon Musk", "Tesla"],
        "language": "en",
        "quality_score": 0.85
    }
    """
    client = client or get_client()

    # Generate a unique ID if none provided
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())

    try:
        # Build the payload — everything stored alongside the vector
        payload = {
            "text": text,           # Always store the original text
            **metadata              # Merge in all metadata fields
        }

        # PointStruct = one item in Qdrant (id + vector + payload)
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload=payload
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]          # Can pass a list for batch inserts
        )

        logger.debug(f"Stored chunk {chunk_id[:8]}... | source: {metadata.get('source', 'unknown')}")
        return chunk_id

    except Exception as e:
        logger.error(f"Failed to store chunk: {e}")
        return None


def store_chunks_batch(
    chunks: list[dict],
    client: Optional[QdrantClient] = None
) -> list[str]:
    """
    Stores multiple chunks at once — much faster than calling store_chunk in a loop.

    Each item in chunks must be a dict with keys:
    - "embedding" : list[float]
    - "text"      : str
    - "metadata"  : dict
    - "chunk_id"  : str (optional)

    Returns list of successfully stored chunk IDs.
    """
    client = client or get_client()
    points = []
    ids = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
        payload = {
            "text": chunk["text"],
            **chunk.get("metadata", {})
        }
        points.append(PointStruct(
            id=chunk_id,
            vector=chunk["embedding"],
            payload=payload
        ))
        ids.append(chunk_id)

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logger.success(f"Batch stored {len(points)} chunks into Qdrant.")
        return ids

    except Exception as e:
        logger.error(f"Batch store failed: {e}")
        return []


# ── Searching Vectors ─────────────────────────────────────────────────────────

def semantic_search(
    query_embedding: list[float],   # Embed your query text first, pass result here
    top_k: int = 10,                # How many results to return
    score_threshold: float = 0.5,   # Minimum similarity score (0.0 to 1.0)
    filters: Optional[dict] = None, # Optional metadata filters
    client: Optional[QdrantClient] = None
) -> list[dict]:
    """
    Finds the most semantically similar chunks to a query vector.

    This is the fallback search — used when keyword search finds nothing.
    Higher score = more similar (1.0 is identical, 0.0 is unrelated).

    filters example (only search RSS sources in English):
    {"source": "rss", "language": "en"}

    Returns list of dicts with keys: id, score, text, and all metadata fields.
    """
    client = client or get_client()

    # Build Qdrant filter if provided
    qdrant_filter = _build_filter(filters) if filters else None

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True
        )

        # Format results into clean dicts
        formatted = []
        for hit in response.points:
            result = {
                "id": str(hit.id),
                "score": round(hit.score, 4),
                **hit.payload
            }
            formatted.append(result)

        logger.info(f"Semantic search returned {len(formatted)} results.")
        return formatted

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def keyword_search(
    keyword: str,               # The word or phrase to search for
    field: str = "text",        # Which payload field to search in
    top_k: int = 10,
    filters: Optional[dict] = None,
    client: Optional[QdrantClient] = None
) -> list[dict]:
    """
    Searches for chunks where a specific field contains a keyword.
    Faster and more precise than semantic search for known terms.

    Example: keyword_search("Elon Musk", field="text")
    Example: keyword_search("rss", field="source")

    Returns same format as semantic_search (without a score field).
    """
    client = client or get_client()

    # Build a text-match filter for the keyword
    keyword_condition = FieldCondition(
        key=field,
        match=MatchText(text=keyword)   # MatchText does substring matching
    )

    # Combine with any additional filters
    all_conditions = [keyword_condition]
    if filters:
        for key, value in filters.items():
            all_conditions.append(FieldCondition(
                key=key,
                match=MatchValue(value=value)
            ))

    qdrant_filter = Filter(must=all_conditions)

    try:
        # scroll() retrieves by filter (no vector needed)
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )

        formatted = []
        for point in results:
            result = {
                "id": str(point.id),
                **point.payload
            }
            formatted.append(result)

        logger.info(f"Keyword search '{keyword}' returned {len(formatted)} results.")
        return formatted

    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        return []


def entity_search(
    entity_name: str,           # Name to search for e.g. "Tesla" or "Elon Musk"
    top_k: int = 10,
    client: Optional[QdrantClient] = None
) -> list[dict]:
    """
    Searches for chunks that contain a specific named entity.
    Entities are stored as a list in the payload under "entities".

    This uses MatchText on the text field as a fallback since Qdrant
    doesn't support list-contains natively without indexing.

    Returns same format as keyword_search.
    """
    # Search in the text body — entities appear naturally in text
    return keyword_search(
        keyword=entity_name,
        field="text",
        top_k=top_k,
        client=client
    )


def filter_by_metadata(
    filters: dict,              # e.g. {"source": "rss", "language": "en"}
    top_k: int = 20,
    client: Optional[QdrantClient] = None
) -> list[dict]:
    """
    Retrieves chunks matching exact metadata values.
    Useful for: "give me all chunks from source=pdf" or "language=fr"

    filters: dict of field → exact value pairs (all must match).
    """
    client = client or get_client()

    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]

    try:
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=conditions),
            limit=top_k,
            with_payload=True
        )

        formatted = [{"id": str(p.id), **p.payload} for p in results]
        logger.info(f"Metadata filter returned {len(formatted)} results.")
        return formatted

    except Exception as e:
        logger.error(f"Metadata filter search failed: {e}")
        return []


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_chunk(chunk_id: str, client: Optional[QdrantClient] = None) -> bool:
    """Deletes a single chunk by its ID."""
    client = client or get_client()
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[chunk_id]
        )
        logger.info(f"Deleted chunk: {chunk_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete chunk {chunk_id}: {e}")
        return False


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _build_filter(filters: dict) -> Filter:
    """
    Converts a simple dict like {"source": "rss"} into a Qdrant Filter object.
    Internal helper used by semantic_search.
    """
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test Qdrant connection and basic operations.
    Command: python layer3_storage/qdrant_store.py
    Make sure Docker + Qdrant are running first!
    """
    logger.info("Testing Qdrant store...")

    client = get_client()

    # Step 1: Create collection
    create_collection(client)

    # Step 2: Store a fake chunk (normally embedding comes from embedder.py)
    fake_embedding = [0.1] * 768       # Real embedding would be 768 floats from nomic
    chunk_id = store_chunk(
        embedding=fake_embedding,
        text="Elon Musk announced a new Tesla model in Berlin.",
        metadata={
            "source": "rss",
            "url": "https://example.com/tesla-news",
            "title": "Tesla Berlin Announcement",
            "date": "2024-01-15",
            "entities": ["Elon Musk", "Tesla", "Berlin"],
            "language": "en",
            "quality_score": 0.9
        },
        client=client
    )
    logger.info(f"Stored chunk ID: {chunk_id}")

    # Step 3: Check collection stats
    info = get_collection_info(client)
    logger.info(f"Collection info: {info}")

    # Step 4: Keyword search
    results = keyword_search("Tesla", client=client)
    logger.info(f"Keyword search results: {len(results)} found")
    for r in results:
        logger.info(f"  → {r.get('title')} | {r.get('source')}")

    # Step 5: Entity search
    entity_results = entity_search("Elon Musk", client=client)
    logger.info(f"Entity search results: {len(entity_results)} found")

    logger.success("Qdrant store test complete!")