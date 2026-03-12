# layer3_storage/sqlite_store.py
# PURPOSE: Stores structured metadata about every document and chunk
# SQLite is a simple local database — no server needed, just a .db file
# Think of it as a spreadsheet you can query with code

import sqlite3          # Built into Python — no installation needed
import json             # For storing lists (like entities) as text
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime


# ── Database Location ─────────────────────────────────────────────────────────

# The .db file will be created here automatically on first run
DB_PATH = Path("layer3_storage/intel.db")


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database file.
    Also enables foreign keys and returns dict-like rows.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Create folder if missing

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row   # Makes rows behave like dicts (row["title"])
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Table Creation ────────────────────────────────────────────────────────────

def create_tables() -> bool:
    """
    Creates two tables if they don't exist:

    1. documents — one row per original document (article, PDF, etc.)
    2. chunks    — one row per chunk (piece of a document)

    Safe to call multiple times — won't overwrite existing data.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # ── Table 1: documents ───────────────────────────────────────────────
        # Stores info about the full original document
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,   -- Unique ID (UUID)
                title       TEXT,               -- Article/document title
                source      TEXT,               -- Where it came from: rss, pdf, web, etc.
                url         TEXT UNIQUE,        -- Original URL (must be unique)
                date        TEXT,               -- Publication date (ISO format)
                language    TEXT DEFAULT 'en',  -- Language code
                author      TEXT,               -- Author name if available
                entities    TEXT,               -- JSON list: ["Elon Musk", "Tesla"]
                summary     TEXT,               -- Short summary if available
                quality_score REAL DEFAULT 0.0, -- Quality score from quality_filter.py
                created_at  TEXT DEFAULT (datetime('now')), -- When we ingested it
                chunk_count INTEGER DEFAULT 0   -- How many chunks this doc produced
            )
        """)

        # ── Table 2: chunks ──────────────────────────────────────────────────
        # Stores info about each chunk (piece of a document)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,   -- Same ID as stored in Qdrant
                document_id TEXT,               -- Links back to documents table
                text        TEXT,               -- The actual chunk text
                chunk_index INTEGER,            -- Position in document (0, 1, 2...)
                token_count INTEGER,            -- Approx size of chunk
                entities    TEXT,               -- JSON list of entities in this chunk
                quality_score REAL DEFAULT 0.0,
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        conn.commit()
        logger.success("SQLite tables ready.")
        return True

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False
    finally:
        conn.close()    # Always close the connection when done


# ── Storing Documents ─────────────────────────────────────────────────────────

def store_document(
    doc_id: str,
    title: str,
    source: str,
    url: Optional[str] = None,
    date: Optional[str] = None,
    language: str = "en",
    author: Optional[str] = None,
    entities: Optional[list] = None,
    summary: Optional[str] = None,
    quality_score: float = 0.0,
    chunk_count: int = 0
) -> bool:
    """
    Saves one document's metadata to the documents table.
    If a document with the same URL already exists, it updates it (upsert).

    Parameters:
    - doc_id        : unique string ID for this document
    - title         : document title
    - source        : "rss", "pdf", "web", "news", etc.
    - url           : original URL (optional for PDFs)
    - date          : publication date as string e.g. "2024-01-15"
    - language      : ISO language code e.g. "en", "fr"
    - author        : author name if known
    - entities      : list of named entities e.g. ["Tesla", "Elon Musk"]
    - summary       : short text summary
    - quality_score : float 0.0–1.0 from quality_filter.py
    - chunk_count   : how many chunks this document was split into

    Returns True if saved successfully.
    """
    conn = get_connection()
    try:
        # Convert entities list to JSON string for storage
        entities_json = json.dumps(entities or [])

        # INSERT OR REPLACE = if URL exists, overwrite it (upsert)
        conn.execute("""
            INSERT OR REPLACE INTO documents
                (id, title, source, url, date, language, author,
                 entities, summary, quality_score, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, title, source, url, date, language, author,
              entities_json, summary, quality_score, chunk_count))

        conn.commit()
        logger.debug(f"Stored document: '{title[:50]}' | source={source}")
        return True

    except Exception as e:
        logger.error(f"Failed to store document '{title}': {e}")
        return False
    finally:
        conn.close()


def store_chunk_metadata(
    chunk_id: str,
    document_id: str,
    text: str,
    chunk_index: int = 0,
    token_count: int = 0,
    entities: Optional[list] = None,
    quality_score: float = 0.0
) -> bool:
    """
    Saves one chunk's metadata to the chunks table.
    The chunk_id must match the ID used in Qdrant (so we can cross-reference).

    Parameters:
    - chunk_id    : same UUID used when storing in Qdrant
    - document_id : ID of the parent document
    - text        : the chunk text content
    - chunk_index : position within the document (0 = first chunk)
    - token_count : approximate word/token count
    - entities    : named entities found in this chunk
    - quality_score : quality score for this specific chunk
    """
    conn = get_connection()
    try:
        entities_json = json.dumps(entities or [])

        conn.execute("""
            INSERT OR REPLACE INTO chunks
                (id, document_id, text, chunk_index, token_count,
                 entities, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, document_id, text, chunk_index,
              token_count, entities_json, quality_score))

        conn.commit()
        logger.debug(f"Stored chunk {chunk_index} for doc {document_id[:8]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to store chunk metadata: {e}")
        return False
    finally:
        conn.close()


# ── Retrieving Documents ──────────────────────────────────────────────────────

def get_document(doc_id: str) -> Optional[dict]:
    """
    Fetches a single document by its ID.
    Returns a dict or None if not found.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if row:
            return _row_to_dict(row)
        return None
    finally:
        conn.close()


def get_document_by_url(url: str) -> Optional[dict]:
    """
    Checks if we already have a document from this URL.
    Useful for deduplication — don't re-ingest what we already have.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE url = ?", (url,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def get_chunks_for_document(document_id: str) -> list[dict]:
    """
    Returns all chunks belonging to a specific document, in order.
    Useful for reconstructing the full document text.
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
        """, (document_id,)).fetchall()

        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def search_documents(
    keyword: Optional[str] = None,     # Search in title or summary
    source: Optional[str] = None,      # Filter by source type
    language: Optional[str] = None,    # Filter by language
    date_from: Optional[str] = None,   # Start date "2024-01-01"
    date_to: Optional[str] = None,     # End date   "2024-12-31"
    min_quality: float = 0.0,          # Minimum quality score
    limit: int = 20
) -> list[dict]:
    """
    Flexible document search with multiple optional filters.
    All filters are optional — only applied if provided.

    Example: search_documents(source="rss", language="en", min_quality=0.7)
    Example: search_documents(keyword="Tesla", date_from="2024-01-01")
    """
    conn = get_connection()
    try:
        # Build query dynamically based on which filters are provided
        query = "SELECT * FROM documents WHERE 1=1"  # 1=1 lets us append ANDs cleanly
        params = []

        if keyword:
            query += " AND (title LIKE ? OR summary LIKE ?)"
            params += [f"%{keyword}%", f"%{keyword}%"]   # % = wildcard

        if source:
            query += " AND source = ?"
            params.append(source)

        if language:
            query += " AND language = ?"
            params.append(language)

        if date_from:
            query += " AND date >= ?"
            params.append(date_from)

        if date_to:
            query += " AND date <= ?"
            params.append(date_to)

        if min_quality > 0:
            query += " AND quality_score >= ?"
            params.append(min_quality)

        query += " ORDER BY date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    except Exception as e:
        logger.error(f"Document search failed: {e}")
        return []
    finally:
        conn.close()


def search_by_entity(entity_name: str, limit: int = 20) -> list[dict]:
    """
    Finds documents that mention a specific entity.
    Searches inside the JSON-stored entities list using LIKE.

    Example: search_by_entity("Elon Musk")
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM documents
            WHERE entities LIKE ?
            ORDER BY date DESC
            LIMIT ?
        """, (f"%{entity_name}%", limit)).fetchall()

        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Returns a summary of what's stored in the database.
    Useful for dashboards and health checks.
    """
    conn = get_connection()
    try:
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Count documents per source type
        source_counts = conn.execute("""
            SELECT source, COUNT(*) as count
            FROM documents
            GROUP BY source
            ORDER BY count DESC
        """).fetchall()

        # Most recent ingestion date
        latest = conn.execute(
            "SELECT MAX(created_at) FROM documents"
        ).fetchone()[0]

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "sources": {row["source"]: row["count"] for row in source_counts},
            "latest_ingestion": latest
        }
    except Exception as e:
        logger.error(f"Could not get stats: {e}")
        return {}
    finally:
        conn.close()


# ── Helper ────────────────────────────────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row) -> dict:
    """
    Converts a database row to a regular Python dict.
    Also parses the entities JSON string back into a list.
    """
    d = dict(row)
    # Parse entities back from JSON string to list
    if "entities" in d and d["entities"]:
        try:
            d["entities"] = json.loads(d["entities"])
        except Exception:
            d["entities"] = []
    return d


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test SQLite storage.
    Command: python layer3_storage/sqlite_store.py
    No Docker needed — just Python!
    """
    logger.info("Testing SQLite store...")

    # Step 1: Create tables
    create_tables()

    # Step 2: Store a fake document
    import uuid
    doc_id = str(uuid.uuid4())

    store_document(
        doc_id=doc_id,
        title="Tesla Announces New Berlin Gigafactory Expansion",
        source="rss",
        url="https://example.com/tesla-berlin-2024",
        date="2024-01-15",
        language="en",
        author="John Doe",
        entities=["Elon Musk", "Tesla", "Berlin", "Gigafactory"],
        summary="Tesla plans to expand its Berlin factory to double production capacity.",
        quality_score=0.88,
        chunk_count=3
    )
    logger.info(f"Stored document with ID: {doc_id}")

    # Step 3: Store a fake chunk
    chunk_id = str(uuid.uuid4())
    store_chunk_metadata(
        chunk_id=chunk_id,
        document_id=doc_id,
        text="Elon Musk announced plans to expand the Berlin Gigafactory.",
        chunk_index=0,
        token_count=12,
        entities=["Elon Musk", "Berlin", "Gigafactory"],
        quality_score=0.88
    )
    logger.info(f"Stored chunk with ID: {chunk_id}")

    # Step 4: Retrieve document by ID
    doc = get_document(doc_id)
    logger.info(f"Retrieved doc: {doc['title']} | entities: {doc['entities']}")

    # Step 5: Search by keyword
    results = search_documents(keyword="Tesla")
    logger.info(f"Keyword search 'Tesla': {len(results)} result(s)")

    # Step 6: Search by entity
    entity_results = search_by_entity("Elon Musk")
    logger.info(f"Entity search 'Elon Musk': {len(entity_results)} result(s)")

    # Step 7: Check stats
    stats = get_stats()
    logger.info(f"DB Stats: {stats}")

    logger.success("SQLite store test complete!")