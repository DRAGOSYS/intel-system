# layer2_processing/embedder.py
# Converts text chunks into vector embeddings using Ollama (nomic-embed-text)

import time
from dataclasses import dataclass, field
from typing import Optional
import ollama
from loguru import logger

from layer2_processing.chunker import Chunk


@dataclass
class EmbeddedChunk:
    """A Chunk that now also carries its vector embedding."""
    text: str
    embedding: list[float]           # The vector — e.g. 768 numbers
    chunk_index: int
    total_chunks: int
    source_url: str = ""
    source_title: str = ""
    language: str = "en"
    metadata: dict = field(default_factory=dict)


class Embedder:
    """
    Sends text to Ollama's nomic-embed-text model and gets back a vector.
    
    nomic-embed-text produces 768-dimensional vectors —
    meaning each chunk becomes a list of 768 float numbers.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        batch_size: int = 8,          # How many chunks to embed at once
        retry_attempts: int = 3,      # Retry on failure
        retry_delay: float = 2.0      # Seconds to wait between retries
    ):
        self.model = model
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        logger.info(f"Embedder ready — model: {self.model}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Takes a list of Chunk objects, returns a list of EmbeddedChunk objects.
        Processes in small batches so RAM stays manageable.
        """
        if not chunks:
            logger.warning("No chunks passed to embedder.")
            return []

        embedded = []
        total = len(chunks)

        # Process in batches
        for batch_start in range(0, total, self.batch_size):
            batch = chunks[batch_start: batch_start + self.batch_size]
            batch_num = (batch_start // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"Embedding batch {batch_num}/{total_batches} "
                        f"({len(batch)} chunks)...")

            for chunk in batch:
                embedded_chunk = self._embed_single(chunk)
                if embedded_chunk:
                    embedded.append(embedded_chunk)

        logger.info(f"Embedded {len(embedded)}/{total} chunks successfully.")
        return embedded

    def _embed_single(self, chunk: Chunk) -> Optional[EmbeddedChunk]:
        """
        Embeds one chunk with retry logic.
        Returns None if all retries fail.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = ollama.embeddings(
                    model=self.model,
                    prompt=chunk.text
                )
                vector = response["embedding"]

                return EmbeddedChunk(
                    text=chunk.text,
                    embedding=vector,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    source_url=chunk.source_url,
                    source_title=chunk.source_title,
                    language=chunk.language,
                    metadata=chunk.metadata
                )

            except Exception as e:
                logger.warning(f"Embedding attempt {attempt} failed: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)

        logger.error(f"All retries failed for chunk {chunk.chunk_index} "
                      f"from '{chunk.source_title}'")
        return None

    def embed_query(self, query_text: str) -> Optional[list[float]]:
        """
        Embeds a single search query string.
        Used later in Layer 4 when searching for similar content.
        """
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=query_text
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return None


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from layer2_processing.chunker import TextChunker

    sample = """
    Artificial intelligence is transforming industries worldwide. Companies are investing 
    billions in AI research and development. The technology promises to automate routine 
    tasks and augment human decision-making. However, concerns about job displacement 
    and ethical implications remain. Governments are beginning to regulate AI applications.
    New frameworks are being developed to ensure responsible use.
    """ * 3

    # Step 1 — chunk the text
    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_text(
        text=sample,
        source_url="https://example.com/ai-article",
        source_title="AI Trends 2024"
    )

    # Step 2 — embed the chunks
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    # Step 3 — show results
    for ec in embedded_chunks:
        print(f"\nChunk {ec.chunk_index + 1}/{ec.total_chunks}")
        print(f"  Text preview : {ec.text[:80]}...")
        print(f"  Vector length: {len(ec.embedding)}")
        print(f"  First 5 vals : {ec.embedding[:5]}")

    # Also test query embedding
    print("\n--- Query Embedding Test ---")
    vec = embedder.embed_query("What is AI doing to jobs?")
    if vec:
        print(f"Query vector length : {len(vec)}")
        print(f"Query first 5 values: {vec[:5]}")
