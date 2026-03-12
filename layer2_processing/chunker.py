# layer2_processing/chunker.py
# Splits long documents into smaller, overlapping text chunks for embedding

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class Chunk:
    """Represents a single chunk of text from a document."""
    text: str                        # The actual chunk text
    chunk_index: int                 # Position of this chunk in the document (0, 1, 2...)
    total_chunks: int                # Total number of chunks from this document
    source_url: str = ""             # Where the document came from
    source_title: str = ""          # Title of the original document
    language: str = "en"             # Language of the text
    metadata: dict = field(default_factory=dict)  # Any extra info


class TextChunker:
    """
    Splits documents into overlapping chunks.
    
    overlap: how many characters to repeat between chunks
    so context isn't lost at the edges.
    """

    def __init__(
        self,
        chunk_size: int = 512,       # Target characters per chunk
        overlap: int = 100,           # Characters to repeat between chunks
        min_chunk_size: int = 250    # Ignore chunks smaller than this
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        source_url: str = "",
        source_title: str = "",
        language: str = "en",
        metadata: Optional[dict] = None
    ) -> list[Chunk]:
        """
        Main method — takes a document string, returns a list of Chunk objects.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to chunker, skipping.")
            return []

        text = text.strip()
        metadata = metadata or {}

        # Try to split on sentence boundaries first
        chunks_text = self._split_into_chunks(text)

        if not chunks_text:
            logger.warning("No chunks produced from text.")
            return []

        total = len(chunks_text)
        chunks = []

        for i, chunk_text in enumerate(chunks_text):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.min_chunk_size:
                logger.debug(f"Skipping tiny chunk {i} ({len(chunk_text)} chars)")
                continue

            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=i,
                total_chunks=total,
                source_url=source_url,
                source_title=source_title,
                language=language,
                metadata=metadata
            ))

        logger.info(f"Chunked '{source_title or source_url}' → {len(chunks)} chunks")
        return chunks

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Splits text into chunks of ~chunk_size characters with overlap.
        Tries to break at sentence endings (. ! ?) for cleaner chunks.
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                # Last chunk — just take what's left
                chunks.append(text[start:])
                break

            # Try to find a sentence boundary near the end of this chunk
            boundary = self._find_sentence_boundary(text, end)
            chunks.append(text[start:boundary])

            # Move start forward, but step back by overlap so chunks share context
            start = boundary - self.overlap
            if start <= 0:
                start = boundary  # Prevent infinite loop on edge cases

        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Looks backward from 'position' to find a sentence-ending punctuation mark.
        Falls back to the original position if none found within 100 chars.
        """
        search_back = min(100, position)  # How far back to look
        window = text[position - search_back:position]

        # Search for sentence-ending punctuation from right to left
        for i in range(len(window) - 1, -1, -1):
            if window[i] in ".!?\n":
                return position - search_back + i + 1

        return position  # No boundary found, cut at the original position


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Artificial intelligence is transforming industries worldwide. Companies are investing 
    billions in AI research and development. The technology promises to automate routine 
    tasks and augment human decision-making. However, concerns about job displacement 
    and ethical implications remain. Governments are beginning to regulate AI applications.
    New frameworks are being developed to ensure responsible use. The debate continues 
    among experts, policymakers, and the public. Some see it as the greatest technological 
    leap since the internet. Others warn of unforeseen consequences. The future of AI 
    depends on how well humanity navigates these challenges.
    """ * 3  # Repeat to make it long enough to chunk

    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_text(
        text=sample,
        source_url="https://example.com/ai-article",
        source_title="AI Trends 2024"
    )

    for c in chunks:
        print(f"\n--- Chunk {c.chunk_index + 1}/{c.total_chunks} ({len(c.text)} chars) ---")
        print(c.text[:120], "...")


