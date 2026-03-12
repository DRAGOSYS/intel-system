# layer2_processing/pipeline.py
# Connects all 4 Layer 2 steps into one clean pipeline
# Layer 3 will call this directly

from loguru import logger
from layer2_processing.chunker import TextChunker
from layer2_processing.embedder import Embedder
from layer2_processing.entity_extractor import EntityExtractor
from layer2_processing.quality_filter import QualityFilter, EnrichedChunk


class ProcessingPipeline:
    """
    Full Layer 2 pipeline in one class.
    
    Input  : raw text document from Layer 1
    Output : clean, embedded, entity-enriched chunks ready for Layer 3
    
    Flow:
        raw text → chunker → embedder → entity_extractor → quality_filter
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_quality_score: float = 0.5
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder()
        self.extractor = EntityExtractor()
        self.filter = QualityFilter(min_score=min_quality_score)
        logger.info("ProcessingPipeline ready.")

    def process(
        self,
        text: str,
        source_url: str = "",
        source_title: str = "",
        language: str = "en",
        metadata: dict = None
    ) -> list[EnrichedChunk]:
        """
        Runs a raw text document through the full pipeline.
        Returns only the chunks that passed quality filtering.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text received for '{source_title}', skipping.")
            return []

        metadata = metadata or {}

        logger.info(f"Pipeline starting: '{source_title or source_url}'")

        # Step 1 — Chunk
        chunks = self.chunker.chunk_text(
            text=text,
            source_url=source_url,
            source_title=source_title,
            language=language,
            metadata=metadata
        )
        if not chunks:
            logger.warning("No chunks produced, stopping pipeline.")
            return []

        # Step 2 — Embed
        embedded = self.embedder.embed_chunks(chunks)
        if not embedded:
            logger.warning("No embeddings produced, stopping pipeline.")
            return []

        # Step 3 — Extract entities
        enriched = self.extractor.extract_from_chunks(embedded)
        if not enriched:
            logger.warning("No enriched chunks produced, stopping pipeline.")
            return []

        # Step 4 — Filter quality
        passed, _ = self.filter.filter_chunks(enriched)

        logger.info(
            f"Pipeline complete: {len(passed)}/{len(chunks)} chunks passed "
            f"for '{source_title or source_url}'"
        )

        return passed


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Elon Musk's company SpaceX successfully launched a Falcon 9 rocket from Cape 
    Canaveral, Florida on March 5, 2024. The mission cost approximately $67 million 
    and was contracted by NASA. Meanwhile, Apple Inc. announced a $3.5 billion 
    investment in artificial intelligence research at their headquarters in Cupertino, 
    California. CEO Tim Cook met with President Biden in Washington D.C. last Tuesday 
    to discuss technology policy and regulation frameworks for the coming year.
    """ * 4  # Repeat to generate multiple chunks

    pipeline = ProcessingPipeline(chunk_size=300, overlap=50)
    results = pipeline.process(
        text=sample,
        source_url="https://example.com/tech-news",
        source_title="Tech News March 2024"
    )

    print(f"\n{'='*50}")
    print(f"Final output: {len(results)} chunks ready for Layer 3")
    print('='*50)

    for chunk in results:
        print(f"\nChunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
        print(f"  Text preview : {chunk.text[:80]}...")
        print(f"  Vector dims  : {len(chunk.embedding)}")
        print(f"  Persons      : {chunk.entities.persons}")
        print(f"  Orgs         : {chunk.entities.organizations}")
        print(f"  Locations    : {chunk.entities.locations}")
