# layer2_processing/entity_extractor.py
# Extracts named entities (people, places, orgs, dates) from text chunks
# Uses phi3:mini via Ollama — runs fully locally

import json
import time
from dataclasses import dataclass, field
from typing import Optional
import ollama
from loguru import logger

from layer2_processing.embedder import EmbeddedChunk


@dataclass
class ExtractedEntities:
    """Holds all named entities found in a single chunk."""
    persons: list[str] = field(default_factory=list)       # People names
    organizations: list[str] = field(default_factory=list) # Companies, agencies
    locations: list[str] = field(default_factory=list)     # Cities, countries
    dates: list[str] = field(default_factory=list)         # Dates, time references
    money: list[str] = field(default_factory=list)         # Money amounts
    misc: list[str] = field(default_factory=list)          # Anything else notable


@dataclass
class EnrichedChunk:
    """An EmbeddedChunk that now also carries extracted entities."""
    text: str
    embedding: list[float]
    chunk_index: int
    total_chunks: int
    source_url: str = ""
    source_title: str = ""
    language: str = "en"
    metadata: dict = field(default_factory=dict)
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)


# The prompt we send to phi3:mini for entity extraction
EXTRACTION_PROMPT = """Extract named entities from the text below.
Return ONLY a JSON object with these exact keys:
- persons: list of people's names
- organizations: list of company, agency, or institution names
- locations: list of cities, countries, or places
- dates: list of dates or time references
- money: list of monetary amounts
- misc: list of other notable named entities

If none found for a category, use an empty list [].
Return ONLY the JSON object, no explanation, no markdown.

Text:
{text}"""


class EntityExtractor:
    """
    Sends each chunk to phi3:mini and asks it to extract named entities.
    Returns enriched chunks with entities attached.
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        retry_attempts: int = 3,
        retry_delay: float = 2.0
    ):
        self.model = model
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        logger.info(f"EntityExtractor ready — model: {self.model}")

    def extract_from_chunks(
        self, chunks: list[EmbeddedChunk]
    ) -> list[EnrichedChunk]:
        """
        Takes a list of EmbeddedChunks, returns EnrichedChunks with entities.
        """
        if not chunks:
            logger.warning("No chunks passed to entity extractor.")
            return []

        enriched = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            logger.info(f"Extracting entities {i + 1}/{total}...")
            entities = self._extract_single(chunk.text)

            enriched.append(EnrichedChunk(
                text=chunk.text,
                embedding=chunk.embedding,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                source_url=chunk.source_url,
                source_title=chunk.source_title,
                language=chunk.language,
                metadata=chunk.metadata,
                entities=entities
            ))

        logger.info(f"Entity extraction complete for {len(enriched)} chunks.")
        return enriched

    def _extract_single(self, text: str) -> ExtractedEntities:
        """
        Sends one chunk to phi3:mini and parses the JSON response.
        Falls back to empty entities if parsing fails.
        """
        prompt = EXTRACTION_PROMPT.format(text=text[:1000])  # Cap at 1000 chars

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}  # Deterministic output for parsing
                )

                raw = response["message"]["content"].strip()
                return self._parse_response(raw)

            except Exception as e:
                logger.warning(f"Extraction attempt {attempt} failed: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)

        logger.error("All retries failed — returning empty entities.")
        return ExtractedEntities()

    def _parse_response(self, raw: str) -> ExtractedEntities:
        """
        Parses the JSON string from phi3:mini into an ExtractedEntities object.
        Handles cases where the model adds extra text around the JSON.
        """
        try:
            # Find the JSON object even if model adds extra text around it
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response.")

            json_str = raw[start:end]
            data = json.loads(json_str)

            return ExtractedEntities(
                persons=data.get("persons", []),
                organizations=data.get("organizations", []),
                locations=data.get("locations", []),
                dates=data.get("dates", []),
                money=data.get("money", []),
                misc=data.get("misc", [])
            )

        except Exception as e:
            logger.warning(f"Failed to parse entity JSON: {e} | Raw: {raw[:200]}")
            return ExtractedEntities()


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from layer2_processing.chunker import TextChunker
    from layer2_processing.embedder import Embedder

    sample = """
    Elon Musk's company SpaceX successfully launched a Falcon 9 rocket from Cape 
    Canaveral, Florida on March 5, 2024. The mission cost approximately $67 million 
    and was contracted by NASA. Meanwhile, Apple Inc. announced a $3.5 billion 
    investment in artificial intelligence research at their headquarters in Cupertino, 
    California. CEO Tim Cook met with President Biden in Washington D.C. last Tuesday 
    to discuss technology policy and regulation.
    """

    # Full pipeline: chunk → embed → extract
    chunker = TextChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text(
        text=sample,
        source_url="https://example.com/tech-news",
        source_title="Tech News March 2024"
    )

    embedder = Embedder()
    embedded = embedder.embed_chunks(chunks)

    extractor = EntityExtractor()
    enriched = extractor.extract_from_chunks(embedded)

    for ec in enriched:
        print(f"\n--- Chunk {ec.chunk_index + 1}/{ec.total_chunks} ---")
        print(f"Persons       : {ec.entities.persons}")
        print(f"Organizations : {ec.entities.organizations}")
        print(f"Locations     : {ec.entities.locations}")
        print(f"Dates         : {ec.entities.dates}")
        print(f"Money         : {ec.entities.money}")
        print(f"Misc          : {ec.entities.misc}")
