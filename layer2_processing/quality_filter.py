# layer2_processing/quality_filter.py
# Scores each enriched chunk and filters out low-quality content
# No AI model needed — uses simple but effective rule-based checks

from dataclasses import dataclass, field
from loguru import logger

from layer2_processing.entity_extractor import EnrichedChunk


# Phrases that indicate junk content (cookie banners, nav menus, etc.)
JUNK_PHRASES = [
    "cookie policy", "accept cookies", "privacy policy",
    "terms of service", "subscribe to our newsletter",
    "click here", "follow us on", "all rights reserved",
    "javascript is disabled", "please enable javascript",
    "404 not found", "access denied", "page not found"
]


@dataclass
class ScoredChunk:
    """An EnrichedChunk with a quality score and decision attached."""
    chunk: EnrichedChunk
    score: float              # 0.0 to 1.0 — higher is better
    passed: bool              # True = keep, False = discard
    reasons: list[str] = field(default_factory=list)  # Why it passed/failed


class QualityFilter:
    """
    Scores chunks across 5 criteria and keeps only high-quality ones.

    Scoring breakdown (each out of 1.0, then averaged):
        1. Length score       — is the chunk long enough?
        2. Junk score         — does it contain junk phrases?
        3. Language score     — is it a supported language?
        4. Diversity score    — is it more than just repeated words?
        5. Entity score       — did we find any named entities?
    """

    def __init__(
        self,
        min_score: float = 0.5,           # Minimum score to pass
        min_length: int = 100,            # Minimum characters
        max_length: int = 2000,           # Maximum characters
        supported_languages: list = None  # None = accept all languages
    ):
        self.min_score = min_score
        self.min_length = min_length
        self.max_length = max_length
        self.supported_languages = supported_languages or []

    def filter_chunks(
        self, chunks: list[EnrichedChunk]
    ) -> tuple[list[EnrichedChunk], list[ScoredChunk]]:
        """
        Main method — takes enriched chunks, returns:
          - passed_chunks : list of EnrichedChunk that passed quality check
          - all_scores    : list of ScoredChunk with full scoring details
        """
        if not chunks:
            logger.warning("No chunks passed to quality filter.")
            return [], []

        all_scores = []
        passed_chunks = []

        for chunk in chunks:
            scored = self._score_chunk(chunk)
            all_scores.append(scored)
            if scored.passed:
                passed_chunks.append(chunk)

        total = len(chunks)
        passed = len(passed_chunks)
        dropped = total - passed

        logger.info(
            f"Quality filter: {passed}/{total} passed, {dropped} dropped "
            f"(threshold: {self.min_score})"
        )

        return passed_chunks, all_scores

    def _score_chunk(self, chunk: EnrichedChunk) -> ScoredChunk:
        """Runs all 5 scoring checks and combines them into a final score."""
        reasons = []
        scores = {}

        # 1. Length score
        scores["length"], msg = self._score_length(chunk.text)
        reasons.append(msg)

        # 2. Junk score
        scores["junk"], msg = self._score_junk(chunk.text)
        reasons.append(msg)

        # 3. Language score
        scores["language"], msg = self._score_language(chunk.language)
        reasons.append(msg)

        # 4. Diversity score
        scores["diversity"], msg = self._score_diversity(chunk.text)
        reasons.append(msg)

        # 5. Entity score
        scores["entity"], msg = self._score_entities(chunk.entities)
        reasons.append(msg)

        # Final score = average of all 5
        final_score = sum(scores.values()) / len(scores)
        passed = final_score >= self.min_score

        return ScoredChunk(
            chunk=chunk,
            score=round(final_score, 3),
            passed=passed,
            reasons=reasons
        )

    def _score_length(self, text: str) -> tuple[float, str]:
        """Longer chunks (up to a point) score higher."""
        length = len(text)
        if length < self.min_length:
            return 0.0, f"Too short ({length} chars)"
        if length > self.max_length:
            return 0.7, f"Very long ({length} chars) — acceptable"
        # Scale between min and max
        score = (length - self.min_length) / (self.max_length - self.min_length)
        return round(min(score, 1.0), 3), f"Length OK ({length} chars)"

    def _score_junk(self, text: str) -> tuple[float, str]:
        """Penalizes chunks containing known junk phrases."""
        text_lower = text.lower()
        found = [p for p in JUNK_PHRASES if p in text_lower]
        if found:
            return 0.0, f"Junk phrases found: {found[:3]}"
        return 1.0, "No junk phrases"

    def _score_language(self, language: str) -> tuple[float, str]:
        """Accepts all languages if no filter set, otherwise checks the list."""
        if not self.supported_languages:
            return 1.0, f"Language accepted ({language})"
        if language in self.supported_languages:
            return 1.0, f"Language supported ({language})"
        return 0.0, f"Unsupported language ({language})"

    def _score_diversity(self, text: str) -> tuple[float, str]:
        """
        Checks word diversity — repetitive text scores lower.
        Ratio = unique words / total words.
        """
        words = text.lower().split()
        if not words:
            return 0.0, "No words found"
        ratio = len(set(words)) / len(words)
        if ratio < 0.3:
            return 0.0, f"Too repetitive (diversity: {ratio:.2f})"
        if ratio < 0.5:
            return 0.5, f"Moderate diversity ({ratio:.2f})"
        return 1.0, f"Good diversity ({ratio:.2f})"

    def _score_entities(self, entities) -> tuple[float, str]:
        """Chunks with more named entities are likely more informative."""
        total = (
            len(entities.persons) +
            len(entities.organizations) +
            len(entities.locations) +
            len(entities.dates) +
            len(entities.money) +
            len(entities.misc)
        )
        if total == 0:
            return 0.3, "No entities found"   # Not zero — entity-free text can still be useful
        if total <= 2:
            return 0.6, f"Few entities ({total})"
        if total <= 5:
            return 0.9, f"Good entities ({total})"
        return 1.0, f"Rich entities ({total})"


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from layer2_processing.chunker import TextChunker
    from layer2_processing.embedder import Embedder
    from layer2_processing.entity_extractor import EntityExtractor

    # Two samples — one good, one junk
    good_text = """
    Elon Musk's company SpaceX successfully launched a Falcon 9 rocket from Cape 
    Canaveral, Florida on March 5, 2024. The mission cost approximately $67 million 
    and was contracted by NASA. Meanwhile, Apple Inc. announced a $3.5 billion 
    investment in artificial intelligence research at their headquarters in Cupertino, 
    California. CEO Tim Cook met with President Biden in Washington D.C. last Tuesday 
    to discuss technology policy and regulation frameworks for the coming year.
    """

    junk_text = """
    Accept cookies. Click here to accept our cookie policy and privacy policy.
    Follow us on social media. All rights reserved. Subscribe to our newsletter.
    Click here for more. Accept cookies. Click here. Follow us. All rights reserved.
    """

    chunker = TextChunker(chunk_size=512, overlap=50)
    embedder = Embedder()
    extractor = EntityExtractor()
    quality = QualityFilter(min_score=0.5)

    for label, text in [("GOOD CONTENT", good_text), ("JUNK CONTENT", junk_text)]:
        print(f"\n{'='*50}")
        print(f"Testing: {label}")
        print('='*50)

        chunks = chunker.chunk_text(text=text, source_title=label)
        embedded = embedder.embed_chunks(chunks)
        enriched = extractor.extract_from_chunks(embedded)
        passed, scores = quality.filter_chunks(enriched)

        for s in scores:
            print(f"\nScore  : {s.score} | Passed: {s.passed}")
            for reason in s.reasons:
                print(f"  → {reason}")
                print(f"\nResult : {len(passed)}/{len(scores)} chunks passed")

