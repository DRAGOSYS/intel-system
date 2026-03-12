import hashlib
from loguru import logger

class Deduplicator:

    def __init__(self):
        self.seen_hashes = set()

    def get_hash(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def is_duplicate(self, text):
        h = self.get_hash(text)
        if h in self.seen_hashes:
            logger.debug("Duplicate content found, skipping")
            return True
        self.seen_hashes.add(h)
        return False

    def filter_duplicates(self, documents):
        unique = []
        for doc in documents:
            if not self.is_duplicate(doc.get("content", "")):
                unique.append(doc)
        logger.info(f"Kept {len(unique)} unique docs out of {len(documents)}")
        return unique