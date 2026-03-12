import re
from loguru import logger

class TextCleaner:

    def clean(self, text):
        if not text:
            return ""
        
        text = self.remove_html(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_boilerplate(text)
        
        logger.debug(f"Cleaned text length: {len(text)} chars")
        return text

    def remove_html(self, text):
        return re.sub(r"<[^>]+>", " ", text)

    def remove_extra_whitespace(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def remove_boilerplate(self, text):
        noise = [
            r"cookie policy.*?accept",
            r"subscribe to our newsletter",
            r"click here to read more",
            r"advertisement",
            r"follow us on social media",
        ]
        for pattern in noise:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text