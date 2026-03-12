import os
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner

class PDFParser:

    def __init__(self):
        self.cleaner = TextCleaner()

    def parse(self, filepath):
        try:
            import pypdf
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None

            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            text = self.cleaner.clean(text)

            if len(text) < 50:
                logger.warning(f"Very little text extracted from {filepath}")
                return None

            logger.info(f"Parsed PDF: {filepath} ({len(reader.pages)} pages)")
            return {
                "title":   os.path.basename(filepath),
                "content": text,
                "source":  "pdf",
                "pages":   len(reader.pages),
                "path":    filepath
            }

        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return None