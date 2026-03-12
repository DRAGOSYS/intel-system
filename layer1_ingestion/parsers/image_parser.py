import os
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner

class ImageParser:

    def __init__(self):
        self.cleaner = TextCleaner()

    def parse(self, filepath):
        try:
            from PIL import Image
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None

            img  = Image.open(filepath)
            text = pytesseract.image_to_string(img)
            text = self.cleaner.clean(text)

            logger.info(f"Parsed image: {filepath}")
            return {
                "title":   os.path.basename(filepath),
                "content": text,
                "source":  "image",
                "path":    filepath
            }

        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            return None