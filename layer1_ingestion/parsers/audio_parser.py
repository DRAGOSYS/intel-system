import os
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner

class AudioParser:

    def __init__(self):
        self.cleaner = TextCleaner()

    def parse(self, filepath):
        try:
            from faster_whisper import WhisperModel

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None

            # tiny model = fastest, lowest RAM usage
            model    = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, info = model.transcribe(filepath)

            text = " ".join([segment.text for segment in segments])
            text = self.cleaner.clean(text)

            logger.info(f"Transcribed audio: {filepath} language: {info.language}")
            return {
                "title":    os.path.basename(filepath),
                "content":  text,
                "source":   "audio",
                "language": info.language,
                "path":     filepath
            }

        except Exception as e:
            logger.error(f"Audio parsing failed: {e}")
            return None