import os
import subprocess
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner
from layer1_ingestion.parsers.audio_parser import AudioParser

class VideoParser:

    def __init__(self):
        self.cleaner      = TextCleaner()
        self.audio_parser = AudioParser()

    def parse(self, filepath):
        try:
            import cv2

            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None

            # Step 1: Extract audio from video
            audio_path = filepath.replace(".mp4", ".wav")
            subprocess.run([
                "ffmpeg", "-i", filepath,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path, "-y"
            ], capture_output=True)

            # Step 2: Transcribe audio
            transcript = ""
            if os.path.exists(audio_path):
                audio_result = self.audio_parser.parse(audio_path)
                if audio_result:
                    transcript = audio_result["content"]
                os.remove(audio_path)

            # Step 3: Extract key frames
            cap    = cv2.VideoCapture(filepath)
            fps    = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            count  = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # 1 frame every 5 seconds
                if count % (int(fps) * 5) == 0:
                    frame_path = f"data/uploads/frame_{count}.jpg"
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)
                count += 1
            cap.release()

            duration = count / fps if fps > 0 else 0
            logger.info(f"Parsed video: {filepath} duration: {duration:.1f}s frames: {len(frames)}")

            return {
                "title":      os.path.basename(filepath),
                "content":    transcript,
                "source":     "video",
                "frames":     frames,
                "duration":   duration,
                "path":       filepath
            }

        except Exception as e:
            logger.error(f"Video parsing failed: {e}")
            return None