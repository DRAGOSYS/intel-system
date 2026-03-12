from dotenv import load_dotenv
load_dotenv()
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("logs/system.log", rotation="10 MB", level="DEBUG")

def main():
    logger.info("Intel System Starting...")

    import config
    logger.info(f"Model: {config.LOCAL_MODEL}")
    logger.info(f"Qdrant: {config.QDRANT_URL}")

    from layer1_ingestion.scheduler import IngestionScheduler
    scheduler = IngestionScheduler()
    scheduler.fetch_rss()
    scheduler.fetch_news()
    logger.info(f"Total documents collected: {len(scheduler.all_data)}")
    logger.info("Layer 1 running successfully!")

if __name__ == "__main__":
    main()