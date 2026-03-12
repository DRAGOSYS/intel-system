from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from layer1_ingestion.connectors.rss_connector import RSSConnector
from layer1_ingestion.connectors.news_connector import NewsConnector
from layer1_ingestion.cleaners.deduplicator import Deduplicator

class IngestionScheduler:

    def __init__(self):
        self.scheduler    = BackgroundScheduler()
        self.rss          = RSSConnector()
        self.news         = NewsConnector()
        self.deduplicator = Deduplicator()
        self.all_data     = []

    def fetch_rss(self):
        logger.info("Running RSS fetch...")
        articles = self.rss.fetch_all()
        articles = self.deduplicator.filter_duplicates(articles)
        self.all_data.extend(articles)
        logger.info(f"Total documents collected: {len(self.all_data)}")

    def fetch_news(self):
        logger.info("Running News fetch...")
        articles = self.news.fetch("latest news", max_articles=20)
        articles = self.deduplicator.filter_duplicates(articles)
        self.all_data.extend(articles)
        logger.info(f"Total documents collected: {len(self.all_data)}")

    def setup_jobs(self):
        # RSS every 5 minutes
        self.scheduler.add_job(
            func=self.fetch_rss,
            trigger=IntervalTrigger(minutes=5),
            id="rss_job",
            name="RSS Fetcher"
        )
        # News every 30 minutes
        self.scheduler.add_job(
            func=self.fetch_news,
            trigger=IntervalTrigger(minutes=30),
            id="news_job",
            name="News Fetcher"
        )

    def start(self):
        self.setup_jobs()
        self.scheduler.start()
        logger.info("Scheduler started successfully")
        logger.info("RSS runs every 5 minutes")
        logger.info("News runs every 30 minutes")

    def stop(self):
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")