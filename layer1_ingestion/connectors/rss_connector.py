import feedparser
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner

class RSSConnector:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.feeds = [
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
            "https://feeds.reuters.com/reuters/topNews",
        ]

    def fetch(self, url):
        try:
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries:
                content = entry.get("summary", "")
                content = self.cleaner.clean(content)
                if len(content) < 50:
                    continue
                articles.append({
                    "title":   entry.get("title", ""),
                    "content": content,
                    "url":     entry.get("link", ""),
                    "source":  feed.feed.get("title", url),
                })
            logger.info(f"Fetched {len(articles)} articles from {url}")
            return articles
        except Exception as e:
            logger.error(f"RSS fetch failed for {url}: {e}")
            return []

    def fetch_all(self):
        all_articles = []
        for url in self.feeds:
            all_articles.extend(self.fetch(url))
        return all_articles