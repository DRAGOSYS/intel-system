import requests
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner
import config

class NewsConnector:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.api_key = config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, query, max_articles=20):
        try:
            params = {
                "q":        query,
                "sortBy":   "publishedAt",
                "language": "en",
                "apiKey":   self.api_key,
                "pageSize": max_articles
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message')}")
                return []

            articles = []
            for item in data.get("articles", []):
                content = self.cleaner.clean(item.get("content") or item.get("description") or "")
                if len(content) < 50:
                    continue
                articles.append({
                    "title":   item.get("title", ""),
                    "content": content,
                    "url":     item.get("url", ""),
                    "source":  item.get("source", {}).get("name", ""),
                })
            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []