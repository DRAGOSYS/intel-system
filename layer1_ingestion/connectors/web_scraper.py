import requests
from bs4 import BeautifulSoup
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner
import time
import random

class WebScraper:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def scrape(self, url):
        try:
            time.sleep(random.uniform(1, 2))
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            content = self.cleaner.clean(soup.get_text(separator="\n"))
            title = soup.title.string if soup.title else ""

            if len(content) < 100:
                logger.warning(f"Very little content scraped from {url}")
                return None

            logger.info(f"Scraped {len(content)} chars from {url}")
            return {
                "title":   title,
                "content": content,
                "url":     url,
                "source":  "web"
            }
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return None