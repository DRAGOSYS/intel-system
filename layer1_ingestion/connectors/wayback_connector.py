import requests
from loguru import logger
from layer1_ingestion.cleaners.text_cleaner import TextCleaner
from layer1_ingestion.connectors.web_scraper import WebScraper
import time

class WaybackConnector:

    def __init__(self):
        self.cleaner = TextCleaner()
        self.scraper = WebScraper()
        self.api_url = "http://archive.org/wayback/available"

    def fetch(self, url, year=None):
        try:
            params = {"url": url}
            if year:
                params["timestamp"] = f"{year}0101"

            # Retry up to 3 times
            for attempt in range(3):
                try:
                    response = requests.get(self.api_url, params=params, timeout=15)
                    if response.text.strip() == "":
                        logger.warning(f"Empty response, retrying {attempt+1}/3")
                        time.sleep(2)
                        continue
                    data = response.json()
                    break
                except Exception:
                    time.sleep(2)
                    continue
            else:
                logger.error("All retries failed")
                return None

            snapshots = data.get("archived_snapshots", {})
            if not snapshots:
                logger.warning(f"No archive found for {url}")
                return None

            archive_url = snapshots["closest"]["url"]
            archive_year = snapshots["closest"]["timestamp"][:4]
            logger.info(f"Found archive: {archive_url}")

            result = self.scraper.scrape(archive_url)
            if result:
                result["source"] = "wayback_machine"
                result["original_url"] = url
                result["archive_year"] = archive_year
            return result

        except Exception as e:
            logger.error(f"Wayback fetch failed for {url}: {e}")
            return None