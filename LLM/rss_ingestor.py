from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from django.utils.text import slugify

logger = logging.getLogger(__name__)


RSS_SCHEMA = [
    "title",
    "summary",
    "link",
    "published",
    "content",
    "author",
    "categories",
    "source",
    "uid",
]


@dataclass
class RSSIngestResult:
    dataset_path: str
    dataset_name: str
    num_rows: int
    schema: List[str]


class RSSIngestor:
    """Fetches an RSS/Atom feed and saves it as a CSV dataset following RSS_SCHEMA."""

    def __init__(self, base_dataset_dir: str, scrape_content: bool = True, timeout: int = 10):
        self.base_dataset_dir = Path(base_dataset_dir)
        self.base_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.scrape_content = scrape_content
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def ingest(self, feed_url: str, user_id: str, dataset_name: str | None = None) -> RSSIngestResult:
        parsed_feed = feedparser.parse(feed_url)
        if parsed_feed.bozo:
            raise ValueError(f"Unable to parse RSS feed: {parsed_feed.bozo_exception}")
        if not parsed_feed.entries:
            raise ValueError("The provided RSS feed does not contain any entries.")

        dataset_slug = slugify(dataset_name or parsed_feed.feed.get("title") or "rss-feed")
        if not dataset_slug:
            dataset_slug = f"feed-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        user_dir = self.base_dataset_dir / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        csv_path = user_dir / f"{dataset_slug}_{timestamp}.csv"

        df = self._entries_to_dataframe(parsed_feed.entries)
        if df.empty:
            raise ValueError("No valid entries could be extracted from the RSS feed.")

        df.to_csv(csv_path, index=False)

        return RSSIngestResult(
            dataset_path=str(csv_path),
            dataset_name=dataset_slug,
            num_rows=len(df),
            schema=list(RSS_SCHEMA),
        )

    def _entries_to_dataframe(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for entry in entries:
            link = entry.get("link", "").strip()
            # Extract content from RSS feed first
            rss_content = self._extract_content(entry)
            
            # If scraping is enabled and we have a link, try to scrape full content
            if self.scrape_content and link:
                scraped_content = self._scrape_article_content(link)
                # Use scraped content if available, otherwise fall back to RSS content
                content = scraped_content if scraped_content else rss_content
            else:
                content = rss_content
            
            rows.append(
                {
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", "").strip(),
                    "link": link,
                    "published": self._format_published(entry),
                    "content": content,
                    "author": entry.get("author", "").strip(),
                    "categories": self._extract_categories(entry),
                    "source": self._extract_source(entry),
                    "uid": self._extract_uid(entry),
                }
            )

        return pd.DataFrame(rows, columns=RSS_SCHEMA)

    @staticmethod
    def _format_published(entry: Dict[str, Any]) -> str:
        published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if published_parsed:
            return datetime(*published_parsed[:6]).isoformat()
        return entry.get("published") or entry.get("updated") or ""

    @staticmethod
    def _extract_content(entry: Dict[str, Any]) -> str:
        contents = entry.get("content")
        if isinstance(contents, list) and contents:
            return "\n".join([part.get("value", "").strip() for part in contents])
        return entry.get("description", "").strip()

    @staticmethod
    def _extract_categories(entry: Dict[str, Any]) -> str:
        tags = entry.get("tags") or []
        categories = [tag.get("term") for tag in tags if isinstance(tag, dict) and tag.get("term")]
        return ", ".join(categories)

    @staticmethod
    def _extract_source(entry: Dict[str, Any]) -> str:
        source = entry.get("source")
        if isinstance(source, dict):
            return source.get("title") or source.get("href") or ""
        return source or ""

    @staticmethod
    def _extract_uid(entry: Dict[str, Any]) -> str:
        return entry.get("id") or entry.get("guid") or entry.get("link") or ""

    def _scrape_article_content(self, url: str) -> str:
        """
        Scrape the full content from an article URL.
        Returns the scraped content as plain text, or empty string if scraping fails.
        """
        if not url or not url.startswith(('http://', 'https://')):
            return ""
        
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                script.decompose()
            
            # Try to find main content in common article containers
            content_selectors = [
                'article',
                '[role="main"]',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.main-content',
                '#content',
                '#article-body',
                '.article-body',
                '.post-body',
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get text from the first matching element
                    content_text = elements[0].get_text(separator='\n', strip=True)
                    if len(content_text) > 200:  # Only use if substantial content found
                        break
            
            # If no specific content area found, try to get body text
            if not content_text or len(content_text) < 200:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = [line.strip() for line in content_text.split('\n') if line.strip()]
            content_text = '\n'.join(lines)
            
            # Limit content length to avoid extremely long articles
            max_length = 50000
            if len(content_text) > max_length:
                content_text = content_text[:max_length] + "..."
            
            return content_text if content_text else ""
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to scrape content from {url}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Error parsing content from {url}: {e}")
            return ""
        finally:
            # Small delay to be respectful to servers
            time.sleep(0.5)

