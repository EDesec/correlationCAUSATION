#!/usr/bin/env python3
#!/usr/bin/env python3
"""
pipelines/fetch_news.py

Fetch news articles from The Guardian API and save to Parquet.
- Stable schema: id, url, source, section, published_at, title, description, body
- Idempotent: merges with existing Parquet and dedupes by URL
- Resilient: retries + rate-limit backoff
- CLI flags for query, date window, pages, page size, and output path

Usage:
  export GUARDIAN_API_KEY=your_api_key
  python pipelines/fetch_news.py \
      --query "energy OR oil OR fed OR inflation" \
      --since "2024-01-01" \
      --until "2025-08-09" \
      --pages 5 \
      --page-size 100 \
      --out data/raw/news_guardian.parquet

Notes:
- Guardian fields used: headline, trailText, bodyText (plain text).
- Timestamps are stored in UTC ISO 8601.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from dateutil import parser as dtp


# -----------------------
# Logging configuration
# -----------------------
def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# -----------------------
# Helpers & normalization
# -----------------------
API_URL = "https://content.guardianapis.com/search"


def iso_or_none(value: Optional[str]) -> Optional[str]:
    """Best-effort parse to UTC ISO string (e.g., '2024-01-01T12:00:00+00:00')."""
    if not value:
        return None
    try:
        dt = dtp.parse(value)
        if not dt.tzinfo:
            # treat naive as UTC
            dt = dt.tz_localize("UTC") if hasattr(dt, "tz_localize") else dt
        return dt.astimezone(tz=pd.Timestamp.utcnow().tz).isoformat()
    except Exception:
        return None


def normalize_rows(items: List[Dict]) -> pd.DataFrame:
    """Map Guardian API 'results' array into our stable schema."""
    rows: List[Dict] = []
    for a in items:
        f = a.get("fields", {}) or {}
        rows.append(
            {
                "id": a.get("id"),
                "url": a.get("webUrl"),
                "source": "guardian",
                "section": a.get("sectionName"),
                "published_at": iso_or_none(a.get("webPublicationDate")),
                "title": f.get("headline"),
                "description": f.get("trailText"),
                "body": f.get("bodyText"),
            }
        )
    df = pd.DataFrame(rows)
    # Ensure consistent column order even if empty
    cols = ["id", "url", "source", "section", "published_at", "title", "description", "body"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def merge_incremental(path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge with existing Parquet by URL and keep newest by published_at."""
    if path.exists():
        try:
            df_old = pd.read_parquet(path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            logging.warning("Failed reading %s (%s). Overwriting with new data.", path, e)
            df_all = df_new.copy()
    else:
        df_all = df_new.copy()

    # Sort so drop_duplicates keeps the latest version of any URL
    df_all["published_at"] = pd.to_datetime(df_all["published_at"], errors="coerce", utc=True)
    df_all = (
        df_all.sort_values(["url", "published_at"], na_position="last")
              .drop_duplicates(subset=["url"], keep="last")
              .reset_index(drop=True)
    )
    return df_all


# -----------------------
# Core fetch logic
# -----------------------
@dataclass
class FetchConfig:
    api_key: str
    query: str
    pages: int
    page_size: int
    since: Optional[str]  # ISO date (YYYY-MM-DD) or None
    until: Optional[str]  # ISO date (YYYY-MM-DD) or None
    out_path: Path
    retries: int = 3
    backoff: float = 1.25  # seconds multiplier between retries
    sleep_between_pages: float = 0.2


def fetch_page(cfg: FetchConfig, page: int) -> List[Dict]:
    """Fetch a single page from the Guardian API with basic retry/backoff."""
    params = {
        "q": cfg.query,
        "page": page,
        "page-size": cfg.page_size,
        "order-by": "newest",
        "show-fields": "headline,trailText,bodyText",
        "api-key": cfg.api_key,
    }
    if cfg.since:
        params["from-date"] = cfg.since  # YYYY-MM-DD
    if cfg.until:
        params["to-date"] = cfg.until

    for attempt in range(cfg.retries):
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code == 429:
                # Rate limited – wait and retry
                wait_s = (attempt + 1) * cfg.backoff
                logging.warning("HTTP 429 rate limited (page %d). Sleeping %.2fs…", page, wait_s)
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            payload = r.json()
            return payload.get("response", {}).get("results", [])
        except Exception as e:
            wait_s = (attempt + 1) * cfg.backoff
            logging.warning("Page %d fetch failed (attempt %d/%d): %s. Sleeping %.2fs…",
                            page, attempt + 1, cfg.retries, e, wait_s)
            time.sleep(wait_s)
    logging.error("Page %d failed after retries.", page)
    return []


def fetch_guardian(cfg: FetchConfig) -> pd.DataFrame:
    """Fetch multiple pages and return a normalized DataFrame (can be empty)."""
    all_items: List[Dict] = []
    for p in range(1, cfg.pages + 1):
        items = fetch_page(cfg, p)
        if not items:
            logging.warning("No items on page %d.", p)
        all_items.extend(items)
        time.sleep(cfg.sleep_between_pages)
    df = normalize_rows(all_items)
    return df


def save_parquet(path: Path, df: pd.DataFrame) -> Path:
    """Idempotent write by merging and deduping."""
    if df.empty:
        logging.warning("No rows fetched; nothing written.")
        return path
    merged = merge_incremental(path, df)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return path


# -----------------------
# CLI / Main
# -----------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch news from The Guardian and write a Parquet file.")
    p.add_argument("--query", default="energy OR oil OR gas OR fed OR inflation",
                   help="Guardian search query string.")
    p.add_argument("--since", default=None, help='Start date (YYYY-MM-DD)')
    p.add_argument("--until", default=None, help='End date (YYYY-MM-DD)')
    p.add_argument("--pages", type=int, default=5, help="Number of pages to fetch")
    p.add_argument("--page-size", type=int, default=100, help="Items per page (max ~200)")
    p.add_argument("--out", default="data/raw/news_guardian.parquet", help="Output Parquet path")
    p.add_argument("--api-key", default=None, help="Guardian API key (or set GUARDIAN_API_KEY env var)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(getattr(logging, args.log_level))

    api_key = args.api_key or os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        logging.error("Missing API key. Provide --api-key or set GUARDIAN_API_KEY env var.")
        return 1

    cfg = FetchConfig(
        api_key=api_key,
        query=args.query,
        pages=args.pages,
        page_size=args.page_size,
        since=args.since,
        until=args.until,
        out_path=Path(args.out),
    )

    logging.info("Fetching Guardian: query=%r, pages=%d, page_size=%d, since=%s, until=%s",
                 cfg.query, cfg.pages, cfg.page_size, cfg.since, cfg.until)

    df = fetch_guardian(cfg)
    path = save_parquet(cfg.out_path, df)
    logging.info("Done. %d rows → %s", len(df), path)
    return 0 if not df.empty else 1


if __name__ == "__main__":
    sys.exit(main())


