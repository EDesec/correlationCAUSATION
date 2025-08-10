#!/usr/bin/env python3
"""
Build a weakly-labeled training set by joining news articles with market prices.

- Loads Guardian news (parquet) and SPY adjusted close prices (parquet)
- Applies weak labels (causal / correlational) to composed article text
- Aligns each article timestamp to the most recent SPY close at/preceding publish time
- Computes forward returns at configurable trading-day horizons
- Writes a tidy parquet suitable for model training

Example:
    python build_dataset.py \
        --news data/raw/news_guardian.parquet \
        --prices data/raw/prices_SPY.parquet \
        --out data/processed/train.parquet \
        --horizons 1 3 5
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.append("/Users/eamondwight/Documents/correlationCAUSATION/src")

from nlp.weak_labels import weak_label


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_NEWS = Path("data/raw/news_guardian.parquet")
DEFAULT_PRICES = Path("data/raw/prices_SPY.parquet")
DEFAULT_OUT = Path("data/processed/train.parquet")
DEFAULT_HORIZONS = (3,)  # trading-day steps to compute forward price (and delta)

NEWS_REQUIRED_COLS = {"published_at"}
NEWS_TEXT_PARTS = ("title", "description", "body")
PRICE_ACCEPTED_DATE_COLS = ("date", "Date")
PRICE_ACCEPTED_ADJ_COLS = (("adj close", "adj_close"), ("Adj Close", "adj_close"))
PRICE_FALLBACK_CLOSE_COLS = ("close", "Close")


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logger = logging.getLogger("build_dataset")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    news: Path
    prices: Path
    out: Path


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} missing required columns: {sorted(missing)}")


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns ['ts', 'adj_close'] sorted by ts ascending.

    - Accepts 'date' or 'Date'
    - Normalizes adjusted close to 'adj_close', falling back to 'close' if needed
    - Drops rows with invalid timestamps or prices
    """
    # Resolve timestamp column
    for c in PRICE_ACCEPTED_DATE_COLS:
        if c in df.columns:
            date_col = c
            break
    else:
        raise ValueError("Prices must include a 'date' or 'Date' column.")

    # Normalize adjusted close column name if present
    df = df.copy()
    for (src, dst) in PRICE_ACCEPTED_ADJ_COLS:
        if src in df.columns:
            df = df.rename(columns={src: dst})
    if "adj_close" not in df.columns:
        # Fallback to 'close' / 'Close'
        for c in PRICE_FALLBACK_CLOSE_COLS:
            if c in df.columns:
                df["adj_close"] = df[c]
                break
        else:
            raise ValueError(
                "Prices must include 'adj close'/'Adj Close' or a 'close'/'Close' column."
            )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (
        df.dropna(subset=[date_col, "adj_close"])
          .sort_values(date_col)
          .reset_index(drop=True)
    )
    df["ts"] = df[date_col]  # naive datetime64[ns] assumed to represent session end
    return df[["ts", "adj_close"]]


def _compose_text(row: pd.Series) -> str:
    parts = [str(row.get(col) or "") for col in NEWS_TEXT_PARTS]
    return " ".join(parts).strip()


def _prepare_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare news:
      - parse published_at as UTC
      - create naive 'ts' (UTC) for alignment
      - compose text and weak-label it
      - keep only labeled rows, sorted by ts
    """
    _ensure_columns(df, NEWS_REQUIRED_COLS, "news")
    df = df.copy()

    # Published at → tz-aware UTC, then make naive UTC for join
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"]).reset_index(drop=True)
    df["ts"] = df["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Compose text & label
    df["text"] = df.apply(_compose_text, axis=1)
    df["label"] = df["text"].map(weak_label)

    df = df.dropna(subset=["label"]).sort_values("ts").reset_index(drop=True)
    return df


def _align_prices_at_horizons(
    prices: pd.DataFrame,
    event_ts: np.ndarray,
    horizons: Iterable[int]
    ) -> dict[int, np.ndarray]:
    """
    For each horizon h, return price at index_of(t0) + h where t0 is the last price
    at/preceding event_ts. Missing/insufficient lookahead yields NaN.
    """
    spy_ts = prices["ts"].to_numpy()
    spy_px = prices["adj_close"].to_numpy()

    # index of last known price at or before each event time
    idx0 = np.searchsorted(spy_ts, event_ts, side="right") - 1
    valid0 = idx0 >= 0

    aligned = {}
    for h in horizons:
        idx_h = idx0 + h
        valid_h = valid0 & (idx_h < len(spy_px))
        out = np.full(len(event_ts), np.nan, dtype=float)
        out[valid_h] = spy_px[idx_h[valid_h]]
        aligned[h] = out
    return aligned


def _to_naive_numpy_ts(series: pd.Series) -> np.ndarray:
    # Ensure dtype is compatible for np.searchsorted over datetime64[ns]
    return series.to_numpy(dtype="datetime64[ns]")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

def build_dataset(news_path: Path, prices_path: Path, out_path: Path, horizons: Iterable[int]) -> pd.DataFrame:
    """Run the full pipeline and write the parquet."""
    logger.info("Loading news from %s", news_path)
    news_raw = pd.read_parquet(news_path)
    news = _prepare_news(news_raw)

    logger.info("Loading prices from %s", prices_path)
    prices_raw = pd.read_parquet(prices_path)
    prices = _normalize_price_frame(prices_raw)

    if news.empty:
        raise RuntimeError("No labeled news rows found after preprocessing.")
    if prices.empty:
        raise RuntimeError("No price rows found after preprocessing.")

    logger.info("Aligning %d articles to %d price points at horizons %s",
                len(news), len(prices), list(horizons))

    event_ts = _to_naive_numpy_ts(news["ts"])
    aligned = _align_prices_at_horizons(prices, event_ts, horizons=[0, *horizons])

    spy_t0 = aligned[0]
    data = {
        "text": news["text"].astype(str),
        "label": news["label"].astype(str),
        "published_at": news["published_at"],
        "url": news.get("url"),
        "section": news.get("section"),
        "spy_t0": spy_t0,
    }

    # Add horizon prices and deltas
    for h in horizons:
        data[f"spy_t{h}"] = aligned[h]
        data[f"delta_{h}d"] = aligned[h] - spy_t0

    out_df = pd.DataFrame(data)
    out_df = out_df.dropna(subset=["label"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s", len(out_df), out_path)

    return out_df


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build weakly-labeled news/price dataset.")
    p.add_argument("--news", type=Path, default=DEFAULT_NEWS, help="Path to news parquet.")
    p.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="Path to prices parquet.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output parquet path.")
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Trading-day horizons for forward prices (e.g., 1 3 5).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    paths = Paths(news=args.news, prices=args.prices, out=args.out)
    build_dataset(paths.news, paths.prices, paths.out, horizons=args.horizons)


if __name__ == "__main__":
    main()

