#!/usr/bin/env python3
"""
pipelines/fetch_prices.py

Download OHLCV price data with yfinance, normalize columns, and write Parquet files.
Designed to be idempotent (safe to re-run), robust to yfinance quirks, and easy to extend.

Usage:
  python pipelines/fetch_prices.py \
      --tickers SPY,CL=F,^VIX \
      --period 5y \
      --interval 1d \
      --out-dir data/raw

Notes:
- yfinance supports: equities (AAPL), ETFs (SPY), futures (CL=F), indices (^VIX).
- We keep both 'close' and 'adj_close' (unadjusted vs adjusted).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf


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
def sanitize_filename(ticker: str) -> str:
    """Make a safe filename segment from a ticker (e.g., '^VIX' -> 'IDX_VIX', 'CL=F' -> 'CL_F')."""
    return ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_").strip()


def flatten_columns(df: pd.DataFrame) -> List[str]:
    """
    yfinance can return MultiIndex columns. Turn everything into flat, lowercase strings.
    Examples:
      ('Adj Close', '') -> 'adj close'
      'Open' -> 'open'
    """
    flat: List[str] = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = "_".join(str(x) for x in c if x)  # drop empty tuple parts
        flat.append(str(c).lower())
    return flat


def normalize_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize yfinance output to:
      date (datetime64[ns], UTC-naive midnight), open, high, low, close, adj_close, volume, ticker
    Only keep rows with a valid date AND numeric adj_close.
    """
    target_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]

    if df is None or df.empty:
        return pd.DataFrame(columns=target_cols)

    df = df.reset_index()
    df.columns = flatten_columns(df)

    rename_map = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
        "datetime": "date",
        "index": "date",   # just in case
    }
    df = df.rename(columns=rename_map)

    # Ensure a date column
    if "date" not in df.columns:
        logging.warning("No 'date' column present after normalization.")
        return pd.DataFrame(columns=target_cols)

    # Parse date as pandas Timestamp (keep full datetime), then normalize to midnight and make naive
    dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    dt = dt.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df["date"] = dt

    # Ensure expected price/volume columns exist
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce numeric columns
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = (
            df[col]
            .replace({"None": None, "none": None, "null": None, "": None})
            .astype(str).str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ticker"] = ticker

    # Keep only rows with valid date and adj_close (drop the junk!)
    df = df.dropna(subset=["date", "adj_close"])

    # Final ordering/sorting
    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df



def merge_incremental(path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    If a Parquet file exists at 'path', merge and dedupe with df_new on (date, ticker).
    Always return a sorted DataFrame.
    """
    if path.exists():
        try:
            df_old = pd.read_parquet(path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            logging.warning("Failed reading existing parquet %s (%s). Overwriting with new data.", path, e)
            df_all = df_new.copy()
    else:
        df_all = df_new.copy()

    df_all = (
        df_all.drop_duplicates(subset=["date", "ticker"], keep="last")
              .sort_values(["ticker", "date"])
              .reset_index(drop=True)
    )
    return df_all


# -----------------------
# Core fetch logic
# -----------------------
@dataclass
class FetchConfig:
    tickers: List[str]
    period: str
    interval: str
    out_dir: Path
    retries: int = 3
    backoff: float = 1.25  # seconds multiplier


def fetch_one(cfg: FetchConfig, ticker: str) -> pd.DataFrame:
    """
    Download a single ticker with a couple of retries, normalize, and return a DataFrame.
    """
    for attempt in range(cfg.retries):
        try:
            # group_by="column" avoids MultiIndex columns in many cases
            raw = yf.download(
                ticker,
                period=cfg.period,
                interval=cfg.interval,
                progress=False,
                auto_adjust=False,
                group_by="column",
                threads=False,
            )
            df = normalize_download(raw, ticker)
            if df.empty:
                logging.warning("%s: no data returned (attempt %d/%d).", ticker, attempt + 1, cfg.retries)
            else:
                return df
        except Exception as e:
            logging.warning("%s: download failed (attempt %d/%d): %s", ticker, attempt + 1, cfg.retries, e)

        # simple backoff before next try
        sleep_s = (attempt + 1) * cfg.backoff
        time.sleep(sleep_s)

    # All retries failed
    logging.error("%s: exhausted retries; returning empty DataFrame.", ticker)
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"])


def save_ticker_parquet(out_dir: Path, df: pd.DataFrame) -> Path:
    """
    Write/merge a single ticker's DataFrame to Parquet (one file per ticker).
    Requires at least one non-null adj_close row.
    """
    if df.empty or df["adj_close"].notna().sum() == 0:
        raise ValueError("Refusing to write: no usable rows (adj_close all null).")

    ticker = str(df["ticker"].iloc[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"prices_{sanitize_filename(ticker)}.parquet"

    merged = merge_incremental(path, df)
    # Drop any accidental nulls in the merged result too
    merged = merged.dropna(subset=["date", "adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Refusing to write: merged result is empty after dropping nulls.")

    merged.to_parquet(path, index=False)
    return path



# -----------------------
# CLI / Main
# -----------------------
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch OHLCV data via yfinance and write Parquet files.")
    p.add_argument("--tickers", default="SPY,CL=F,^VIX", help="Comma-separated list, e.g. 'AAPL,SPY,CL=F,^VIX'")
    p.add_argument("--period", default="5y", choices=["1y", "2y", "5y", "10y", "max"])
    p.add_argument("--interval", default="1d", choices=["1d", "1h", "1wk", "1mo"])
    p.add_argument("--out-dir", default="data/raw", help="Directory for Parquet outputs")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(getattr(logging, args.log_level))

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = FetchConfig(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        out_dir=Path(args.out_dir),
    )

    logging.info("Starting fetch: %d tickers | period=%s | interval=%s | out_dir=%s",
                 len(cfg.tickers), cfg.period, cfg.interval, cfg.out_dir)

    successes = 0
    for t in cfg.tickers:
        try:
            df = fetch_one(cfg, t)
            if df.empty:
                logging.warning("%s: empty after all retries; skipped.", t)
                continue
            path = save_ticker_parquet(cfg.out_dir, df)
            logging.info("%s: %d rows → %s", t, len(df), path)
            successes += 1
        except Exception as e:
            logging.exception("%s: failed to save: %s", t, e)

    logging.info("Done. %d/%d tickers written.", successes, len(cfg.tickers))
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
