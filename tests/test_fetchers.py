# tests/test_fetchers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pytest
import src.pipelines

# --- Imports from your modules ---
from src.pipelines.fetch_prices import (
    normalize_download,
    merge_incremental as merge_incremental_prices,
    save_ticker_parquet,
    FetchConfig as PriceConfig,
    fetch_one as fetch_one_price,
)
from src.pipelines.fetch_news import (
    normalize_rows,
    merge_incremental as merge_incremental_news,
    save_parquet as save_news_parquet,
    fetch_guardian,
    FetchConfig as NewsConfig,
)

# -------------------------------
# Helpers
# -------------------------------
def sample_price_df() -> pd.DataFrame:
    """Create a small yfinance-like DataFrame (datetime index + standard columns)."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.1, 2.1, 3.1],
            "Adj Close": [1.05, 2.05, 3.05],
            "Volume": [10, 20, 30],
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def sample_guardian_items() -> List[Dict[str, Any]]:
    """Minimal Guardian API 'results' payload."""
    return [
        {
            "id": "env/2024/jan/01/example-1",
            "webUrl": "https://www.theguardian.com/example-1",
            "sectionName": "Business",
            "webPublicationDate": "2024-01-01T12:00:00Z",
            "fields": {
                "headline": "Oil prices rise on supply cuts",
                "trailText": "Short summary.",
                "bodyText": "Full article text here.",
            },
        },
        {
            "id": "env/2024/jan/02/example-2",
            "webUrl": "https://www.theguardian.com/example-2",
            "sectionName": "Energy",
            "webPublicationDate": "2024-01-02T09:30:00Z",
            "fields": {
                "headline": "Natural gas demand surges",
                "trailText": "Another summary.",
                "bodyText": "More full text.",
            },
        },
    ]


# -------------------------------
# Prices tests
# -------------------------------
def test_prices_normalize_download_schema():
    raw = sample_price_df()
    out = normalize_download(raw, ticker="SPY")

    assert list(out.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ticker",
    ]
    assert out["ticker"].nunique() == 1
    assert out["ticker"].iloc[0] == "SPY"
    # dates are just calendar dates
    assert pd.api.types.is_object_dtype(out["date"]) or pd.api.types.is_datetime64_any_dtype(out["date"])
    assert len(out) == 3


def test_prices_merge_incremental_dedup(tmp_path: Path):
    # initial write
    df1 = normalize_download(sample_price_df(), ticker="SPY")
    path = tmp_path / "prices_SPY.parquet"
    merged1 = merge_incremental_prices(path, df1)
    merged1.to_parquet(path, index=False)

    # append overlapping data (duplicate date), should dedupe by (date, ticker)
    df2 = df1.copy()
    df2.loc[0, "close"] = df2.loc[0, "close"] + 1.0  # tweak first row to ensure "keep last"
    merged2 = merge_incremental_prices(path, df2)
    merged2.to_parquet(path, index=False)

    final = pd.read_parquet(path)
    # still 3 rows, and first row reflects the "latest" value
    assert len(final) == 3
    # since we sorted by date ascending, check min date row has updated close
    min_date = final["date"].min()
    updated_close = final.loc[final["date"] == min_date, "close"].iloc[0]
    assert updated_close == pytest.approx(df2.loc[0, "close"])


def test_fetch_one_price_monkeypatched(monkeypatch, tmp_path: Path):
    """Mock yfinance.download to avoid network and ensure fetch_one returns normalized data."""
    import src.pipelines.fetch_prices as fp

    def fake_download(ticker, period, interval, progress, auto_adjust, group_by, threads):
        return sample_price_df()

    monkeypatch.setattr(fp.yf, "download", fake_download)

    cfg = PriceConfig(
        tickers=["SPY"],
        period="5y",
        interval="1d",
        out_dir=tmp_path,
    )
    df = fetch_one_price(cfg, "SPY")
    assert not df.empty
    assert set(df.columns) == {"date", "open", "high", "low", "close", "adj_close", "volume", "ticker"}


# -------------------------------
# News tests
# -------------------------------
def test_news_normalize_rows_schema():
    items = sample_guardian_items()
    df = normalize_rows(items)

    assert list(df.columns) == ["id", "url", "source", "section", "published_at", "title", "description", "body"]
    assert len(df) == 2
    assert (df["source"] == "guardian").all()
    # timestamp parseable
    ts = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    assert ts.notna().all()


def test_news_merge_incremental_dedup(tmp_path: Path):
    items = sample_guardian_items()
    df1 = normalize_rows(items)

    path = tmp_path / "news_guardian.parquet"
    merged1 = merge_incremental_news(path, df1)
    merged1.to_parquet(path, index=False)

    # duplicate URL with newer published_at should win
    df2 = df1.copy()
    df2.loc[0, "published_at"] = "2024-01-03T00:00:00Z"  # newer
    merged2 = merge_incremental_news(path, df2)
    merged2.to_parquet(path, index=False)

    final = pd.read_parquet(path)
    assert len(final) == 2
    # ensure URL dedup kept the newer row
    newer_ts = pd.to_datetime(final.loc[final["url"] == df2.loc[0, "url"], "published_at"], utc=True).iloc[0]
    assert newer_ts == pd.Timestamp("2024-01-03T00:00:00Z")


def test_fetch_guardian_monkeypatched(monkeypatch, tmp_path: Path):
    """Mock fetch_page to test pagination without hitting the real API."""
    import src.pipelines.fetch_news as fn

    # two pages: second page returns empty to simulate end-of-results
    page_payloads = {
        1: sample_guardian_items(),
        2: [],
    }

    def fake_fetch_page(cfg, page: int):
        return page_payloads.get(page, [])

    monkeypatch.setattr(fn, "fetch_page", fake_fetch_page)

    cfg = NewsConfig(
        api_key="DUMMY",
        query="energy",
        pages=2,
        page_size=100,
        since="2024-01-01",
        until="2024-01-31",
        out_path=tmp_path / "news_guardian.parquet",
    )
    df = fetch_guardian(cfg)
    assert len(df) == 2
    assert set(df.columns) == {"id", "url", "source", "section", "published_at", "title", "description", "body"}

    # Test idempotent save
    out_path = cfg.out_path
    save_news_parquet(out_path, df)
    # Write same again — should dedupe
    save_news_parquet(out_path, df)
    final = pd.read_parquet(out_path)
    assert len(final) == 2
