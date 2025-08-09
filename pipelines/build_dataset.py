#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from nlp.weak_labels import weak_label

RAW = Path("data/raw")
OUT = Path("data/processed")

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={"adj close":"adj_close","Adj Close":"adj_close"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close", df.get("Close"))
    df = df.dropna(subset=[date_col, "adj_close"]).sort_values(date_col).reset_index(drop=True)
    df["ts"] = df[date_col]  # naive datetime64[ns]
    return df[["ts","adj_close"]]

def run():
    news = pd.read_parquet(RAW / "news_guardian.parquet").copy()
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce", utc=True)
    news = news.dropna(subset=["published_at"]).reset_index(drop=True)
    # make naive timestamps for easy alignment with market data
    news["ts"] = news["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    # compose text and weak-label it
    news["text"] = (news["title"].fillna("") + " " +
                    news["description"].fillna("") + " " +
                    news["body"].fillna(""))
    news["label"] = news["text"].map(weak_label)
    news = news.dropna(subset=["label"]).sort_values("ts").reset_index(drop=True)

    spy = load_prices(str(RAW / "prices_SPY.parquet")).sort_values("ts").reset_index(drop=True)

    # vectorized alignment: last known SPY close at/<= article time
    spy_ts = spy["ts"].to_numpy()
    spy_px = spy["adj_close"].to_numpy()
    art_ts = news["ts"].to_numpy()

    idx0 = np.searchsorted(spy_ts, art_ts, side="right") - 1  # price at/just before article
    valid0 = idx0 >= 0
    adj0 = np.full(len(news), np.nan)
    adj0[valid0] = spy_px[idx0[valid0]]

    idx3 = idx0 + 3  # +3 trading days
    valid3 = valid0 & (idx3 < len(spy_px))
    adj3 = np.full(len(news), np.nan)
    adj3[valid3] = spy_px[idx3[valid3]]

    delta3 = adj3 - adj0

    out = pd.DataFrame({
        "text": news["text"],
        "label": news["label"],
        "published_at": news["published_at"],
        "url": news["url"],
        "section": news["section"],
        "spy_t0": adj0,
        "spy_t3": adj3,
        "delta_3d": delta3,
    }).dropna(subset=["label"]).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "train.parquet", index=False)
    print(f"wrote {len(out)} rows → {OUT / 'train.parquet'}")

if __name__ == "__main__":
    run()
