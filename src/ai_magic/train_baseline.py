#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from pathlib import Path

DATA = Path("data/processed/train.parquet")

def main():
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=["text","label"])
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # time-aware split is better; start simple with random split to sanity-check
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=100_000, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1))
    ])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    print(classification_report(yte, yhat, digits=3))

if __name__ == "__main__":
    main()
