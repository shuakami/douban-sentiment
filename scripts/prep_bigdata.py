# -*- coding: utf-8 -*-
"""Clean the CSU Douban dataset comments.csv -> parquet on E:.

Source: <data_root>/瀹屾暣鏁版嵁/comments.csv (gb18030)
Output: <data_root>/clean/comments.parquet
"""
import os, re, sys
import pandas as pd
from _paths import DATA_ROOT

SRC = str(DATA_ROOT / "瀹屾暣鏁版嵁" / "comments.csv")
OUT_DIR = str(DATA_ROOT / "clean")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "comments.pkl")

CHUNK = 200_000
frames = []
total = 0
for chunk in pd.read_csv(SRC, encoding="gb18030", encoding_errors="replace",
                         chunksize=CHUNK,
                         dtype={"COMMENT_ID": "int64", "MOVIE_ID": "int64",
                                "VOTES": "int64", "RATING": "float64"},
                         on_bad_lines="skip"):
    total += len(chunk)
    frames.append(chunk)
    print(f"read {total:,} rows", flush=True)

df = pd.concat(frames, ignore_index=True)
print("raw rows:", len(df))
print("cols:", list(df.columns))
print("rating dist:\n", df["RATING"].value_counts(dropna=False).sort_index())

# clean content
c = df["CONTENT"].astype(str)
c = c.str.replace(r"\s+", " ", regex=True).str.strip()
df["content"] = c
mask = (
    df["RATING"].between(1, 5)
    & df["content"].str.len().between(4, 500)
    & ~df["content"].str.contains(r"^https?://|www\.", regex=True)
)
df = df[mask].copy()
print("after filter:", len(df))

# dedupe by normalized content
df["norm"] = df["content"].str.lower()
df = df.drop_duplicates(subset="norm").drop(columns="norm")
print("after dedupe:", len(df))

# keep useful columns
df = df.rename(columns={
    "COMMENT_ID": "cid", "USER_MD5": "uid", "MOVIE_ID": "mid",
    "CONTENT": "raw", "VOTES": "votes", "COMMENT_TIME": "ctime",
    "RATING": "rating",
})
df = df[["cid", "uid", "mid", "raw", "content", "votes", "ctime", "rating"]]
df["len"] = df["content"].str.len()
df.to_pickle(OUT)
print("saved:", OUT, len(df), "rows")
print("len dist:\n", df["len"].describe())
print("votes dist:\n", df["votes"].describe())
print("rating dist after clean:\n", df["rating"].value_counts().sort_index())
