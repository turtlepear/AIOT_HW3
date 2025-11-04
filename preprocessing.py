import argparse
import os
import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def _remove_non_letters(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_stopwords(text: str, stopwords: Iterable[str]) -> str:
    tokens = text.split()
    kept = [t for t in tokens if t not in stopwords]
    return " ".join(kept)


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_non_letters(text)
    text = _normalize_whitespace(text)
    if remove_stopwords:
        text = _remove_stopwords(text, ENGLISH_STOP_WORDS)
    return text


def encode_label(label: str) -> int:
    # Maps typical SMS spam dataset labels. Defaults to 0 for non-spam
    value = str(label).strip().lower()
    if value in {"spam", "1", "true", "yes"}:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SMS spam dataset")
    parser.add_argument("--input_csv", required=True, help="Path to raw CSV (can be no-header or with header)")
    parser.add_argument("--output_csv", required=True, help="Path to write cleaned CSV")
    parser.add_argument("--text_column", default="text", help="Text column name (if input has header)")
    parser.add_argument("--label_column", default="label", help="Label column name (if input has header)")
    parser.add_argument("--no_header", action="store_true", help="Set if the input CSV has no header row")
    args = parser.parse_args()

    if args.no_header:
        df = pd.read_csv(args.input_csv, header=None, names=[args.label_column, args.text_column])
    else:
        df = pd.read_csv(args.input_csv)

    if args.text_column not in df.columns or args.label_column not in df.columns:
        raise SystemExit(f"Expected columns '{args.label_column}' and '{args.text_column}' not found in input CSV")

    df[args.text_column] = df[args.text_column].apply(clean_text)
    df[args.label_column] = df[args.label_column].apply(encode_label)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote cleaned dataset to: {args.output_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()


