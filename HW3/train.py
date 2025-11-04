import argparse
import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def load_dataset(path: str, text_column: str, label_column: str) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if text_column not in df.columns or label_column not in df.columns:
        raise SystemExit(f"Expected columns '{label_column}' and '{text_column}' not found in cleaned CSV")
    X = df[text_column].astype(str)
    y = df[label_column].astype(int)
    return X, y


def train_and_evaluate(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    max_features: int = 20000,
    random_state: int = 42,
    class_weight: str | None = None,
    C: float = 1.0,
):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight=class_weight, C=C)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, vectorizer, metrics, y_prob.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression spam classifier")
    parser.add_argument("--clean_csv", required=True, help="Path to cleaned dataset CSV")
    parser.add_argument("--model_dir", required=True, help="Directory to save model artifacts")
    parser.add_argument("--text_column", default="text", help="Text column name")
    parser.add_argument("--label_column", default="label", help="Label column name (0/1)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size (0-1)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--max_features", type=int, default=20000, help="Max vocabulary size for TF-IDF")
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="balanced", help="Class weight strategy")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for LogisticRegression")
    args = parser.parse_args()

    X, y = load_dataset(args.clean_csv, args.text_column, args.label_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    cw = None if args.class_weight == "none" else "balanced"
    model, vectorizer, metrics, y_prob = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        max_features=args.max_features,
        random_state=args.random_state,
        class_weight=cw,
        C=args.C,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved artifacts:")
    print(f"  model:      {os.path.join(args.model_dir, 'model.joblib')}")
    print(f"  vectorizer: {os.path.join(args.model_dir, 'vectorizer.joblib')}")
    print(f"  metrics:    {os.path.join(args.model_dir, 'metrics.json')}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


