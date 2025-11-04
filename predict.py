import argparse
import os

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam probability for input text")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.joblib and vectorizer.joblib")
    parser.add_argument("--text", required=True, help="Text to classify")
    parser.add_argument("--threshold", type=float, default=0.35, help="Decision threshold for spam (default 0.35 to favor recall)")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.joblib")
    vec_path = os.path.join(args.model_dir, "vectorizer.joblib")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise SystemExit("Missing model artifacts. Expected 'model.joblib' and 'vectorizer.joblib'.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    X = vectorizer.transform([args.text])
    prob_spam = float(model.predict_proba(X)[0, 1])
    pred = int(prob_spam >= args.threshold)

    print(f"spam_probability={prob_spam:.6f}")
    print(f"prediction={pred}")
    print(f"threshold={args.threshold}")


if __name__ == "__main__":
    main()


