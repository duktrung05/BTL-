"""Predict a medical abstract category with the saved Naive Bayes bundle."""

import argparse
import json
import pickle
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = BACKEND_DIR / "artifacts" / "naive_bayes_model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="English medical abstract or query")
    parser.add_argument("--file", type=Path, help="Read text from a UTF-8 file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if bool(args.text) == bool(args.file):
        parser.error("provide exactly one of text or --file")
    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    with args.model.open("rb") as handle:
        bundle = pickle.load(handle)
    vector = bundle["vectorizer"].transform([text])
    category = bundle["classifier"].predict(vector)[0]
    probabilities = bundle["classifier"].predict_proba(vector)[0]
    print(
        json.dumps(
            {
                "category": category,
                "probabilities": dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True)),
                "model": bundle["metadata"]["best_configuration"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

