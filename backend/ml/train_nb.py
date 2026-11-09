"""Run NB baselines, select on validation, and evaluate test once."""

from __future__ import annotations

import csv
import gc
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ml.metrics import classification_metrics
from ml.simple_ml import MultinomialNaiveBayes, SparseTextVectorizer


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data" / "processed"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
REPORT_DIR = BACKEND_DIR / "reports"
ALPHAS = (0.1, 0.5, 1.0)
MIN_DFS = (2, 3, 5)
MAX_FEATURES = 40_000
FAMILIES = (
    {"name": "NB-1", "mode": "count", "ngram_range": (1, 1)},
    {"name": "NB-2", "mode": "tfidf", "ngram_range": (1, 1)},
    {"name": "NB-3", "mode": "tfidf", "ngram_range": (1, 2)},
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_confusion_matrix_svg(path: Path, matrix: list[list[int]], labels: list[str]) -> None:
    short = {
        "Cardiovascular diseases": "Cardiovascular",
        "Digestive system diseases": "Digestive",
        "General pathological conditions": "General",
        "Neoplasms": "Neoplasms",
        "Nervous system diseases": "Nervous",
    }
    size, left, top = 80, 210, 130
    width, height = left + size * len(labels) + 40, top + size * len(labels) + 90
    maximum = max(max(row) for row in matrix) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="20" font-weight="bold">Naive Bayes — Test confusion matrix</text>',
        f'<text x="{left + 130}" y="65" font-size="15">Predicted label</text>',
        '<text x="20" y="330" font-size="15" transform="rotate(-90 20 330)">True label</text>',
    ]
    for column, label in enumerate(labels):
        x = left + column * size + size / 2
        parts.append(f'<text x="{x}" y="{top - 12}" text-anchor="end" font-size="12" transform="rotate(-35 {x} {top - 12})">{short[label]}</text>')
    for row, label in enumerate(labels):
        y = top + row * size + size / 2 + 5
        parts.append(f'<text x="{left - 12}" y="{y}" text-anchor="end" font-size="12">{short[label]}</text>')
        for column, value in enumerate(matrix[row]):
            intensity = value / maximum
            blue = int(245 - 150 * intensity)
            fill = f"rgb({blue},{blue},{255})"
            x = left + column * size
            cell_y = top + row * size
            parts.append(f'<rect x="{x}" y="{cell_y}" width="{size}" height="{size}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{x + size / 2}" y="{cell_y + size / 2 + 6}" text-anchor="middle" font-size="16">{value}</text>')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def discriminative_terms(
    vectorizer: SparseTextVectorizer, classifier: MultinomialNaiveBayes, limit: int = 20
) -> dict[str, list[dict[str, float | str]]]:
    terms = vectorizer.get_feature_names()
    output: dict[str, list[dict[str, float | str]]] = {}
    for target, label in enumerate(classifier.classes_):
        other_targets = [index for index in range(len(classifier.classes_)) if index != target]
        scores = []
        for feature, term in enumerate(terms):
            other_mean = sum(classifier.feature_log_prob_[index][feature] for index in other_targets) / len(other_targets)
            scores.append((classifier.feature_log_prob_[target][feature] - other_mean, term))
        scores.sort(reverse=True)
        output[label] = [{"term": term, "log_odds": round(score, 6)} for score, term in scores[:limit]]
    return output


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(DATA_DIR / "train.jsonl")
    validation_rows = load_jsonl(DATA_DIR / "validation.jsonl")
    test_rows = load_jsonl(DATA_DIR / "test.jsonl")
    train_texts = [row["abstract"] for row in train_rows]
    train_labels = [row["category"] for row in train_rows]
    validation_texts = [row["abstract"] for row in validation_rows]
    validation_labels = [row["category"] for row in validation_rows]
    labels = sorted(set(train_labels))

    experiments: list[dict[str, Any]] = []
    experiment_number = 0
    for family in FAMILIES:
        for min_df in MIN_DFS:
            started = perf_counter()
            vectorizer = SparseTextVectorizer(
                mode=str(family["mode"]),
                ngram_range=family["ngram_range"],
                min_df=min_df,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
            )
            train_vectors = vectorizer.fit_transform(train_texts)
            validation_vectors = vectorizer.transform(validation_texts)
            vectorization_seconds = perf_counter() - started
            for alpha in ALPHAS:
                experiment_number += 1
                model_started = perf_counter()
                classifier = MultinomialNaiveBayes(alpha=alpha).fit(
                    train_vectors, train_labels, len(vectorizer.vocabulary_)
                )
                predictions = classifier.predict(validation_vectors)
                metrics = classification_metrics(validation_labels, predictions, labels)
                record = {
                    "experiment": experiment_number,
                    "model": family["name"],
                    "mode": family["mode"],
                    "ngram_range": list(family["ngram_range"]),
                    "min_df": min_df,
                    "alpha": alpha,
                    "max_features": MAX_FEATURES,
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_macro_precision": metrics["macro_precision"],
                    "validation_macro_recall": metrics["macro_recall"],
                    "vectorization_seconds": round(vectorization_seconds, 3),
                    "model_seconds": round(perf_counter() - model_started, 3),
                }
                experiments.append(record)
                print(
                    f"[{experiment_number:02d}/27] {family['name']} min_df={min_df} alpha={alpha}: "
                    f"macro_f1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}, "
                    f"features={len(vectorizer.vocabulary_)}"
                )
                del classifier
                gc.collect()
            del vectorizer, train_vectors, validation_vectors
            gc.collect()

    best = max(
        experiments,
        key=lambda row: (row["validation_macro_f1"], row["validation_accuracy"], -row["vocabulary_size"]),
    )
    print(f"Selected experiment {best['experiment']}: {best['model']} (validation macro_f1={best['validation_macro_f1']})")

    combined_rows = train_rows + validation_rows
    combined_texts = [row["abstract"] for row in combined_rows]
    combined_labels = [row["category"] for row in combined_rows]
    final_vectorizer = SparseTextVectorizer(
        mode=best["mode"],
        ngram_range=tuple(best["ngram_range"]),
        min_df=best["min_df"],
        max_features=best["max_features"],
        sublinear_tf=True,
    )
    combined_vectors = final_vectorizer.fit_transform(combined_texts)
    final_classifier = MultinomialNaiveBayes(alpha=best["alpha"]).fit(
        combined_vectors, combined_labels, len(final_vectorizer.vocabulary_)
    )
    test_texts = [row["abstract"] for row in test_rows]
    test_labels = [row["category"] for row in test_rows]
    test_vectors = final_vectorizer.transform(test_texts)
    test_predictions = final_classifier.predict(test_vectors)
    test_metrics = classification_metrics(test_labels, test_predictions, labels)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "pure-python MultinomialNB (dependency-free reference implementation)",
        "selection_metric": "validation_macro_f1",
        "text_field": "abstract",
        "best_configuration": best,
        "final_training_records": len(combined_rows),
        "test_records": len(test_rows),
        "class_order": labels,
        "test_metrics": {key: value for key, value in test_metrics.items() if key not in {"per_class", "confusion_matrix"}},
    }
    with (ARTIFACT_DIR / "naive_bayes_model.pkl").open("wb") as handle:
        pickle.dump(
            {"vectorizer": final_vectorizer, "classifier": final_classifier, "metadata": metadata},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    (REPORT_DIR / "nb_experiments.json").write_text(
        json.dumps({"experiments": experiments, "selected": best}, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "nb_test_metrics.json").write_text(
        json.dumps({**metadata, "per_class": test_metrics["per_class"]}, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "nb_discriminative_terms.json").write_text(
        json.dumps(discriminative_terms(final_vectorizer, final_classifier), indent=2) + "\n", encoding="utf-8"
    )
    with (REPORT_DIR / "classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["category", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            writer.writerow({"category": label, **test_metrics["per_class"][label]})
    with (REPORT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, test_metrics["confusion_matrix"], strict=True):
            writer.writerow([label, *row])
    write_confusion_matrix_svg(
        REPORT_DIR / "confusion_matrix.svg", test_metrics["confusion_matrix"], labels
    )
    print(json.dumps({"best_configuration": best, "test_metrics": metadata["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

