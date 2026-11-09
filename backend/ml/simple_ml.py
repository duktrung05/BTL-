"""Small, deterministic Count/TF-IDF vectorizers and Multinomial Naive Bayes.

This module is used because the execution environment has no NumPy or
scikit-learn. The formulas match the standard algorithms and the public API is
kept deliberately small so a scikit-learn implementation can replace it later.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9'-]+")


class SparseTextVectorizer:
    def __init__(
        self,
        mode: str = "tfidf",
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_features: int | None = 40_000,
        sublinear_tf: bool = True,
    ) -> None:
        if mode not in {"count", "tfidf"}:
            raise ValueError("mode must be 'count' or 'tfidf'")
        self.mode = mode
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.idf_: list[float] = []

    def _analyze(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        terms: list[str] = []
        minimum, maximum = self.ngram_range
        for size in range(minimum, maximum + 1):
            if size == 1:
                terms.extend(tokens)
            else:
                terms.extend(" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1))
        return terms

    def fit(self, texts: Iterable[str]) -> "SparseTextVectorizer":
        documents = list(texts)
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()
        for text in documents:
            terms = self._analyze(text)
            term_frequency.update(terms)
            document_frequency.update(set(terms))
        candidates = [term for term, frequency in document_frequency.items() if frequency >= self.min_df]
        candidates.sort(key=lambda term: (-term_frequency[term], term))
        if self.max_features is not None:
            candidates = candidates[: self.max_features]
        # Alphabetical feature order makes serialized artifacts reproducible.
        candidates.sort()
        self.vocabulary_ = {term: index for index, term in enumerate(candidates)}
        count = len(documents)
        self.idf_ = [
            math.log((1 + count) / (1 + document_frequency[term])) + 1.0
            for term in candidates
        ]
        return self

    def transform(self, texts: Iterable[str]) -> list[dict[int, float]]:
        if not self.vocabulary_:
            raise RuntimeError("Vectorizer has not been fitted")
        output: list[dict[int, float]] = []
        for text in texts:
            counts: Counter[int] = Counter(
                self.vocabulary_[term] for term in self._analyze(text) if term in self.vocabulary_
            )
            if self.mode == "count":
                output.append({index: float(value) for index, value in counts.items()})
                continue
            values = {
                index: ((1.0 + math.log(value)) if self.sublinear_tf else float(value)) * self.idf_[index]
                for index, value in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in values.values()))
            if norm:
                values = {index: value / norm for index, value in values.items()}
            output.append(values)
        return output

    def fit_transform(self, texts: Iterable[str]) -> list[dict[int, float]]:
        documents = list(texts)
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> list[str]:
        output = [""] * len(self.vocabulary_)
        for term, index in self.vocabulary_.items():
            output[index] = term
        return output


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.classes_: list[str] = []
        self.class_log_prior_: list[float] = []
        self.feature_log_prob_: list[list[float]] = []
        self.n_features_: int = 0

    def fit(self, vectors: list[dict[int, float]], labels: list[str], n_features: int) -> "MultinomialNaiveBayes":
        if len(vectors) != len(labels) or not vectors:
            raise ValueError("vectors and labels must be non-empty and aligned")
        self.classes_ = sorted(set(labels))
        self.n_features_ = n_features
        class_index = {label: index for index, label in enumerate(self.classes_)}
        document_counts = [0] * len(self.classes_)
        feature_counts = [[0.0] * n_features for _ in self.classes_]
        feature_totals = [0.0] * len(self.classes_)
        for vector, label in zip(vectors, labels, strict=True):
            target = class_index[label]
            document_counts[target] += 1
            for feature, value in vector.items():
                feature_counts[target][feature] += value
                feature_totals[target] += value
        total_documents = len(labels)
        self.class_log_prior_ = [math.log(count / total_documents) for count in document_counts]
        self.feature_log_prob_ = []
        for target in range(len(self.classes_)):
            denominator = feature_totals[target] + self.alpha * n_features
            self.feature_log_prob_.append(
                [math.log((count + self.alpha) / denominator) for count in feature_counts[target]]
            )
        return self

    def _joint_log_likelihood(self, vector: dict[int, float]) -> list[float]:
        scores = self.class_log_prior_.copy()
        for target, feature_probabilities in enumerate(self.feature_log_prob_):
            scores[target] += sum(value * feature_probabilities[feature] for feature, value in vector.items())
        return scores

    def predict(self, vectors: Iterable[dict[int, float]]) -> list[str]:
        return [self.classes_[max(range(len(self.classes_)), key=lambda i: scores[i])] for scores in map(self._joint_log_likelihood, vectors)]

    def predict_proba(self, vectors: Iterable[dict[int, float]]) -> list[dict[str, float]]:
        output = []
        for scores in map(self._joint_log_likelihood, vectors):
            maximum = max(scores)
            exponentials = [math.exp(score - maximum) for score in scores]
            total = sum(exponentials)
            output.append({label: value / total for label, value in zip(self.classes_, exponentials, strict=True)})
        return output

