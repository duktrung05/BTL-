import unittest

from ml.metrics import classification_metrics
from ml.simple_ml import MultinomialNaiveBayes, SparseTextVectorizer


class SimpleMLTests(unittest.TestCase):
    def test_vectorizer_and_naive_bayes(self):
        texts = [
            "heart cardiac blood pressure hypertension",
            "cardiovascular coronary heart treatment",
            "brain seizure neuron epilepsy",
            "neurological brain epilepsy treatment",
        ]
        labels = ["cardiology", "cardiology", "neurology", "neurology"]
        vectorizer = SparseTextVectorizer(mode="tfidf", ngram_range=(1, 2), min_df=1)
        vectors = vectorizer.fit_transform(texts)
        classifier = MultinomialNaiveBayes(alpha=0.5).fit(vectors, labels, len(vectorizer.vocabulary_))
        predictions = classifier.predict(vectorizer.transform(["coronary heart blood pressure", "brain seizure"]))
        self.assertEqual(predictions, ["cardiology", "neurology"])

    def test_metrics(self):
        metrics = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()

