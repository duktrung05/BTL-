import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


def record(pmid: str, category: str, suffix: str = ""):
    return {
        "pmid": pmid,
        "title": f"A sufficiently long medical document title {pmid} {suffix}",
        "abstract": "A clinically meaningful abstract about medical research and patient outcomes. " * 2,
        "category": category,
        "authors": [" A  Researcher "],
        "keywords": ["health"],
    }


class PrepareDatasetTests(unittest.TestCase):
    def test_clean_removes_duplicate_pmid_and_short_abstract(self):
        rows = [record("1", "Cardiovascular diseases"), record("1", "Cardiovascular diseases", "duplicate"), record("2", "Nervous system diseases")]
        rows.append({**record("3", "Nervous system diseases"), "abstract": "short"})
        cleaned, reasons = module.clean_and_deduplicate(rows)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(reasons["duplicate_pmid"], 1)
        self.assertEqual(reasons["missing_or_short_abstract"], 1)

    def test_split_keeps_each_category_in_all_sets(self):
        rows = [
            record(f"{category[:3]}-{index}", category)
            for category in ("Cardiovascular diseases", "Nervous system diseases")
            for index in range(10)
        ]
        cleaned, _ = module.clean_and_deduplicate(rows)
        output = module.stratified_split(cleaned, seed=42)
        for category in ("Cardiovascular diseases", "Nervous system diseases"):
            self.assertEqual(
                {row["split"] for row in output if row["category"] == category},
                {"train", "validation", "test"},
            )

    def test_derived_repeated_titles_are_not_false_duplicates(self):
        first = record("", "Cardiovascular diseases")
        second = record("", "Cardiovascular diseases")
        first.update({"source": "TimSchopf/medical_abstracts", "title": "BACKGROUND.", "abstract": "A" * 100})
        second.update({"source": "TimSchopf/medical_abstracts", "title": "BACKGROUND.", "abstract": "B" * 100})
        cleaned, reasons = module.clean_and_deduplicate([first, second])
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(reasons["duplicate_normalized_title"], 0)

    def test_same_content_with_conflicting_labels_is_fully_removed(self):
        abstract = "One medical abstract that was assigned to multiple disease categories. " * 2
        first = record("10", "Cardiovascular diseases")
        second = record("11", "Nervous system diseases")
        first.update({"title": "Shared medical research title", "abstract": abstract})
        second.update({"title": "Shared medical research title", "abstract": abstract})
        cleaned, reasons = module.clean_and_deduplicate([first, second])
        self.assertEqual(cleaned, [])
        self.assertEqual(reasons["ambiguous_multilabel_content"], 2)


if __name__ == "__main__":
    unittest.main()

