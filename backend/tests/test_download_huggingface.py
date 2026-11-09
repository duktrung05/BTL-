import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_huggingface.py"
spec = importlib.util.spec_from_file_location("download_huggingface", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


class HuggingFaceDownloadTests(unittest.TestCase):
    def test_convert_row(self):
        result = module.convert_record(
            4,
            "Treatment of hypertension. This study evaluates blood pressure control in adults.",
            "train",
            12,
        )
        self.assertEqual(result["category"], "Cardiovascular diseases")
        self.assertEqual(result["title"], "Treatment of hypertension.")
        self.assertEqual(result["id"], "HF_train_12")
        self.assertEqual(result["license"], "CC BY-SA 3.0")
        self.assertTrue(result["title_is_derived"])

    def test_all_five_labels_are_mapped(self):
        self.assertEqual(set(module.LABELS), {1, 2, 3, 4, 5})

    def test_parse_source_csv(self):
        payload = (
            'condition_label,medical_abstract\n'
            '3,"Epilepsy treatment. A sufficiently detailed neurological abstract for testing."\n'
        ).encode()
        rows = module.parse_split_csv(payload, "test")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "Nervous system diseases")


if __name__ == "__main__":
    unittest.main()
