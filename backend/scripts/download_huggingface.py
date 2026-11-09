"""Download the TimSchopf/medical_abstracts corpus without authentication.

Hugging Face's paginated /rows endpoint can return HTTP 429 after many calls.
The Hugging Face dataset card links to the authors' canonical repository, which
provides the same train/test data as two CSV files. Downloading those files takes
only two requests and requires no account, token, email, pandas, or pyarrow.
"""

import argparse
import csv
import io
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET = "TimSchopf/medical_abstracts"
CANONICAL_REPOSITORY = "https://github.com/sebischair/Medical-Abstracts-TC-Corpus"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/sebischair/"
    "Medical-Abstracts-TC-Corpus/refs/heads/main"
)
SPLIT_URLS = {
    "train": f"{RAW_BASE_URL}/medical_tc_train.csv",
    "test": f"{RAW_BASE_URL}/medical_tc_test.csv",
}
LABELS = {
    1: "Neoplasms",
    2: "Digestive system diseases",
    3: "Nervous system diseases",
    4: "Cardiovascular diseases",
    5: "General pathological conditions",
}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def download_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            request = Request(url, headers={"User-Agent": "medical-document-discovery/0.1"})
            with urlopen(request, timeout=120) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * 2**attempt)
            else:
                delay = min(30, 2**attempt)
        except (URLError, TimeoutError) as exc:
            last_error = exc
            delay = min(30, 2**attempt)
        print(f"Download failed ({last_error}); retrying in {delay}s...")
        time.sleep(delay)
    raise RuntimeError(f"Dataset download failed after retries: {last_error}")


def title_from_abstract(text: str) -> str:
    """The source stores title + abstract together; recover a short title."""
    first_sentence = SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()
    if len(first_sentence) < 10:
        return text.strip()[:180]
    return first_sentence[:300]


def convert_record(label: object, text: object, split: str, row_index: int) -> dict[str, Any]:
    label_id = int(str(label).strip())
    abstract = str(text).strip()
    if label_id not in LABELS:
        raise ValueError(f"Unknown condition label: {label_id}")
    return {
        "id": f"HF_{split}_{row_index}",
        "title": title_from_abstract(abstract),
        "title_is_derived": True,
        "abstract": abstract,
        "category": LABELS[label_id],
        "source": DATASET,
        "source_split": split,
        "source_row": row_index,
        "license": "CC BY-SA 3.0",
        "url": f"https://huggingface.co/datasets/{DATASET}",
        "canonical_repository": CANONICAL_REPOSITORY,
        "authors": [],
        "keywords": [],
        "pmid": None,
        "doi": None,
        "year": None,
        "journal": None,
    }


def parse_split_csv(payload: bytes, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    required = {"condition_label", "medical_abstract"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
    output = []
    for row_index, row in enumerate(reader):
        output.append(
            convert_record(row["condition_label"], row["medical_abstract"], split, row_index)
        )
        if limit is not None and len(output) >= limit:
            break
    return output


def download_split(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    print(f"Downloading {split} in one request...")
    payload = download_bytes(SPLIT_URLS[split])
    records = parse_split_csv(payload, split, limit)
    print(f"{split}: parsed {len(records)} records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/medical_abstracts.jsonl"))
    parser.add_argument("--limit-per-split", type=int, help="Only for a quick pipeline test")
    args = parser.parse_args()

    records = []
    for split in ("train", "test"):
        records.extend(download_split(split, args.limit_per_split))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

