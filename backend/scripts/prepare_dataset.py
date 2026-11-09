"""Clean, deduplicate, balance, stratify, and report the medical corpus."""

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = {
    "Neoplasms",
    "Digestive system diseases",
    "Nervous system diseases",
    "Cardiovascular diseases",
    "General pathological conditions",
}
SPACE = re.compile(r"\s+")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return SPACE.sub(" ", text).strip()


def fingerprint(value: str) -> str:
    canonical = NON_ALNUM.sub(" ", value.lower()).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON input must contain a list")
        return data
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def clean_and_deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    reasons: Counter = Counter()
    cleaned: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    seen_contents: set[str] = set()

    # The corpus repeats some abstracts under several condition labels. Build
    # the complete label set before retaining any row so the chosen label can
    # never depend on source ordering.
    labels_by_content: dict[str, set[str]] = defaultdict(set)
    for raw in records:
        title = clean_text(raw.get("title"))
        abstract = clean_text(raw.get("abstract"))
        category = clean_text(raw.get("category"))
        if category in CATEGORIES and len(title) >= 10 and len(abstract) >= 80:
            labels_by_content[fingerprint(f"{title} {abstract}")].add(category)

    for raw in records:
        title = clean_text(raw.get("title"))
        abstract = clean_text(raw.get("abstract"))
        category = clean_text(raw.get("category"))
        pmid = clean_text(raw.get("pmid"))
        doi = clean_text(raw.get("doi")).lower()
        if category not in CATEGORIES:
            reasons["invalid_or_ambiguous_category"] += 1
            continue
        if len(title) < 10:
            reasons["missing_or_short_title"] += 1
            continue
        if len(abstract) < 80:
            reasons["missing_or_short_abstract"] += 1
            continue

        title_hash = fingerprint(title)
        content_hash = fingerprint(f"{title} {abstract}")
        # medical_abstracts has no separate title field. Its title is inferred
        # from the first sentence, which is often a repeated section heading
        # such as "BACKGROUND"; it must not be used as a duplicate key.
        title_is_derived = bool(raw.get("title_is_derived")) or raw.get("source") == "TimSchopf/medical_abstracts"
        if len(labels_by_content[content_hash]) > 1:
            reasons["ambiguous_multilabel_content"] += 1
            continue
        if pmid and pmid in seen_pmids:
            reasons["duplicate_pmid"] += 1
            continue
        if doi and doi in seen_dois:
            reasons["duplicate_doi"] += 1
            continue
        if not title_is_derived and title_hash in seen_titles:
            reasons["duplicate_normalized_title"] += 1
            continue
        if content_hash in seen_contents:
            reasons["duplicate_content"] += 1
            continue

        item = dict(raw)
        item.update(
            {
                "id": f"PMID_{pmid}" if pmid else f"DOC_{content_hash[:12]}",
                "title": title,
                "abstract": abstract,
                "category": category,
                "pmid": pmid or None,
                "doi": doi or None,
                "title_is_derived": title_is_derived,
                "authors": [clean_text(author) for author in raw.get("authors", []) if clean_text(author)],
                "keywords": sorted({clean_text(word) for word in raw.get("keywords", []) if clean_text(word)}),
            }
        )
        cleaned.append(item)
        if pmid:
            seen_pmids.add(pmid)
        if doi:
            seen_dois.add(doi)
        if not title_is_derived:
            seen_titles.add(title_hash)
        seen_contents.add(content_hash)
    return cleaned, reasons


def balance_records(records: list[dict[str, Any]], maximum: int | None, seed: int) -> list[dict[str, Any]]:
    if maximum is None:
        return records
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    output = []
    for category in sorted(grouped):
        group = grouped[category]
        rng.shuffle(group)
        output.extend(group[:maximum])
    return output


def stratified_split(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)

    output = []
    for category, group in sorted(grouped.items()):
        if len(group) < 3:
            raise ValueError(f"Category {category!r} needs at least 3 documents to create three splits")
        rng.shuffle(group)
        train_count = max(1, int(len(group) * 0.70))
        validation_count = max(1, int(len(group) * 0.15))
        if train_count + validation_count >= len(group):
            train_count = len(group) - 2
            validation_count = 1
        for index, record in enumerate(group):
            item = dict(record)
            if index < train_count:
                item["split"] = "train"
            elif index < train_count + validation_count:
                item["split"] = "validation"
            else:
                item["split"] = "test"
            output.append(item)
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_review_sample(records: list[dict[str, Any]], per_category: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    sample = []
    for category, group in sorted(grouped.items()):
        for record in rng.sample(group, min(per_category, len(group))):
            sample.append(
                {
                    "id": record["id"], "category": category, "title": record["title"],
                    "abstract_preview": record["abstract"][:500],
                    "source_split": record.get("source_split") or "",
                    "source_row": record.get("source_row", ""),
                    "label_valid": "", "review_notes": ""
                }
            )
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-per-category", type=int)
    parser.add_argument("--review-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_records(args.input)
    cleaned, rejected = clean_and_deduplicate(raw)
    balanced = balance_records(cleaned, args.max_per_category, args.seed)
    split_records = stratified_split(balanced, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all.jsonl", split_records)
    for split in ("train", "validation", "test"):
        write_jsonl(args.output_dir / f"{split}.jsonl", [row for row in split_records if row["split"] == split])

    review = build_review_sample(balanced, args.review_per_category, args.seed)
    with (args.output_dir / "manual_label_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review[0].keys() if review else ["id"])
        writer.writeheader()
        writer.writerows(review)

    summary = {
        "input_records": len(raw),
        "clean_records_before_balancing": len(cleaned),
        "output_records": len(split_records),
        "rejected": dict(rejected),
        "by_category": dict(sorted(Counter(row["category"] for row in split_records).items())),
        "by_split": dict(sorted(Counter(row["split"] for row in split_records).items())),
        "seed": args.seed,
        "split_policy": "Per-category deterministic split, approximately 70/15/15",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

