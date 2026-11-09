"""Validate split integrity and create reproducible dataset audit reports."""

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data" / "processed"
REPORT_DIR = BACKEND_DIR / "reports"
SPLITS = ("train", "validation", "test")
TOKEN = re.compile(r"[a-z][a-z0-9'-]+")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
STOP_WORDS = {
    "the", "and", "of", "in", "to", "a", "with", "for", "was", "were", "is", "are",
    "that", "this", "by", "on", "as", "from", "at", "an", "be", "or", "we", "patients",
    "study", "results", "which", "between", "have", "has", "had", "not", "but", "these",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_fingerprint(record: dict[str, Any]) -> str:
    value = f"{record['title']} {record['abstract']}".lower()
    canonical = NON_ALNUM.sub(" ", value).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def percentile(values: list[int], ratio: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p25": round(percentile(values, 0.25), 2),
        "median": round(float(statistics.median(values)), 2),
        "mean": round(float(statistics.mean(values)), 2),
        "p75": round(percentile(values, 0.75), 2),
        "max": max(values),
    }


def write_distribution_svg(path: Path, counts: dict[str, int]) -> None:
    width, height = 920, 360
    left, top, chart_height = 330, 35, 270
    maximum = max(counts.values())
    rows = []
    for index, (label, count) in enumerate(sorted(counts.items())):
        y = top + index * 55
        bar_width = int((width - left - 80) * count / maximum)
        rows.append(f'<text x="15" y="{y + 20}" font-size="14">{label}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{bar_width}" height="28" rx="4" fill="#2563eb"/>')
        rows.append(f'<text x="{left + bar_width + 10}" y="{y + 20}" font-size="14">{count}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="15" y="25" font-size="19" font-weight="bold">Class distribution after preprocessing</text>'
        + "".join(rows) + '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records_by_split = {split: load_jsonl(DATA_DIR / f"{split}.jsonl") for split in SPLITS}
    errors: list[str] = []
    all_ids: dict[str, str] = {}
    fingerprints: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter = Counter()
    split_counts: dict[str, Counter] = {}
    lengths_by_category: dict[str, list[int]] = defaultdict(list)
    terms_by_category: dict[str, Counter] = defaultdict(Counter)

    required = {"id", "title", "abstract", "category", "split"}
    for split, records in records_by_split.items():
        split_counts[split] = Counter()
        for row_number, record in enumerate(records, start=1):
            missing = required - record.keys()
            if missing:
                errors.append(f"{split}:{row_number} missing fields {sorted(missing)}")
                continue
            if record["split"] != split:
                errors.append(f"{record['id']} has split={record['split']} inside {split}.jsonl")
            if record["id"] in all_ids:
                errors.append(f"duplicate id {record['id']} in {all_ids[record['id']]} and {split}")
            all_ids[record["id"]] = split
            fingerprints[content_fingerprint(record)].add(split)
            category = record["category"]
            class_counts[category] += 1
            split_counts[split][category] += 1
            tokens = TOKEN.findall(record["abstract"].lower())
            lengths_by_category[category].append(len(tokens))
            terms_by_category[category].update(token for token in tokens if token not in STOP_WORDS and len(token) > 2)

    leakage = {fingerprint: sorted(splits) for fingerprint, splits in fingerprints.items() if len(splits) > 1}
    if leakage:
        errors.append(f"{len(leakage)} content fingerprints occur in more than one split")

    length_stats = {category: describe(values) for category, values in sorted(lengths_by_category.items())}
    top_terms = {
        category: [{"term": term, "count": count} for term, count in counts.most_common(20)]
        for category, counts in sorted(terms_by_category.items())
    }
    files = {
        f"{split}.jsonl": {
            "sha256": sha256_file(DATA_DIR / f"{split}.jsonl"),
            "records": len(records_by_split[split]),
        }
        for split in SPLITS
    }
    first_record = next(iter(records_by_split["train"]), {})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": first_record.get("source"),
        "source_url": first_record.get("url"),
        "license": first_record.get("license"),
        "seed": 42,
        "text_field_for_model": "abstract",
        "total_records": sum(len(rows) for rows in records_by_split.values()),
        "classes": sorted(class_counts),
        "files": files,
    }
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "duplicate_ids": len(all_ids) != sum(len(rows) for rows in records_by_split.values()),
        "cross_split_content_leakage": len(leakage),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "abstract_token_lengths": length_stats,
        "top_terms_by_category": top_terms,
        "manual_review_file": "data/processed/manual_label_review.csv",
        "manual_review_status": "PENDING",
    }

    (DATA_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (REPORT_DIR / "class_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", *SPLITS, "total"])
        for category in sorted(class_counts):
            writer.writerow([category, *(split_counts[split][category] for split in SPLITS), class_counts[category]])
    with (REPORT_DIR / "abstract_length_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["category", "min", "p25", "median", "mean", "p75", "max"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for category, values in length_stats.items():
            writer.writerow({"category": category, **values})
    write_distribution_svg(REPORT_DIR / "class_distribution.svg", dict(class_counts))
    print(json.dumps({key: audit[key] for key in ("status", "errors", "cross_split_content_leakage", "class_counts")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

