from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PARAMETER_COLUMNS = [
    "chunk_size_seconds",
    "chunk_overlap_seconds",
    "num_beams",
    "temperature",
    "max_new_tokens",
    "length_penalty",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "do_sample",
    "early_stopping",
]

METRIC_COLUMNS = [
    "wer",
    "cer",
    "bleu_score",
    "per",
    "semantic_similarity",
    "composite_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the top-ranked tuning results from tune.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv-path", type=Path, default=Path("tune.csv"), help="CSV file produced by tune.py")
    parser.add_argument("--limit", type=int, default=10, help="Number of top rows to print")
    parser.add_argument("--song", type=str, default="", help="Optional song_name filter")
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = [column for column in ["song_name", "genre", *METRIC_COLUMNS] if column not in reader.fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"CSV file is missing required columns: {missing_text}")
        return [row for row in reader if row]


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def format_metrics(row: dict[str, str]) -> str:
    return (
        f"score={parse_float(row, 'composite_score'):.2f} "
        f"wer={parse_float(row, 'wer'):.3f} "
        f"cer={parse_float(row, 'cer'):.3f} "
        f"bleu={parse_float(row, 'bleu_score'):.2f} "
        f"per={parse_float(row, 'per'):.3f} "
        f"sem={parse_float(row, 'semantic_similarity'):.3f}"
    )


def format_parameters(row: dict[str, str]) -> str:
    parts = [f"duration={row.get('duration', '0')}s"]
    for key in PARAMETER_COLUMNS:
        if key in row:
            parts.append(f"{key}={row[key]}")
    return ", ".join(parts)


def get_top_by_metric(rows: list[dict[str, str]], metric: str, reverse: bool = True, limit: int = 3) -> list[dict[str, str]]:
    """Get top N rows sorted by a specific metric."""
    return sorted(rows, key=lambda row: parse_float(row, metric), reverse=reverse)[:limit]


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        print("--limit must be greater than 0", file=sys.stderr)
        return 1

    try:
        rows = read_rows(args.csv_path.resolve())
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.song:
        rows = [row for row in rows if row.get("song_name", "") == args.song]

    if not rows:
        print("No ranking rows found.")
        return 0

    print(f"Total tests executed: {len(rows)}")

    # Define metrics with their sort order (True = higher is better, False = lower is better)
    metrics = [
        ("composite_score", True, "Combined Score"),
        ("wer", False, "Word Error Rate (WER)"),
        ("cer", False, "Character Error Rate (CER)"),
        ("bleu_score", True, "BLEU Score"),
        ("per", False, "Phoneme Error Rate (PER)"),
        ("semantic_similarity", True, "Semantic Similarity"),
    ]

    for metric_key, is_higher_better, display_name in metrics:
        ranked = get_top_by_metric(rows, metric_key, reverse=is_higher_better, limit=3)
        print(f"\nTop {len(ranked)} by {display_name}:")
        print("=" * 80)

        for index, row in enumerate(ranked, start=1):
            print(f"{index}. {row.get('song_name', '')} [{row.get('genre', '')}]")
            print(f"   {format_metrics(row)}")
            print(f"   params: {format_parameters(row)}")
            summary = row.get("summary", "").strip()
            if summary:
                print(f"   summary: {summary}")
            timestamp = row.get("timestamp", "").strip()
            if timestamp:
                print(f"   timestamp: {timestamp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
