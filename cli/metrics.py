import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import Metrics, load_input_text  # noqa: E402 pylint: disable=wrong-import-position


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute lyric quality metrics between two text/json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("reference", type=Path, help="Reference lyrics input (.txt or .json)")
    parser.add_argument("hypothesis", type=Path, help="Hypothesis lyrics input (.txt or .json)")
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for semantic model")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional model cache directory")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    reference_text = load_input_text(args.reference)
    hypothesis_text = load_input_text(args.hypothesis)

    evaluator = Metrics.with_resources(
        semantic_model=args.semantic_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    metrics = evaluator(reference_text, hypothesis_text)

    if args.output == "json":
        payload = asdict(metrics)
        payload["summary"] = Metrics.summarize(metrics)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Reference: {args.reference}")
        print(f"Hypothesis: {args.hypothesis}")
        print(f"WER={metrics.wer:.6f}")
        print(f"CER={metrics.cer:.6f}")
        print(f"BLEU={metrics.bleu_score:.6f}")
        print(f"PER={metrics.per:.6f}")
        print(f"Semantic={metrics.semantic_similarity:.6f}")
        print(f"Composite={metrics.composite_score:.6f}")
        print(f"Summary={Metrics.summarize(metrics)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
