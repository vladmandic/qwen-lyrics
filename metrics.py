from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import editdistance
from rich import print as rp

if TYPE_CHECKING:
    from g2p_en import G2p
    from sentence_transformers import SentenceTransformer


COMPOSITE_WEIGHTS = {
    "wer": 0.35,
    "per": 0.25,
    "semantic_similarity": 0.20,
    "cer": 0.10,
    "bleu_score": 0.10,
}

TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b")
BRACKETED_RE = re.compile(r"\[[^\]]*\]")


@dataclass(frozen=True)
class EvaluationMetrics:
    wer: float
    cer: float
    bleu_score: float
    per: float
    semantic_similarity: float
    composite_score: float


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    lowered = lowered.replace("\r\n", "\n").replace("\r", "\n")
    lowered = re.sub(r"[^\w\s']", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def strip_bracketed_annotations(text: str) -> str:
    stripped = BRACKETED_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return TOKEN_RE.findall(normalized)


def tokenize_chars(text: str) -> list[str]:
    normalized = normalize_text(text)
    return list(normalized)


def bleu_score(reference_tokens: list[str], hypothesis_tokens: list[str], max_order: int = 4) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        reference_ngrams = Counter(
            tuple(reference_tokens[index : index + order])
            for index in range(max(0, len(reference_tokens) - order + 1))
        )
        hypothesis_ngrams = Counter(
            tuple(hypothesis_tokens[index : index + order])
            for index in range(max(0, len(hypothesis_tokens) - order + 1))
        )
        if not hypothesis_ngrams:
            precisions.append(1e-9)
            continue

        overlap = sum(min(count, reference_ngrams[ngram]) for ngram, count in hypothesis_ngrams.items())
        total = sum(hypothesis_ngrams.values())
        precisions.append((overlap + 1.0) / (total + 1.0))

    reference_len = len(reference_tokens)
    hypothesis_len = len(hypothesis_tokens)
    if hypothesis_len == 0:
        return 0.0

    brevity_penalty = 1.0 if hypothesis_len > reference_len else math.exp(1.0 - (reference_len / hypothesis_len))
    score = brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_order)
    return score * 100.0


def error_rate(reference_items: list[str], hypothesis_items: list[str]) -> float:
    if not reference_items:
        return 0.0 if not hypothesis_items else 1.0
    distance = editdistance.eval(reference_items, hypothesis_items)
    return distance / len(reference_items)


def semantic_similarity(model: SentenceTransformer, reference_text: str, hypothesis_text: str) -> float:
    if not reference_text.strip() and not hypothesis_text.strip():
        return 1.0
    if not reference_text.strip() or not hypothesis_text.strip():
        return 0.0

    from sentence_transformers.util import cos_sim

    embeddings = model.encode([reference_text, hypothesis_text], convert_to_tensor=True)
    return float(cos_sim(embeddings[0], embeddings[1]).item())


def phonemes_for_text(converter: G2p, text: str) -> list[str]:
    words = tokenize_words(text)
    if not words:
        return []
    phonemes = converter(" ".join(words))
    return [token for token in phonemes if token.strip() and token != " "]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_composite_score(*, wer: float, cer: float, bleu_value: float, per: float, semantic_value: float) -> float:
    normalized = {
        "wer": 1.0 - clamp01(wer),
        "cer": 1.0 - clamp01(cer),
        "bleu_score": clamp01(bleu_value / 100.0),
        "per": 1.0 - clamp01(per),
        "semantic_similarity": clamp01((semantic_value + 1.0) / 2.0),
    }
    weighted_total = sum(COMPOSITE_WEIGHTS[name] * normalized[name] for name in COMPOSITE_WEIGHTS)
    return weighted_total * 100.0


def evaluate_text(
    *,
    reference_text: str,
    hypothesis_text: str,
    phoneme_converter: G2p,
    semantic_model: SentenceTransformer,
) -> EvaluationMetrics:
    cleaned_reference_text = strip_bracketed_annotations(reference_text)

    reference_words = tokenize_words(cleaned_reference_text)
    hypothesis_words = tokenize_words(hypothesis_text)
    reference_chars = tokenize_chars(cleaned_reference_text)
    hypothesis_chars = tokenize_chars(hypothesis_text)
    reference_phonemes = phonemes_for_text(phoneme_converter, cleaned_reference_text)
    hypothesis_phonemes = phonemes_for_text(phoneme_converter, hypothesis_text)

    wer_value = error_rate(reference_words, hypothesis_words)
    cer_value = error_rate(reference_chars, hypothesis_chars)
    bleu_value = bleu_score(reference_words, hypothesis_words)
    per_value = error_rate(reference_phonemes, hypothesis_phonemes)
    semantic_value = semantic_similarity(
        semantic_model,
        normalize_text(cleaned_reference_text),
        normalize_text(hypothesis_text),
    )

    return EvaluationMetrics(
        wer=wer_value,
        cer=cer_value,
        bleu_score=bleu_value,
        per=per_value,
        semantic_similarity=semantic_value,
        composite_score=compute_composite_score(
            wer=wer_value,
            cer=cer_value,
            bleu_value=bleu_value,
            per=per_value,
            semantic_value=semantic_value,
        ),
    )


def summarize_metrics(metrics: EvaluationMetrics) -> str:
    parts: list[str] = []
    if metrics.wer <= 0.2:
        parts.append("strong word match")
    elif metrics.wer <= 0.5:
        parts.append("moderate word match")
    else:
        parts.append("weak word match")

    if metrics.per <= 0.25:
        parts.append("phonetics preserved")
    elif metrics.per <= 0.6:
        parts.append("phonetics partially preserved")
    else:
        parts.append("phonetics drifted")

    if metrics.semantic_similarity >= 0.85:
        parts.append("semantic match high")
    elif metrics.semantic_similarity >= 0.65:
        parts.append("semantic match fair")
    else:
        parts.append("semantic match low")

    if metrics.composite_score >= 85.0:
        parts.append("overall score strong")
    elif metrics.composite_score >= 65.0:
        parts.append("overall score fair")
    else:
        parts.append("overall score weak")

    return "; ".join(parts)


def _coerce_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if parts:
            return "\n".join(parts)
    return None


def load_input_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"json input must be an object: {path}")

        text = _coerce_text(payload.get("lytics"))
        if text is not None:
            return text

        # Compatibility fallback for current repository samples that store 'lyrics'.
        fallback = _coerce_text(payload.get("lyrics"))
        if fallback is not None:
            return fallback

        raise ValueError(f"json input must include a non-empty 'lytics' field: {path}")

    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()

    from g2p_en import G2p
    from sentence_transformers import SentenceTransformer

    reference_text = load_input_text(args.reference)
    hypothesis_text = load_input_text(args.hypothesis)

    rp(f"[bold cyan]Loading semantic model[/bold cyan]: {args.semantic_model} [dim](device={args.device})[/dim]")
    semantic_model = SentenceTransformer(
        args.semantic_model,
        device=args.device,
        cache_folder=args.cache_dir,
    )
    phoneme_converter = G2p()

    metrics = evaluate_text(
        reference_text=reference_text,
        hypothesis_text=hypothesis_text,
        phoneme_converter=phoneme_converter,
        semantic_model=semantic_model,
    )
    summary = summarize_metrics(metrics)

    if args.output == "json":
        payload = asdict(metrics)
        payload["summary"] = summary
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        rp(f"[bold]Reference:[/bold] {args.reference}")
        rp(f"[bold]Hypothesis:[/bold] {args.hypothesis}")
        rp(f"WER={metrics.wer:.6f}")
        rp(f"CER={metrics.cer:.6f}")
        rp(f"BLEU={metrics.bleu_score:.6f}")
        rp(f"PER={metrics.per:.6f}")
        rp(f"Semantic={metrics.semantic_similarity:.6f}")
        rp(f"Composite={metrics.composite_score:.6f}")
        rp(f"Summary={summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
