from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Iterable

import torch
from rich import print as rp

from src.lyrics import LyricsExtract
from src.metrics import EvaluationMetrics, evaluate_text, summarize_metrics

METRIC_COLUMNS = [
    "wer",
    "cer",
    "bleu_score",
    "per",
    "semantic_similarity",
    "composite_score",
]

PARAMETER_GRID = {
    "chunk_size_seconds": [16.0, 10.0, 4.0],
    "chunk_overlap_seconds": [0.5, 1.0, 2.0],
    "num_beams": [1, 2, 4],
    "temperature": [0.75, 0.5, 1.0],
    "max_new_tokens": [256],
    "length_penalty": [0.75, 0.5, 1.0],
    "repetition_penalty": [0.75, 0.5, 1.0],
    "no_repeat_ngram_size": [2, 3],
    "do_sample": [True, False],
    "early_stopping": [True, False],
}

PARAMETER_COLUMNS = list(PARAMETER_GRID.keys())
CSV_COLUMNS = [
    "song_name",
    "genre",
    "duration",
    *PARAMETER_COLUMNS,
    *METRIC_COLUMNS,
    "summary",
    "timestamp",
]


@dataclass(frozen=True)
class EvaluationResult:
    row: dict[str, object]
    metrics: EvaluationMetrics
    hypothesis_text: str
    output: dict[str, object]
    params: dict[str, object]
    extraction_time: float
    processing_time: float
    total_eval_time: float


def validate_args(args) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    if "cuda" in args.device.lower():
        if not torch.cuda.is_available():
            raise ValueError(f"Device '{args.device}' requested but CUDA is not available. Use --device cpu instead.")
    rp(f"[bold cyan]Using device:[/bold cyan] {args.device} (CUDA available: {torch.cuda.is_available()})")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def get_default_parameters() -> dict[str, object]:
    defaults = {
        "chunk_size_seconds": LyricsExtract.DEFAULT_CHUNK_SIZE_SECONDS,
        "chunk_overlap_seconds": LyricsExtract.DEFAULT_CHUNK_OVERLAP_SECONDS,
        "num_beams": LyricsExtract.DEFAULT_NUM_BEAMS,
        "temperature": LyricsExtract.DEFAULT_TEMPERATURE,
        "max_new_tokens": LyricsExtract.DEFAULT_MAX_NEW_TOKENS,
        "length_penalty": LyricsExtract.DEFAULT_LENGTH_PENALTY,
        "repetition_penalty": LyricsExtract.DEFAULT_REPETITION_PENALTY,
        "no_repeat_ngram_size": LyricsExtract.DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "do_sample": LyricsExtract.DEFAULT_DO_SAMPLE,
        "early_stopping": LyricsExtract.DEFAULT_EARLY_STOPPING,
    }

    params: dict[str, object] = {}
    for name in PARAMETER_COLUMNS:
        grid_values = PARAMETER_GRID[name]
        default_val = defaults.get(name)
        if default_val in grid_values:
            params[name] = default_val
        else:
            if isinstance(default_val, (int, float)) and isinstance(grid_values[0], (int, float)):
                target = float(default_val)
                best_value = grid_values[0]
                best_distance = abs(float(best_value) - target)
                for candidate in grid_values[1:]:
                    distance = abs(float(candidate) - target)
                    if distance < best_distance:
                        best_distance = distance
                        best_value = candidate
                params[name] = best_value
            else:
                params[name] = grid_values[0]

    return params


def parameter_combinations() -> Iterable[dict[str, object]]:
    default_params = get_default_parameters()
    yield default_params

    values = [PARAMETER_GRID[name] for name in PARAMETER_COLUMNS]
    for combination in itertools.product(*values):
        params = dict(zip(PARAMETER_COLUMNS, combination, strict=True))
        if params != default_params:
            yield params


def describe_variation(params: dict[str, object], baseline: dict[str, object] | None) -> str:
    if baseline is None:
        return "initial_run"

    changed = [
        f"{name}={params[name]}"
        for name in PARAMETER_COLUMNS
        if params.get(name) != baseline.get(name)
    ]
    if not changed:
        return "no_change"
    return ", ".join(changed)


def dedupe_key(song_name: str, duration: float, params: dict[str, object]) -> tuple[str, ...]:
    row = [song_name, str(duration)]
    for name in PARAMETER_COLUMNS:
        row.append(str(params[name]))
    return tuple(row)


def load_existing_results(csv_path: Path) -> set[tuple[str, ...]]:
    if not csv_path.exists():
        return set()

    seen: set[tuple[str, ...]] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            key_values = [row.get("song_name", ""), row.get("duration", "0.0")]
            for name in PARAMETER_COLUMNS:
                key_values.append(row.get(name, ""))
            seen.add(tuple(key_values))
    return seen


def ensure_csv_header(csv_path: Path) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def discover_song_pairs(samples_dir: Path, selected_songs: set[str]) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    audio_files = sorted(samples_dir.glob("*.mp3"))
    stems = {audio_path.stem for audio_path in audio_files}

    grouped: dict[str, dict[str, Path | None]] = {}

    for audio_path in audio_files:
        stem = audio_path.stem
        base_name = stem
        variant_name = ""

        if "-" in stem:
            possible_base, possible_variant = stem.rsplit("-", 1)
            if possible_base in stems:
                base_name = possible_base
                variant_name = possible_variant

        entry = grouped.setdefault(base_name, {"base": None, "vocals": None})
        if variant_name == "vocals":
            entry["vocals"] = audio_path
        elif variant_name:
            continue
        else:
            entry["base"] = audio_path

    for song_name in sorted(grouped):
        if selected_songs and song_name not in selected_songs:
            continue

        selected_audio = grouped[song_name]["vocals"] or grouped[song_name]["base"]
        if selected_audio is None:
            continue

        lyrics_path = samples_dir / f"{song_name}.txt"
        if not lyrics_path.exists():
            print(f"Skipping {selected_audio.name}: missing reference lyrics {lyrics_path.name}", file=sys.stderr)
            continue

        pairs.append((selected_audio, lyrics_path, song_name))

    return pairs


def append_result(csv_path: Path, row: dict[str, object]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def save_best_output(path: Path, output: dict[str, object]) -> None:
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def load_semantic_model(args):
    from sentence_transformers import SentenceTransformer

    rp(f"[bold cyan]Loading semantic model[/bold cyan]: {args.semantic_model} [dim](device={args.device})[/dim]")
    return SentenceTransformer(
        args.semantic_model,
        device=args.device,
        cache_folder=args.cache_dir,
    )


def build_evaluation_fn(args, semantic_model):
    worker_local = threading.local()
    device_reported = threading.Event()
    single_worker_resources: dict[str, object] = {}

    if args.workers == 1:
        from g2p_en import G2p

        init_t0 = time.monotonic()
        single_worker_resources["phoneme_converter"] = G2p()
        single_worker_resources["extractor"] = LyricsExtract(
            model=args.model,
            aligner=args.aligner,
            device=torch.device(args.device),
            dtype=resolve_dtype(args.dtype),
            cache_dir=args.cache_dir,
            share_asr_model=False,
        )
        single_worker_resources["init_time"] = time.monotonic() - init_t0

    def evaluate(
        *,
        audio_path: Path,
        genre: str,
        song_name: str,
        reference_text: str,
        params: dict[str, object],
    ) -> EvaluationResult:
        if args.workers == 1:
            extractor = single_worker_resources["extractor"]
            phoneme_converter = single_worker_resources["phoneme_converter"]
        else:
            if not hasattr(worker_local, "extractor"):
                from g2p_en import G2p

                worker_local.phoneme_converter = G2p()
                worker_local.extractor = LyricsExtract(
                    model=args.model,
                    aligner=args.aligner,
                    device=torch.device(args.device),
                    dtype=resolve_dtype(args.dtype),
                    cache_dir=args.cache_dir,
                    share_asr_model=True,
                )
            extractor = worker_local.extractor
            phoneme_converter = worker_local.phoneme_converter

        if not device_reported.is_set():
            asr_param = next(extractor.asr.model.parameters())
            semantic_device = getattr(semantic_model, "device", None)
            if semantic_device is None:
                semantic_device = getattr(semantic_model, "_target_device", "unknown")
            rp(
                "[bold cyan]Runtime devices:[/bold cyan] "
                f"asr={asr_param.device} asr_dtype={asr_param.dtype} "
                f"semantic={semantic_device}"
            )
            device_reported.set()

        _t0_eval = time.monotonic()
        _t0_extract = time.monotonic()
        result = extractor(
            str(audio_path),
            genre=genre,
            context=args.context,
            language=args.language,
            duration=args.duration,
            batch_size=args.batch,
            **params,
        )
        extraction_time = time.monotonic() - _t0_extract
        hypothesis_text = result.get("lyrics", "")

        _t0_process = time.monotonic()
        metrics = evaluate_text(
            reference_text=reference_text,
            hypothesis_text=hypothesis_text,
            phoneme_converter=phoneme_converter,
            semantic_model=semantic_model,
        )
        processing_time = time.monotonic() - _t0_process
        total_eval_time = time.monotonic() - _t0_eval
        row = {
            "song_name": song_name,
            "genre": genre,
            "duration": f"{args.duration}",
            **params,
            "wer": f"{metrics.wer:.6f}",
            "cer": f"{metrics.cer:.6f}",
            "bleu_score": f"{metrics.bleu_score:.6f}",
            "per": f"{metrics.per:.6f}",
            "semantic_similarity": f"{metrics.semantic_similarity:.6f}",
            "composite_score": f"{metrics.composite_score:.6f}",
            "summary": summarize_metrics(metrics),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return EvaluationResult(
            row=row,
            metrics=metrics,
            hypothesis_text=hypothesis_text,
            output=result,
            params=params,
            extraction_time=extraction_time,
            processing_time=processing_time,
            total_eval_time=total_eval_time,
        )

    return evaluate, single_worker_resources.get("init_time")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune LyricsExtract parameters against reference lyrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"), help="Directory containing .mp3/.txt pairs")
    parser.add_argument("--csv-path", type=Path, default=Path("tune.csv"), help="CSV file to append results to")
    parser.add_argument("--language", type=str, default=LyricsExtract.DEFAULT_LANGUAGE, help="Language passed to LyricsExtract")
    parser.add_argument("--context", type=str, default=LyricsExtract.DEFAULT_CONTEXT, help="Optional fixed prompt context")
    parser.add_argument("--duration", type=float, default=LyricsExtract.DEFAULT_DURATION, help="Process only the first N seconds of each song; 0 means full song")
    parser.add_argument("--batch", type=int, default=LyricsExtract.DEFAULT_BATCH_SIZE, help="Batch size passed to LyricsExtract chunk transcription")
    parser.add_argument("--model", type=str, default=LyricsExtract.REPO_ASR, help="ASR model repo or path")
    parser.add_argument("--aligner", type=str, default=LyricsExtract.REPO_ALIGNER, help="Forced aligner repo or path")
    parser.add_argument("--cache-dir", type=str, default=LyricsExtract.CACHE_DIR, help="Hugging Face cache directory")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for model loading")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Torch dtype")
    parser.add_argument("--semantic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on new evaluations for short test runs")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent evaluations to run")
    parser.add_argument("--songs", nargs="*", default=[], help="Optional song stems to process, e.g. rap rock")
    parser.add_argument("--reset", action="store_true", help="Clear the CSV output file before starting")
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        rp(f"[bold red]Error:[/bold red] {exc}")
        return 1

    samples_dir = args.samples_dir.resolve()
    csv_path = args.csv_path.resolve()

    if not samples_dir.exists():
        rp(f"[bold red]Error:[/bold red] samples directory not found: {samples_dir}")
        return 1

    if args.reset and csv_path.exists():
        csv_path.unlink()
        rp(f"[yellow]Reset[/yellow] removed existing CSV: {csv_path}")

    ensure_csv_header(csv_path)
    seen = load_existing_results(csv_path)
    songs = discover_song_pairs(samples_dir, set(args.songs))
    if not songs:
        rp("[bold red]Error:[/bold red] no song/reference pairs found.")
        return 1

    all_params = list(parameter_combinations())
    total_combinations_per_song = len(all_params)
    pending_by_song: dict[str, int] = {}
    for _audio_path, _reference_path, song_name in songs:
        pending_count = 0
        for params in all_params:
            if dedupe_key(song_name, args.duration, params) not in seen:
                pending_count += 1
        pending_by_song[song_name] = pending_count

    total_pending = sum(pending_by_song.values())
    approx_pending_per_song = (total_pending / len(songs)) if songs else 0.0
    rp(
        "Run estimate: "
        f"[bold]songs[/bold]={len(songs)} "
        f"[bold]combinations_per_song[/bold]={total_combinations_per_song} "
        f"[bold]pending_total[/bold]={total_pending} "
        f"[bold]pending_per_song_avg[/bold]={approx_pending_per_song:.1f}"
    )

    semantic_model = load_semantic_model(args)
    rp(
        "Preparing run: "
        f"[bold]workers[/bold]={args.workers} "
        f"[bold]batch[/bold]={args.batch} "
        f"[bold]device[/bold]={args.device} "
        f"[bold]dtype[/bold]={args.dtype}"
    )
    evaluate_configuration, single_worker_init_time = build_evaluation_fn(args, semantic_model)
    if single_worker_init_time is not None:
        rp(f"Single-worker extractor init={single_worker_init_time:.1f}s")

    total_new_runs = 0
    for audio_path, reference_path, song_name in songs:
        genre = song_name
        reference_text = reference_path.read_text(encoding="utf-8")
        best_out_path = audio_path.with_suffix(".out")
        best_score = float("-inf")
        rp(f"\n[bold magenta]Processing[/bold magenta] {audio_path.name} [dim](genre={genre})[/dim]")

        pending_params: list[dict[str, object]] = []
        for params in all_params:
            key = dedupe_key(song_name, args.duration, params)
            if key in seen:
                continue

            if args.limit and (total_new_runs + len(pending_params)) >= args.limit:
                rp(f"[yellow]Reached limit={args.limit}; stopping early.[/yellow]")
                break

            pending_params.append(params)

        if not pending_params:
            rp(f"[dim]  no new parameter combinations for {song_name}[/dim]")
            continue

        previous_params: dict[str, object] | None = None

        if args.workers == 1:
            for params in pending_params:
                variation = describe_variation(params, previous_params)
                rp(f"  [cyan]starting[/cyan] {song_name} varied={variation} params={params}")
                evaluated = evaluate_configuration(
                    audio_path=audio_path,
                    genre=genre,
                    song_name=song_name,
                    reference_text=reference_text,
                    params=params,
                )
                key = dedupe_key(song_name, args.duration, params)
                append_result(csv_path, evaluated.row)
                seen.add(key)
                total_new_runs += 1

                if evaluated.metrics.composite_score > best_score:
                    best_score = evaluated.metrics.composite_score
                    save_best_output(best_out_path, evaluated.output)

                rp(
                    f"  [green]finished[/green] {song_name} "
                    f"wer={evaluated.metrics.wer:.3f} cer={evaluated.metrics.cer:.3f} "
                    f"bleu={evaluated.metrics.bleu_score:.2f} per={evaluated.metrics.per:.3f} "
                    f"sem={evaluated.metrics.semantic_similarity:.3f} "
                    f"[bold]score={evaluated.metrics.composite_score:.2f}[/bold] "
                    f"extract={evaluated.extraction_time:.1f}s process={evaluated.processing_time:.1f}s "
                    f"total={evaluated.total_eval_time:.1f}s"
                )
                previous_params = params
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_start: dict[Future[EvaluationResult], float] = {}
                for params in pending_params:
                    variation = describe_variation(params, previous_params)
                    rp(f"  [cyan]starting[/cyan] {song_name} varied={variation} params={params}")
                    fut = executor.submit(
                        evaluate_configuration,
                        audio_path=audio_path,
                        genre=genre,
                        song_name=song_name,
                        reference_text=reference_text,
                        params=params,
                    )
                    future_start[fut] = time.monotonic()
                    previous_params = params

                for future in as_completed(future_start):
                    evaluated = future.result()
                    key = dedupe_key(song_name, args.duration, evaluated.params)
                    append_result(csv_path, evaluated.row)
                    seen.add(key)
                    total_new_runs += 1

                    if evaluated.metrics.composite_score > best_score:
                        best_score = evaluated.metrics.composite_score
                        save_best_output(best_out_path, evaluated.output)

                    rp(
                        f"  [green]finished[/green] {song_name} params={evaluated.params} "
                        f"wer={evaluated.metrics.wer:.3f} cer={evaluated.metrics.cer:.3f} "
                        f"bleu={evaluated.metrics.bleu_score:.2f} per={evaluated.metrics.per:.3f} "
                        f"sem={evaluated.metrics.semantic_similarity:.3f} "
                        f"[bold]score={evaluated.metrics.composite_score:.2f}[/bold] "
                        f"extract={evaluated.extraction_time:.1f}s process={evaluated.processing_time:.1f}s "
                        f"total={evaluated.total_eval_time:.1f}s"
                    )

        if best_score > float("-inf"):
            rp(f"[bold blue]Best output[/bold blue] for {song_name}: score={best_score:.2f} file={best_out_path}")

    rp(f"[bold green]Completed[/bold green] {total_new_runs} new evaluations. Results appended to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
