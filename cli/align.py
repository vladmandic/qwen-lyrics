import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich import print as rp  # noqa: E402 pylint: disable=wrong-import-position
from src.align import LyricsAlign  # noqa: E402 pylint: disable=wrong-import-position


def main() -> int:
    parser = argparse.ArgumentParser(description="LyricsAlign", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("lyrics", type=str, help="Path to lyrics text or JSON file")
    parser.add_argument("--output", type=str, default="", help="Optional output file (.json or .txt)")

    parser.add_argument("--model", type=str, default=LyricsAlign.REPO_ASR)
    parser.add_argument("--aligner", type=str, default=LyricsAlign.REPO_ALIGNER)
    parser.add_argument("--language", type=str, default=LyricsAlign.DEFAULT_LANGUAGE)
    parser.add_argument("--sample-rate", type=int, default=LyricsAlign.DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=LyricsAlign.DEFAULT_DURATION)

    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--dtype", type=str, default="float32", help="Torch dtype: float16|float32|bfloat16")

    parser.add_argument(
        "--use-chunks",
        action="store_true",
        default=False,
        help="Enable chunk fallback if full-audio alignment returns no timestamps",
    )
    parser.add_argument("--chunk-size", type=float, default=LyricsAlign.DEFAULT_CHUNK_SIZE_SECONDS)
    parser.add_argument("--chunk-overlap", type=float, default=LyricsAlign.DEFAULT_CHUNK_OVERLAP_SECONDS)
    parser.add_argument(
        "--no-trim-leading-silence",
        action="store_false",
        dest="trim_leading_silence",
        default=LyricsAlign.DEFAULT_TRIM_LEADING_SILENCE,
        help="Disable trimming of long leading silence before alignment",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=LyricsAlign.DEFAULT_SILENCE_THRESHOLD,
        help="Absolute energy threshold for leading silence detection",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=LyricsAlign.DEFAULT_MIN_SILENCE_SECONDS,
        help="Minimum leading silence duration to trim (seconds)",
    )
    parser.add_argument(
        "--silence-frame-size",
        type=float,
        default=LyricsAlign.DEFAULT_SILENCE_FRAME_SECONDS,
        help="Frame size in seconds used for leading silence detection",
    )

    import torch
    from src.align import _parse_dtype

    args = parser.parse_args()
    rp("LyricsAlign CLI")
    aligner_runner = LyricsAlign(
        model=args.model,
        aligner_name=args.aligner,
        device=torch.device(args.device),
        dtype=_parse_dtype(args.dtype),
    )
    rp(f"Align init: {aligner_runner}")
    rp(
        f'Align config: language="{args.language}" sample_rate={args.sample_rate} duration={args.duration} '
        f'use_chunks={args.use_chunks} chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap} '
        f'trim_leading_silence={args.trim_leading_silence} '
        f'silence_threshold={args.silence_threshold} min_silence_duration={args.min_silence_duration} '
        f'silence_frame_size={args.silence_frame_size}'
    )

    output = aligner_runner(
        args.audio,
        args.lyrics,
        language=args.language,
        sample_rate=args.sample_rate,
        duration=args.duration,
        use_chunks=args.use_chunks,
        chunk_size_seconds=args.chunk_size,
        chunk_overlap_seconds=args.chunk_overlap,
        trim_leading_silence=args.trim_leading_silence,
        silence_threshold=args.silence_threshold,
        min_silence_seconds=args.min_silence_duration,
        silence_frame_seconds=args.silence_frame_size,
    )

    rp(f'Timestamps: {len(output["timestamps"])}')
    for ts_start, ts_end, aligned_text in output["timestamps"]:
        rp(f'  {ts_start:.2f}-{ts_end:.2f}: "{aligned_text}"')

    rp(f'Metadata: {output["metadata"]}')
    if args.output:
        aligner_runner.save(args.output, output)
        rp(f"Save: file={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
