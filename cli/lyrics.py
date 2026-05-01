import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lyrics import LyricsExtract  # noqa: E402 pylint: disable=wrong-import-position


def main() -> int:
    import argparse
    from rich import print as rp

    parser = argparse.ArgumentParser(description="LyricsExtract", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--output", type=str, default="", help="Optional output file (.json or .txt)")

    parser.add_argument("--model", type=str, default=LyricsExtract.REPO_ASR)
    parser.add_argument("--aligner", type=str, default=LyricsExtract.REPO_ALIGNER)
    parser.add_argument("--genre", type=str, default=LyricsExtract.DEFAULT_GENRE)
    parser.add_argument("--language", type=str, default=LyricsExtract.DEFAULT_LANGUAGE)
    parser.add_argument("--context", type=str, default=LyricsExtract.DEFAULT_CONTEXT)

    parser.add_argument("--sample-rate", type=int, default=LyricsExtract.DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=LyricsExtract.DEFAULT_DURATION)
    parser.add_argument("--chunk-size", type=float, default=LyricsExtract.DEFAULT_CHUNK_SIZE_SECONDS)
    parser.add_argument("--chunk-overlap", type=float, default=LyricsExtract.DEFAULT_CHUNK_OVERLAP_SECONDS)
    parser.add_argument("--batch-size", type=int, default=LyricsExtract.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-beams", type=int, default=LyricsExtract.DEFAULT_NUM_BEAMS)
    parser.add_argument("--temperature", type=float, default=LyricsExtract.DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=LyricsExtract.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--length-penalty", type=float, default=LyricsExtract.DEFAULT_LENGTH_PENALTY)
    parser.add_argument("--repetition-penalty", type=float, default=LyricsExtract.DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=LyricsExtract.DEFAULT_NO_REPEAT_NGRAM_SIZE)
    parser.add_argument("--do-sample", action="store_true", default=LyricsExtract.DEFAULT_DO_SAMPLE)
    parser.add_argument("--early-stopping", action="store_true", default=LyricsExtract.DEFAULT_EARLY_STOPPING)

    args = parser.parse_args()
    rp('LyricsExtract CLI')
    extractor = LyricsExtract(model=args.model, aligner=args.aligner)
    rp(f"Lyrics init: {extractor}")
    rp(f'Lyrics config: genre="{args.genre}" language="{args.language}" context="{args.context}" sample_rate={args.sample_rate} duration={args.duration} chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap} batch_size={args.batch_size} num_beams={args.num_beams} temperature={args.temperature} max_new_tokens={args.max_new_tokens} length_penalty={args.length_penalty} repetition_penalty={args.repetition_penalty} no_repeat_ngram_size={args.no_repeat_ngram_size} do_sample={args.do_sample} early_stopping={args.early_stopping}')
    output = extractor(
        args.audio,
        genre=args.genre,
        context=args.context,
        language=args.language,
        sample_rate=args.sample_rate,
        duration=args.duration,
        chunk_size_seconds=args.chunk_size,
        chunk_overlap_seconds=args.chunk_overlap,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        do_sample=args.do_sample,
        early_stopping=args.early_stopping,
    )

    rp(f'Timestamps: {len(output["timestamps"])}')
    for chunk in output["chunks"]:
        offset = chunk.get("offset_seconds", 0.0)
        text = chunk.get("text", "")
        rp(f'  offset={offset:.2f}: "{text}"')

    rp(f'Metadata: {output.get("metadata", {})}')
    if args.output:
        extractor.save(args.output, output)
        rp(f"Save: file={args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
