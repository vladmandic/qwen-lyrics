import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.google import GeminiLyrics  # noqa: E402 pylint: disable=wrong-import-position


def main() -> int:
    import argparse
    from rich import print as rp

    parser = argparse.ArgumentParser(description="GeminiLyrics CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--model", type=str, default=GeminiLyrics.DEFAULT_MODEL, help="Gemini model to use")
    parser.add_argument("--genre", type=str, default=GeminiLyrics.DEFAULT_GENRE, help="Genre preset for prompt generation")
    parser.add_argument("--output", type=str, default="", help="Optional output file (.json or .txt)")

    args = parser.parse_args()
    rp("GeminiLyrics CLI")
    rp(f"Audio: {args.audio}")
    rp(f"Model: {args.model}")
    rp(f"Genre: {args.genre}")

    extractor = GeminiLyrics(model=args.model)
    rp(f"GeminiLyrics init: {extractor}")

    result = extractor(args.audio, genre=args.genre)
    rp(f"Lyrics:\n{result['lyrics']}\n")

    if args.output:
        extractor.save(args.output, result)
        rp(f"Saved output to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
