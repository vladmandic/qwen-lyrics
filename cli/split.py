import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich import print as rp  # noqa: E402 pylint: disable=wrong-import-position
from src.split import Dumucs  # noqa: E402 pylint: disable=wrong-import-position


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract vocals")
    parser.add_argument("input", type=str, help="Input audio file path")
    parser.add_argument("--compile", action="store_true", help="Compile the model for faster inference")
    parser.add_argument("--model", type=str, choices=Dumucs.models, default=Dumucs.models[0], help="Dumucs model to use")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--mono", action="store_true", help="Convert audio to mono")
    parser.add_argument("--save", action="store_true", help="Save the output audio files")
    parser.add_argument("--format", type=str, choices=["mp3", "wav", "flac"], default="mp3", help="Output audio format")

    args = parser.parse_args()

    t0 = time.time()
    d = Dumucs(
        _model=args.model,
        _sr=args.sample_rate,
        _mono=args.mono,
        _normalize=True,
        _compile=args.compile,
    )
    t1 = time.time()
    rp(f"Demucs init: {d}")

    result = d(args.input)
    t2 = time.time()
    result = d(args.input)
    t3 = time.time()

    for item in result:
        fn = args.input.rsplit('.', 1)[0] + f"-{item['type']}.mp3" if args.save else None
        _bytes = d.save(fn, wav=item['waveform'], sr=item['sr'], mode=args.format)
        rp(f"Demucs save: type={item['type']} fn='{fn}' sr={item['sr']} shape={item['waveform'].shape} dtype={item['waveform'].dtype} duration={item['duration']:.2f}")
    t4 = time.time()

    rp(f"Demucs timings: init={t1-t0:.2f} 1st={t2-t1:.2f} 2nd={t3-t2:.2f} save={t4-t3:.2f} total={t4-t0:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
