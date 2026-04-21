import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import transformers

transformers.logging.set_verbosity_error()


class LyricsAlign:
    REPO_ASR = "Qwen/Qwen3-ASR-1.7B"
    REPO_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
    CACHE_DIR = "/workspace/sdnext/models/huggingface/"

    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_DURATION = 0.0
    DEFAULT_LANGUAGE = "English"
    DEFAULT_CHUNK_SIZE_SECONDS = 16.0
    DEFAULT_CHUNK_OVERLAP_SECONDS = 0.5

    _SHARED_MODELS: dict[tuple[str, str, str, str, str], Any] = {}
    _SHARED_LOCKS: dict[tuple[str, str, str, str, str], threading.Lock] = {}
    _SHARED_CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        model: str = REPO_ASR,
        aligner_name: str = REPO_ALIGNER,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        cache_dir: str = CACHE_DIR,
        share_asr_model: bool = False,
    ):
        from qwen_asr import Qwen3ASRModel

        self.model_name = model
        self.aligner_name = aligner_name
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.share_asr_model = share_asr_model

        load_kwargs = {
            "dtype": self.dtype,
            "device_map": str(self.device),
            "cache_dir": self.cache_dir,
        }

        model_key = (
            self.model_name,
            self.aligner_name,
            str(self.device),
            str(self.dtype),
            self.cache_dir,
        )

        if self.share_asr_model:
            with self._SHARED_CACHE_LOCK:
                if model_key not in self._SHARED_MODELS:
                    self._SHARED_MODELS[model_key] = Qwen3ASRModel.from_pretrained(
                        self.model_name,
                        forced_aligner=self.aligner_name,
                        forced_aligner_kwargs={**load_kwargs},
                        **load_kwargs,
                    )
                    self._SHARED_LOCKS[model_key] = threading.Lock()

                self.asr = self._SHARED_MODELS[model_key]
                self._asr_infer_lock = self._SHARED_LOCKS[model_key]
        else:
            self.asr = Qwen3ASRModel.from_pretrained(
                self.model_name,
                forced_aligner=self.aligner_name,
                forced_aligner_kwargs={**load_kwargs},
                **load_kwargs,
            )
            self._asr_infer_lock = threading.Lock()

        self.asr.model.to(self.device)
        self.asr.model.eval()

    def __str__(self) -> str:
        return f"LyricsAlign(model={self.model_name} aligner={self.aligner_name} device={self.device} dtype={self.dtype})"

    def load_audio(self, path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
            sr = target_sr
        return waveform.squeeze(0).numpy().astype(np.float32), int(sr)

    def load_lyrics(self, path: str) -> str:
        text = Path(path).read_text(encoding="utf-8")
        # Strip section markers like [Verse 1] to keep alignment focused on sung words.
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("lyrics text is empty after preprocessing")
        return text

    def save(self, path: str, result: dict) -> None:
        output_path = Path(path)
        if output_path.suffix.lower() == ".txt":
            lines = [f"{start:.2f}-{end:.2f}: {aligned_text}" for start, end, aligned_text in result["timestamps"]]
            output_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _limit_duration(self, wav: np.ndarray, sr: int, duration: float) -> np.ndarray:
        if duration <= 0:
            return wav
        max_samples = int(duration * sr)
        if max_samples <= 0:
            return wav[:0]
        return wav[:max_samples]

    def _chunk_audio(self, wav: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float) -> list[tuple[np.ndarray, float]]:
        chunk_samples = int(chunk_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        step = max(chunk_samples - overlap_samples, 1)
        total = len(wav)
        chunks = []
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            chunks.append((wav[start:end], start / sr))
            if end >= total:
                break
            start += step
        return chunks

    def _align_single(self, wav: np.ndarray, sr: int, text: str, language: str) -> list[tuple[float, float, str]]:
        if not self.asr.forced_aligner:
            return []
        with self._asr_infer_lock:
            aligned = self.asr.forced_aligner.align(
                audio=[(wav, sr)],
                text=[text],
                language=[language],
            )

        if not aligned:
            return []

        align_result = aligned[0]
        if not hasattr(align_result, "items"):
            return []

        return [(item.start_time, item.end_time, item.text) for item in align_result.items]

    def _align_chunk_fallback(
        self,
        wav: np.ndarray,
        sr: int,
        text: str,
        language: str,
        chunk_size_seconds: float,
        chunk_overlap_seconds: float,
    ) -> list[tuple[float, float, str]]:
        chunks = self._chunk_audio(wav, sr, chunk_size_seconds, chunk_overlap_seconds)
        remaining_words = text.split()
        timestamps: list[tuple[float, float, str]] = []

        for chunk_wav, offset_sec in chunks:
            if not remaining_words:
                break
            remaining_text = " ".join(remaining_words)
            chunk_items = self._align_single(chunk_wav, sr, remaining_text, language)
            if not chunk_items:
                continue

            for start_time, end_time, item_text in chunk_items:
                timestamps.append((start_time + offset_sec, end_time + offset_sec, item_text))

            consume_count = min(len(chunk_items), len(remaining_words))
            remaining_words = remaining_words[consume_count:]

        return timestamps

    def __call__(
        self,
        audio: str,
        lyrics_path: str,
        *,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration: float = DEFAULT_DURATION,
        use_chunks: bool = False,
        chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
        chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    ) -> dict:
        if duration < 0:
            raise ValueError("duration must be >= 0")

        wav, sr = self.load_audio(audio, target_sr=sample_rate)
        wav = self._limit_duration(wav, sr, duration)
        lyrics = self.load_lyrics(lyrics_path)

        timestamps = self._align_single(wav, sr, lyrics, language)
        alignment_mode = "full"

        if use_chunks and not timestamps:
            timestamps = self._align_chunk_fallback(
                wav,
                sr,
                lyrics,
                language,
                chunk_size_seconds,
                chunk_overlap_seconds,
            )
            alignment_mode = "chunk-fallback"

        metadata = {
            "audio_file": str(audio),
            "lyrics_file": str(lyrics_path),
            "sample_rate": sr,
            "language": language,
            "duration_seconds": len(wav) / sr if sr > 0 else 0.0,
            "alignment_mode": alignment_mode,
            "model": self.model_name,
            "aligner": self.aligner_name,
        }

        return {
            "lyrics": lyrics,
            "timestamps": timestamps,
            "metadata": metadata,
        }


def _parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.strip().lower()
    if key not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype_str}")
    return dtype_map[key]


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="LyricsAlign", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("lyrics", type=str, help="Path to lyrics text file")
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

    return parser


if __name__ == "__main__":
    from rich import print as rp

    cli_parser = _build_parser()
    args = cli_parser.parse_args()

    cli_t0 = time.time()
    rp("LyricsAlign CLI")
    aligner_runner = LyricsAlign(
        model=args.model,
        aligner_name=args.aligner,
        device=torch.device(args.device),
        dtype=_parse_dtype(args.dtype),
    )
    rp(f"Align init: {aligner_runner}")
    cli_t1 = time.time()

    rp(
        f'Align config: language="{args.language}" sample_rate={args.sample_rate} duration={args.duration} '
        f'use_chunks={args.use_chunks} chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}'
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
    )
    cli_t2 = time.time()

    rp(f'Timestamps: {len(output["timestamps"])}')
    for ts_start, ts_end, aligned_text in output["timestamps"]:
        rp(f'  {ts_start:.2f}-{ts_end:.2f}: "{aligned_text}"')

    rp(f'Metadata: {output["metadata"]}')

    if args.output:
        aligner_runner.save(args.output, output)
        rp(f"Save: file={args.output}")

    rp(f'Timer: init={cli_t1 - cli_t0:.2f} process={cli_t2 - cli_t1:.2f} total={cli_t2 - cli_t0:.2f}')
