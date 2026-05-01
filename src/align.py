import json
import re
import threading
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
    CACHE_DIR = None

    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_DURATION = 0.0
    DEFAULT_LANGUAGE = "English"
    DEFAULT_CHUNK_SIZE_SECONDS = 16.0
    DEFAULT_CHUNK_OVERLAP_SECONDS = 0.5
    DEFAULT_TRIM_LEADING_SILENCE = True
    DEFAULT_SILENCE_THRESHOLD = 1e-4
    DEFAULT_MIN_SILENCE_SECONDS = 0.5
    DEFAULT_SILENCE_FRAME_SECONDS = 0.01

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
        }
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir

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

    def _normalize_lyrics_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("lyrics content must be a string")

        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            pass

        # Strip section markers like [Verse 1] to keep alignment focused on sung words.
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("lyrics text is empty after preprocessing")
        return text

    def load_lyrics(self, path: str) -> str:
        path_obj = Path(path)
        raw_text = path_obj.read_text(encoding="utf-8")
        if path_obj.suffix.lower() == ".json":
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse JSON lyrics file: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("JSON lyrics input must be an object with a 'lyrics' field")
            if "lyrics" not in parsed:
                raise ValueError("JSON lyrics input is missing required 'lyrics' field")
            raw_text = parsed["lyrics"]

        return self._normalize_lyrics_text(raw_text)

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

    def _find_leading_silence(
        self,
        wav: np.ndarray,
        sr: int,
        silence_threshold: float,
        min_silence_seconds: float,
        silence_frame_seconds: float,
    ) -> float:
        if len(wav) == 0:
            return 0.0

        frame_samples = max(int(sr * silence_frame_seconds), 1)
        abs_wav = np.abs(wav)
        max_amplitude = float(np.max(abs_wav))
        if max_amplitude <= 0:
            return len(wav) / sr

        threshold = max(silence_threshold, max_amplitude * 0.02)
        total_samples = len(wav)
        position = 0

        while position < total_samples:
            end = min(position + frame_samples, total_samples)
            frame_energy = float(np.mean(abs_wav[position:end]))
            if frame_energy > threshold:
                break
            position += frame_samples

        leading_seconds = position / sr
        if leading_seconds < min_silence_seconds:
            return 0.0
        return leading_seconds

    def _trim_leading_silence(
        self,
        wav: np.ndarray,
        sr: int,
        silence_threshold: float,
        min_silence_seconds: float,
        silence_frame_seconds: float,
    ) -> tuple[np.ndarray, float]:
        leading_seconds = self._find_leading_silence(
            wav,
            sr,
            silence_threshold=silence_threshold,
            min_silence_seconds=min_silence_seconds,
            silence_frame_seconds=silence_frame_seconds,
        )

        if leading_seconds <= 0.0:
            return wav, 0.0

        start_sample = int(round(leading_seconds * sr))
        return wav[start_sample:], leading_seconds

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
        trim_leading_silence: bool = DEFAULT_TRIM_LEADING_SILENCE,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        min_silence_seconds: float = DEFAULT_MIN_SILENCE_SECONDS,
        silence_frame_seconds: float = DEFAULT_SILENCE_FRAME_SECONDS,
    ) -> dict:
        if duration < 0:
            raise ValueError("duration must be >= 0")

        wav, sr = self.load_audio(audio, target_sr=sample_rate)
        original_duration_seconds = len(wav) / sr if sr > 0 else 0.0
        lyrics = self.load_lyrics(lyrics_path)

        leading_silence_seconds = 0.0
        if trim_leading_silence:
            wav, leading_silence_seconds = self._trim_leading_silence(
                wav,
                sr,
                silence_threshold=silence_threshold,
                min_silence_seconds=min_silence_seconds,
                silence_frame_seconds=silence_frame_seconds,
            )

        wav = self._limit_duration(wav, sr, duration)

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

        if leading_silence_seconds > 0.0 and timestamps:
            timestamps = [
                (start + leading_silence_seconds, end + leading_silence_seconds, text)
                for start, end, text in timestamps
            ]

        metadata = {
            "audio_file": str(audio),
            "lyrics_file": str(lyrics_path),
            "sample_rate": sr,
            "language": language,
            "original_duration_seconds": original_duration_seconds,
            "processed_duration_seconds": len(wav) / sr if sr > 0 else 0.0,
            "leading_silence_seconds": leading_silence_seconds,
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
