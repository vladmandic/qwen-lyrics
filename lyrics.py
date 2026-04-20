import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import transformers

transformers.logging.set_verbosity_error()


class LyricsExtract:
    REPO_ASR = "Qwen/Qwen3-ASR-1.7B"
    REPO_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
    CACHE_DIR = "/workspace/sdnext/models/huggingface/"

    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHUNK_SIZE_SECONDS = 16.0
    DEFAULT_CHUNK_OVERLAP_SECONDS = 0.5
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_DURATION = 0.0
    DEFAULT_NUM_BEAMS = 1
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_NEW_TOKENS = 256
    DEFAULT_LENGTH_PENALTY = 0.7
    DEFAULT_REPETITION_PENALTY = 0.8
    DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
    DEFAULT_DO_SAMPLE = True
    DEFAULT_EARLY_STOPPING = False
    DEFAULT_GENRE = "auto"
    DEFAULT_LANGUAGE = "English"
    DEFAULT_CONTEXT = ""
    OUTPUT_FORMAT_GUARD = (
        " Output only lyrics text. Do not add any prefix such as 'Transcribe' or 'Transcription'. If output is empty, return an empty string."
    )

    _SHARED_MODELS: dict[tuple[str, str | None, str, str, str], Any] = {}
    _SHARED_LOCKS: dict[tuple[str, str | None, str, str, str], threading.Lock] = {}
    _SHARED_CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        model: str = REPO_ASR,
        aligner: str | None = REPO_ALIGNER,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        cache_dir: str = CACHE_DIR,
        genres_path: str | None = "genres.json",
        share_asr_model: bool = False,
    ):
        from qwen_asr import Qwen3ASRModel

        self.model_name = model
        self.aligner_name = aligner
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.share_asr_model = share_asr_model
        with open(genres_path, encoding="utf-8") as f:
            self.genres = json.load(f)

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
                    if self.aligner_name:
                        self._SHARED_MODELS[model_key] = Qwen3ASRModel.from_pretrained(
                            self.model_name,
                            forced_aligner=self.aligner_name,
                            forced_aligner_kwargs={**load_kwargs},
                            **load_kwargs,
                        )
                    else:
                        self._SHARED_MODELS[model_key] = Qwen3ASRModel.from_pretrained(self.model_name, **load_kwargs)
                    self._SHARED_LOCKS[model_key] = threading.Lock()

                self.asr = self._SHARED_MODELS[model_key]
                self._asr_infer_lock = self._SHARED_LOCKS[model_key]
        else:
            if self.aligner_name:
                self.asr = Qwen3ASRModel.from_pretrained(
                    self.model_name,
                    forced_aligner=self.aligner_name,
                    forced_aligner_kwargs={"dtype": self.dtype, "device_map": str(self.device), "cache_dir": self.cache_dir},
                    **load_kwargs,
                )
            else:
                self.asr = Qwen3ASRModel.from_pretrained(self.model_name, **load_kwargs)
            self._asr_infer_lock = threading.Lock()
        self.asr.model.to(self.device)
        self.asr.model.eval()

    def __str__(self) -> str:
        return (f"LyricsExtract(model={self.model_name} aligner={self.aligner_name} device={self.device} dtype={self.dtype})")

    def _resolve_context(self, genre: str, context: str) -> str:
        if context.strip():
            return f"{context.strip()}{self.OUTPUT_FORMAT_GUARD}"
        genre_name = genre.strip() if genre else ""
        preset = self.genres.get(genre_name) if genre_name else None
        if preset is None:
            preset = self.genres.get(self.DEFAULT_GENRE, {})
        base_context = preset.get(
            "context",
            "Transcribe the song lyrics accurately. Preserve all words and stylistic choices exactly as sung.",
        )
        return f"{base_context}{self.OUTPUT_FORMAT_GUARD}"

    def load(self, path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
            sr = target_sr
        return waveform.squeeze(0).numpy().astype(np.float32), int(sr)

    def save(self, path: str, result: dict) -> None:
        output_path = Path(path)
        if output_path.suffix.lower() == ".txt":
            output_path.write_text(result["lyrics"], encoding="utf-8")
        else:
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _sanitize_generation_kwargs(self, gen_kwargs: dict) -> dict:
        cleaned = dict(gen_kwargs)
        do_sample = bool(cleaned.get("do_sample", False))
        num_beams = int(cleaned.get("num_beams", 1))
        if not do_sample:
            cleaned.pop("temperature", None)
        if num_beams <= 1:
            cleaned.pop("length_penalty", None)
            cleaned.pop("early_stopping", None)
        return cleaned

    def _limit_duration(self, wav: np.ndarray, sr: int, duration: float) -> np.ndarray:
        if duration <= 0:
            return wav
        max_samples = int(duration * sr)
        if max_samples <= 0:
            return wav[:0]
        return wav[:max_samples]

    def _sanitize_chunk_text(self, text: str) -> str:
        if text.lstrip().startswith("Transcribe"):
            return ""
        return text

    def _transcribe_chunk(
        self,
        wav_chunk: np.ndarray,
        context: str,
        language: str,
        gen_kwargs: dict,
    ) -> tuple[str, str]:
        return self._transcribe_chunks([wav_chunk], context, language, gen_kwargs)[0]

    def _transcribe_chunks(
        self,
        wav_chunks: list[np.ndarray],
        context: str,
        language: str,
        gen_kwargs: dict,
    ) -> list[tuple[str, str]]:
        from qwen_asr.inference.utils import parse_asr_output

        if not wav_chunks:
            return []

        force_lang = language if language else None
        prompts = [self.asr._build_text_prompt(context=context, force_language=force_lang) for _ in wav_chunks]  # noqa: SLF001

        with self._asr_infer_lock:
            inputs = self.asr.processor(
                text=prompts,
                audio=wav_chunks,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.asr.model.device)
            inputs = inputs.to(self.asr.model.dtype)
            pad_token_id = getattr(self.asr.model.config, "eos_token_id", None)
            with torch.no_grad():
                outputs = self.asr.model.generate(**inputs, **gen_kwargs, pad_token_id=pad_token_id)

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        decoded_list = self.asr.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: list[tuple[str, str]] = []
        for raw in decoded_list:
            _, text = parse_asr_output(raw, user_language=force_lang)
            text = self._sanitize_chunk_text(text)
            results.append((raw, text))
        return results

    def _align_chunk(
        self,
        wav_chunk: np.ndarray,
        sr: int,
        text: str,
        language: str,
        offset_sec: float,
    ) -> list[tuple[float, float, str]]:
        if not text.strip() or self.asr.forced_aligner is None:
            return []

        with self._asr_infer_lock:
            aligned = self.asr.forced_aligner.align(
                audio=[(wav_chunk, sr)],
                text=[text],
                language=[language],
            )
        if not aligned:
            return []

        align_result = aligned[0]
        if not hasattr(align_result, "items"):
            return []

        items = []
        for item in align_result.items:
            items.append((item.start_time + offset_sec, item.end_time + offset_sec, item.text))
        return items

    def __call__(
        self,
        audio: str | np.ndarray | torch.Tensor,
        *,
        input_sr: int | None = None,
        genre: str = DEFAULT_GENRE,
        context: str = DEFAULT_CONTEXT,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration: float = DEFAULT_DURATION,
        chunk_size_seconds: float = DEFAULT_CHUNK_SIZE_SECONDS,
        chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_beams: int = DEFAULT_NUM_BEAMS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        do_sample: bool = DEFAULT_DO_SAMPLE,
        early_stopping: bool = DEFAULT_EARLY_STOPPING,
    ) -> dict:
        if isinstance(audio, str):
            wav, sr = self.load(audio, target_sr=sample_rate)
        elif isinstance(audio, torch.Tensor):
            if input_sr is None:
                raise ValueError("input_sr must be provided when audio is a tensor")
            wav_tensor = audio
            if wav_tensor.ndim == 2 and wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
            if wav_tensor.ndim == 2 and wav_tensor.shape[0] == 1:
                wav_tensor = wav_tensor.squeeze(0)
            wav = wav_tensor.detach().cpu().numpy().astype(np.float32)
            sr = int(input_sr)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, sample_rate
                ).squeeze(0).numpy()
                sr = sample_rate
        elif isinstance(audio, np.ndarray):
            if input_sr is None:
                raise ValueError("input_sr must be provided when audio is a numpy array")
            wav = audio.astype(np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=0)
            sr = int(input_sr)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, sample_rate
                ).squeeze(0).numpy()
                sr = sample_rate
        else:
            raise TypeError("audio must be path, torch.Tensor, or numpy.ndarray")

        if duration < 0:
            raise ValueError("duration must be >= 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        wav = self._limit_duration(wav, sr, duration)

        resolved_context = self._resolve_context(genre, context)

        # Intentionally do not apply genre generation_overrides yet.
        gen_kwargs = self._sanitize_generation_kwargs(
            {
                "num_beams": num_beams,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "do_sample": do_sample,
                "early_stopping": early_stopping,
            }
        )

        chunks = self._chunk_audio(wav, sr, chunk_size_seconds, chunk_overlap_seconds)
        chunk_results: list[tuple[float, str, str]] = []

        for start_idx in range(0, len(chunks), batch_size):
            batch = chunks[start_idx : start_idx + batch_size]
            batch_wavs = [chunk_wav for chunk_wav, _ in batch]
            batch_transcriptions = self._transcribe_chunks(batch_wavs, resolved_context, language, gen_kwargs)
            for (_, offset_sec), (raw, text) in zip(batch, batch_transcriptions):
                chunk_results.append((offset_sec, text, raw))

        timestamps: list[tuple[float, float, str]] = []
        if self.asr.forced_aligner is not None:
            for (chunk_wav, offset_sec), (_, text, _) in zip(chunks, chunk_results):
                timestamps.extend(self._align_chunk(chunk_wav, sr, text, language, offset_sec))

        lyrics = "\n".join(text for _, text, _ in chunk_results if text.strip())

        result = {
            "lyrics": lyrics,
            "timestamps": timestamps,
            "tokens": [raw for _, _, raw in chunk_results],
            "chunks": [{"offset_seconds": offset, "text": text} for offset, text, _ in chunk_results],
        }
        return result


def _build_parser():
    import argparse
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
    return parser


if __name__ == "__main__":
    from rich import print as rp

    cli_parser = _build_parser()
    args = cli_parser.parse_args()

    cli_t0 = time.time()
    rp('LyricsExtractor CLI')
    extractor = LyricsExtract(
        model=args.model,
        aligner=args.aligner,
    )
    rp(f"Lyrics init: {extractor}")
    cli_t1 = time.time()

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
    cli_t2 = time.time()

    rp(f'Lyrics: "{output["lyrics"]}"')
    rp(f"Timestamps: {len(output['timestamps'])}")
    for ts_start, ts_end, word in output["timestamps"]:
        rp(f'  {ts_start:.2f}-{ts_end:.2f}: "{word}"')
    rp(f"Tokens: {len(output['tokens'])}")
    for i, token in enumerate(output["tokens"]):
        rp(f'  {i}: "{token}"')
    rp(f"Chunks: {len(output['chunks'])}")
    for i, chunk in enumerate(output["chunks"]):
        rp(f'  {i}: offset={chunk["offset_seconds"]:.2f} text="{chunk["text"]}"')

    if args.output:
        extractor.save(args.output, output)
        rp(f"Save: file={args.output}")

    rp(f'Timer: init={cli_t1 - cli_t0:.2f} process={cli_t2 - cli_t1:.2f} total={cli_t2 - cli_t0:.2f}')
