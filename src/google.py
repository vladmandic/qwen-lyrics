from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from google import genai


class GeminiLyrics:
    DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
    DEFAULT_GENRE = "auto"
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_LANGUAGE = "English"
    OUTPUT_FORMAT_GUARD = (
        " Output only lyrics text. Do not add any prefix such as 'Transcribe' or 'Transcription'. "
        "If output is empty, return an empty string."
    )
    POSSIBLE_MODELS = (
        DEFAULT_MODEL,
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        prompt: str | None = None,
        genres_path: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.prompt = prompt
        self.genres_path = genres_path
        self.genres = self._load_genres(genres_path)

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for GeminiLyrics")
        if self.model not in self.POSSIBLE_MODELS:
            raise ValueError(
                f"Invalid model '{self.model}'. Possible models are: {', '.join(self.POSSIBLE_MODELS)}"
            )

    def __str__(self) -> str:
        return f"GeminiLyrics(model={self.model})"

    def load(self, path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
            sr = target_sr
        return waveform.squeeze(0).numpy().astype(np.float32), int(sr)

    def save(self, path: str, result: dict[str, Any]) -> None:
        output_path = Path(path)
        if output_path.suffix.lower() == ".txt":
            output_path.write_text(result["lyrics"], encoding="utf-8")
        else:
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _create_client(self) -> Any:
        return genai.Client(api_key=self.api_key)

    def _load_genres(self, genres_path: str | None) -> dict[str, Any]:
        path = Path(genres_path or Path(__file__).resolve().parent / "genres.json")
        if not path.exists():
            raise FileNotFoundError(f"Genres file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _resolve_prompt(self, genre: str, language: str, prompt: str | None) -> str:
        if prompt and prompt.strip():
            prompt_text = prompt.strip()
        else:
            genre_name = genre.strip().lower() if genre else self.DEFAULT_GENRE
            preset = self.genres.get(genre_name) or self.genres.get(self.DEFAULT_GENRE, {})
            prompt_text = preset.get(
                "context",
                "Transcribe the song lyrics accurately from the provided audio. Preserve the original wording and style exactly as sung.",
            )
        if language and language.strip() and language.strip().lower() != self.DEFAULT_LANGUAGE.lower():
            prompt_text = f"{prompt_text} Language: {language.strip()}."
        if not prompt_text.endswith("."):
            prompt_text = f"{prompt_text}."
        return f"{prompt_text}{self.OUTPUT_FORMAT_GUARD}"

    def _prepare_audio_file(
        self,
        audio: str | np.ndarray | Any,
        input_sr: int | None,
        sample_rate: int,
        duration: float,
    ) -> tuple[str, bool]:
        if isinstance(audio, str):
            if duration <= 0:
                return audio, False
            wav, _ = self.load(audio, target_sr=sample_rate)
        else:
            if isinstance(audio, np.ndarray):
                wav = audio.astype(np.float32)
            elif hasattr(audio, "numpy"):
                wav = audio.numpy().astype(np.float32)
            else:
                raise TypeError("audio must be a file path, numpy.ndarray, or torch.Tensor")

            if wav.ndim == 2:
                wav = wav.mean(axis=0)

            if input_sr is None:
                raise ValueError("input_sr must be provided for array or tensor audio input")

            sr = int(input_sr)
            if duration > 0:
                wav = wav[: int(duration * sr)]
            if sr != sample_rate:
                wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, sample_rate)
                wav = wav.squeeze(0).numpy()

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(temp_path, torch.from_numpy(np.expand_dims(wav, 0)), sample_rate)
        return temp_path, True

    def _cleanup_temp_file(self, file_path: str, delete: bool) -> None:
        if delete and Path(file_path).exists():
            try:
                Path(file_path).unlink()
            except OSError:
                pass

    def _upload_audio(self, client: Any, file_path: str) -> Any:
        if hasattr(client, "files") and hasattr(client.files, "upload"):
            return client.files.upload(file=file_path, config={"mimeType": "audio/wav"})
        if hasattr(client, "upload_file"):
            return client.upload_file(file=file_path, mimeType="audio/wav")
        if hasattr(client, "upload"):
            return client.upload(file=file_path, mimeType="audio/wav")
        raise RuntimeError("genai client does not support audio file uploads")

    def _get_audio_uri(self, upload_result: Any) -> str:
        if isinstance(upload_result, dict):
            if uri := upload_result.get("uri"):
                return uri
            if uri := upload_result.get("fileUri"):
                return uri
        if hasattr(upload_result, "uri"):
            return getattr(upload_result, "uri")
        if hasattr(upload_result, "fileUri"):
            return getattr(upload_result, "fileUri")
        raise RuntimeError("Could not resolve uploaded audio URI from genai upload result")

    def _generate_content(self, client: Any, audio_uri: str, prompt: str) -> Any:
        params = {
            "model": self.model,
            "contents": {
                "parts": [
                    {"fileData": {"fileUri": audio_uri, "mimeType": "audio/wav"}},
                    {"text": prompt},
                ],
            },
            "config": {
                "responseMimeType": "text/plain",
            },
        }

        if hasattr(client, "models"):
            models = client.models
            if hasattr(models, "generateContent"):
                return models.generateContent(**params)
            if hasattr(models, "generate_content"):
                return models.generate_content(**params)

        if hasattr(client, "generateContent"):
            return client.generateContent(**params)
        if hasattr(client, "generate_content"):
            return client.generate_content(**params)

        raise RuntimeError("genai client does not support generating content")

    def _parse_dict_response(self, response: dict[str, Any]) -> str:
        if text := response.get("text"):
            return text
        if candidates := response.get("candidates"):
            return self._parse_candidates(candidates)
        return ""

    def _parse_candidates(self, candidates: Any) -> str:
        for candidate in candidates or []:
            content = None
            if isinstance(candidate, dict):
                content = candidate.get("content")
            else:
                content = getattr(candidate, "content", None)

            if not content:
                continue

            if isinstance(content, dict):
                parts = content.get("parts", [])
            else:
                parts = getattr(content, "parts", [])

            for part in parts or []:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    return text
        return ""

    def _parse_response(self, response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, dict):
            return self._parse_dict_response(response)
        if hasattr(response, "text") and getattr(response, "text"):
            return getattr(response, "text")
        if hasattr(response, "candidates"):
            return self._parse_candidates(getattr(response, "candidates"))
        return str(response)

    def _transcribe_audio(
        self,
        client: Any,
        audio: str | np.ndarray | Any,
        input_sr: int | None,
        sample_rate: int,
        duration: float,
        resolved_prompt: str,
    ) -> str:
        file_path, cleanup = self._prepare_audio_file(audio, input_sr, sample_rate, duration)
        try:
            upload_result = self._upload_audio(client, file_path)
            audio_uri = self._get_audio_uri(upload_result)
            response = self._generate_content(client, audio_uri, resolved_prompt)
            return self._parse_response(response).strip()
        finally:
            self._cleanup_temp_file(file_path, cleanup)

    def __call__(
        self,
        audio: str | np.ndarray | Any,
        *,
        input_sr: int | None = None,
        genre: str = "auto",
        prompt: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration: float = 0.0,
    ) -> dict[str, Any]:
        client = self._create_client()
        resolved_prompt = self._resolve_prompt(genre, language, prompt)
        lyrics = self._transcribe_audio(client, audio, input_sr, sample_rate, duration, resolved_prompt)

        return {
            "lyrics": lyrics,
            "model": self.model,
            "language": language,
            "genre": genre,
            "prompt": resolved_prompt,
        }
