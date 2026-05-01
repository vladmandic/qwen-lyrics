import os
import tempfile
import warnings
import torch
import torchaudio


warnings.filterwarnings("ignore", category=UserWarning)


class Dumucs:
    models: list[str] = ['htdemucs_ft', 'htdemucs', 'hdemucs_mmi']
    target_sr: int = 16000
    do_compile: bool = False
    do_mono: bool = False
    do_normalize: bool = True

    def __init__(self,
                 _model: str = None,
                 _device: torch.device = torch.device('cuda'),
                 _dtype: torch.dtype = torch.float16,
                 _sr: int = target_sr,
                 _compile: bool = do_compile,
                 _mono: bool = do_mono,
                 _normalize: bool = do_normalize,
                ) -> None:
        from demucs.pretrained import get_model
        self.device = _device
        self.dtype = _dtype
        self.model_name = _model if _model is not None else self.models[0]
        self.model = get_model(self.model_name)
        self.model.to(device=self.device, dtype=self.dtype)
        self.target_sr = _sr
        self.do_compile = _compile
        self.do_mono = _mono
        self.do_normalize = _normalize
        if self.do_compile:
            torch._dynamo.config.suppress_errors = True
            compiled = [torch.compile(m, fullgraph=False, mode='reduce-overhead') for m in self.model.models]
            self.model.models = torch.torch.nn.ModuleList(compiled)
        self.model.eval()
        self.capabilities: list[str] = list(self.model.sources)

    def __str__(self) -> str:
        return f"VoiceDumucs(model={self.model_name} cls={self.model.__class__.__name__} model_sr={self.model.samplerate} channels={self.model.audio_channels} capabilities={self.capabilities} device={self.device} target_sr={self.target_sr} mono={self.do_mono} normalize={self.do_normalize} compile={self.do_compile} dtype={self.dtype})"

    def load(self, path: str) -> tuple[torch.Tensor, int]:
        from demucs.audio import AudioFile
        wav = AudioFile(path).read(streams=0, samplerate=self.model.samplerate, channels=self.model.audio_channels)
        return wav, int(self.model.samplerate)

    def save(self, fn: str | None, wav: torch.Tensor, sr: int, mode: str = 'mp3', *args, **kwargs) -> bytes: # pylint: disable=keyword-arg-before-vararg
        suffix = f'.{mode}' if not mode.startswith('.') else mode
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        torchaudio.save(temp_path, wav, sr, format=mode, *args, **kwargs)
        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()
        if fn is not None:
            os.replace(temp_path, fn)
        else:
            os.remove(temp_path)
        return audio_bytes

    def _normalize(self, wav: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
        max_abs = wav.abs().max().item()
        if max_abs > peak:
            return wav * (peak / max_abs)
        return wav

    def _postprocess(
        self,
        wav: torch.Tensor,
        input_sr: int,
        target_sr: int,
        mono: bool,
        normalize: bool,
    ) -> tuple[torch.Tensor, int]:
        self.target_sr = target_sr
        self.do_mono = mono
        self.do_normalize = normalize
        if self.do_mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if input_sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, input_sr, self.target_sr)
        if self.do_normalize:
            wav = self._normalize(wav)
        return wav.contiguous(), self.target_sr

    def __call__(
        self,
        audio: torch.Tensor | str,
        input_sr: int | None = None,
    ) -> list[dict[str, object]]:
        from demucs.apply import apply_model
        if isinstance(audio, str):
            audio, input_sr = self.load(audio)
        if input_sr is None:
            raise ValueError("input_sr must be provided when audio is a tensor")
        batch = audio[None].to(device=self.device, dtype=self.dtype)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            sources = apply_model(
                self.model,
                mix=batch,
                shifts=1,
                split=True,
                overlap=0.25,
                transition_power=1.0,
                # segment=Fraction(39, 5),
                pool=None,
                device=self.device,
                progress=False
            )[0]

        dct = []
        for i, source in enumerate(self.model.sources):
            duration = sources[i].shape[-1] / self.model.samplerate
            wav, sr = self._postprocess(sources[i].cpu(), int(input_sr), int(self.target_sr), bool(self.do_mono), bool(self.do_normalize))
            dct.append({ 'type': source, 'waveform': wav, 'sr': sr, 'duration': duration })
        return dct
