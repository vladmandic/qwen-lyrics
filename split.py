import torch
import torchaudio


class Dumucs:
    target_sr: int = 16000
    do_mono: bool = True
    do_normalize: bool = True

    def __init__(self, model: str = 'htdemucs_ft', device: torch.device = torch.device('cuda')):
        from demucs.pretrained import get_model
        self.device = device
        self.model_name = model
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.capabilities: list[str] = list(self.model.sources)

    def __str__(self) -> str:
        return f"VoiceDumucs(model={self.model_name} cls={self.model.__class__.__name__} model_sr={self.model.samplerate} channels={self.model.audio_channels} capabilities={self.capabilities} device={self.device} target_sr={self.target_sr} mono={self.do_mono} normalize={self.do_normalize})"

    def load(self, path: str) -> tuple[torch.Tensor, int]:
        from demucs.audio import AudioFile
        wav = AudioFile(path).read(streams=0, samplerate=self.model.samplerate, channels=self.model.audio_channels)
        return wav, int(self.model.samplerate)

    def save(self, path: str, wav: torch.Tensor, sr: int, *args, **kwargs) -> None:
        torchaudio.save(path, wav, sr, *args, **kwargs)

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
        target_sr: int = target_sr,
        mono: bool = do_mono,
        normalize: bool = do_normalize,
        input_sr: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        from demucs.apply import apply_model
        if isinstance(audio, str):
            audio, input_sr = self.load(audio)
        if input_sr is None:
            raise ValueError("input_sr must be provided when audio is a tensor")
        batch = audio[None]
        with torch.no_grad():
            sources = apply_model(self.model, batch, device="cuda", progress=False)[0]

        dct = []
        for i, source in enumerate(self.model.sources):
            wav, sr = self._postprocess(sources[i].cpu(), int(input_sr), int(target_sr), bool(mono), bool(normalize))
            dct.append({ 'type': source, 'waveform': wav, 'sr': sr })
        return dct


if __name__ == "__main__":
    import time
    import argparse
    from rich import print as rp

    parser = argparse.ArgumentParser(description="Extract vocals")
    parser.add_argument("input", type=str, help="Input audio file path")
    _args = parser.parse_args()

    t0 = time.time()
    d = Dumucs()
    rp(f"Demucs init: {d}")
    t1 = time.time()
    res = d(_args.input)
    t2 = time.time()
    for item in res:
        fn = _args.input.rsplit('.', 1)[0] + f"-{item['type']}.mp3"
        d.save(fn, item['waveform'], item['sr'])
        rp(f"Demucs save: type={item['type']} fn={fn}")

    rp(f"Demucs timers: load={t1 - t0:.2f} process={t2 - t1:.2f} total={t2 - t0:.2f}")
