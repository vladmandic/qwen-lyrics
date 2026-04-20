# Qwen-Lyrics

Set of tools for lyrics extraction from audio files using `qwen3-asr` finetuning and `demucs` for voice separation

### Voice separation

Uses `demucs` to separate vocals from the original audio

```bash
python voice.py input.mp3 vocals.mp3
```

### Lyrics extraction

Can work on original or separated vocals

```bash
python lyrics.py vocals.mp3 --genre rap
```

### Forced alignment

When you already have lyrics text and only need timestamps from forced alignment.

```bash
python align.py vocals.mp3 samples/rap.txt --output aligned.json
```

### Params tune

Since `lyrics` implements a lot of differents tuning params, you can run the following command to find the best params for your audio

```bash
python tune.py
```

## Links

- qwen3-asr finetuning: https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- genius.com api: https://docs.genius.com/#/getting-started-h1
- Scraping genius.com: https://medium.com/@rachit.lsoni/scraping-song-lyrics-a-fun-and-practical-guide-c0b07e8e7312
- Demucs#1: https://github.com/adefossez/mdx21_demucs
- Demucs#2: https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A
