## Links

- qwen3-asr finetuning: https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- genius.com api: https://docs.genius.com/#/getting-started-h1
- Scraping genius.com: https://medium.com/@rachit.lsoni/scraping-song-lyrics-a-fun-and-practical-guide-c0b07e8e7312
- Demucs#1: https://github.com/adefossez/mdx21_demucs
- Demucs#2: https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A

### Voice separation

```bash
python voice.py input.mp3 vocals.mp3
```

### Downstream ASR

```bash
python lyrics.py vocals.mp3 --genre rap
```

### Forced alignment with external lyrics

```bash
python align.py vocals.mp3 samples/rap.txt --output aligned.json
```

Use `align.py` when you already have lyrics text and only need timestamps from forced alignment.
