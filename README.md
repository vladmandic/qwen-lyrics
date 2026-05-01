# Lyrics

Set of tools for lyrics extraction from audio files using
- `qwen3-asr` package and finetuned model plus forced aligner,
- `google-genai` google-ai client library used for gemini model access
- `demucs` package for voice separation

## Reference

- [demucs repo](https://github.com/facebookresearch/demucs)
- [gemini docs](https://ai.google.dev/gemini-api/docs/)
- [qwen3-asr repo](https://github.com/QwenLM/Qwen3-ASR)
- [qwen3-asr finetuning docs](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- [genius.com api](https://docs.genius.com/#/getting-started-h1)
- [scraping genius.com](https://medium.com/@rachit.lsoni/scraping-song-lyrics-a-fun-and-practical-guide-c0b07e8e7312)
- [demucs leaderboard project](https://github.com/adefossez/mdx21_demucs)
- [demucs leaderboard project](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A)

## Examples

demucs-audio-separation
> python -m cli.split --compile --save --format mp3 samples/pop.mp3  

extract-lyrics
> python -m cli.lyrics --genre pop --output samples/pop-qwen.json samples/pop-vocals.mp3  
> python -m cli.google --genre pop --output samples/pop-gemini-flash.json samples/pop-vocals.mp3  

evaluate-lyrics
> python -m cli.metrics samples/pop-genius.txt samples/pop-qwen.json  
> python -m cli.metrics samples/pop-genius.txt samples/pop-gemini-flash.json  

align-lyrics
> python -m cli.align samples/pop-vocals.mp3 samples/pop-qwen.json --output samples/pop-qwen-aligned.json  
> python -m cli.align samples/pop-vocals.mp3 samples/pop-gemini-flash.json --output samples/pop-gemini-flash-aligned.json  

## Extra

run lyrics extraction with all possible params
> cli/tune.py  

rank results by metrics scores
> cli/rank.py  
