# Waveworm Audio Player

Create a demo project using React, TypeScript, and Vite that showcases a multi-stem WaveSurfer player.
The demo is built around `src/components/MultiStemWavesurfer.tsx` and uses `stems.json` to load stem metadata at runtime.

## Implemented demo requirements

- App scaffolded with Vite, React, and TypeScript
- Uses `@wavesurfer/react` + `wavesurfer.js`
- Loads stem definitions from `public/stems.json`
- Renders multiple stems as overlaid waveform layers in a single shared waveform display
- Uses a single shared timeline attached to the first WaveSurfer instance
- Includes Play / Pause / Stop transport controls
- Includes global Master Volume and Zoom sliders
- Includes per-stem volume sliders only
- Supports A/B loop region selection with Set A, Set B, and Clear Loop
- Implements millisecond-accurate hover cursor labels over the waveform
- Includes a Spectrogram panel rendered by the first WaveSurfer instance
- Uses a dark grayscale UI theme

## Notes

- The demo intentionally keeps controls minimal and focused on core playback, loop, and volume interactions.
- The app is meant as a reference implementation rather than a full DAW feature set.
