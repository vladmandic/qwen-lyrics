# Multi‑Stem Audio Visualizer (React + TypeScript + WaveSurfer) — Implemented Requirements

## Overview

This document captures the core demo requirements that have been implemented in the current `MultiStemWavesurfer.tsx` reference app.

The component is designed to visualize multiple audio stems in an overlaid waveform stack while keeping playback and timeline behavior synchronized.

The implementation covers:

- UX and interaction rules for the current demo
- Engineering architecture for the live player
- State model for implemented behavior
- Event flow for playback, seeking, and looping
- Plugin configuration for timeline, regions, and hover
- Implementation notes for what is built today

## Goals

- Visualize multiple audio stems in a single overlaid waveform display
- Provide synchronized playback, pause, and stop controls
- Offer per‑stem volume control only
- Provide global master volume and zoom controls
- Support A/B loop region selection with Set A / Set B / Clear Loop
- Use JSON configuration for stem definitions
- Render an always-on Spectrogram panel for the first stem instance
- Add waveform hover cursor labels with millisecond precision

---

## JSON Configuration

Stems are defined in a JSON file loaded at runtime.

### Example: `stems.json`

```json
[
  {
    "id": "vocals",
    "label": "Vocals",
    "file": "/stems/vocals.mp3",
    "color": "#ff4d4d"
  },
  {
    "id": "drums",
    "label": "Drums",
    "file": "/stems/drums.mp3",
    "color": "#4d94ff"
  },
  {
    "id": "bass",
    "label": "Bass",
    "file": "/stems/bass.mp3",
    "color": "#4dff4d"
  },
  {
    "id": "other",
    "label": "Other",
    "file": "/stems/other.mp3",
    "color": "#ffb84d"
  }
]
```

### Required fields

| Field   | Type   | Description                  |
|--------|--------|------------------------------|
| `id`   | string | Unique identifier for stem   |
| `label`| string | Display name                 |
| `file` | string | URL/path to audio file       |
| `color`| string | Waveform + progress color    |

The component loads this file via `fetch("/stems.json")`.

---

## UX Requirements

### 1. Transport bar

Displayed at the top and includes:

- **Play** — start playback on all stems
- **Pause** — pause all stems
- **Stop** — stop playback and reset all stems to time 0

### 2. Loop region controls

Displayed below the transport controls and includes:

- **Set A** — capture the current time from the first stem as the loop start
- **Set B** — capture the current time from the first stem as the loop end
- **Clear Loop** — remove the active loop region

Behavior when a loop region is active:

- A visible region is shown on the first waveform layer
- When playback reaches region end, the first stem jumps back to region start
- All stems are kept in sync by updating the playback time together

### 3. Timeline

- A single shared timeline is displayed below the waveform
- Only the first WaveSurfer instance attaches the Timeline plugin
- Timeline is based on the first stem’s duration and time alignment

### 4. Master volume

- Global volume slider with range `0–1`
- Multiplies all stem volumes uniformly

### 5. Zoom slider

- Global zoom slider with range `0–200`
- Zoom applies to all WaveSurfer instances simultaneously

### 6. Per‑stem controls

Each stem row currently includes:

- **Label** — the stem name
- **Volume slider** — range `0–1`, step `0.01`

### 7. Waveform display

- Multiple stem waveforms are rendered as separate overlay layers in one shared waveform view
- Each stem uses its own WaveSurfer instance with the first instance managing shared plugins
- Stem volume changes are reflected visually through layer opacity and audio gain

### 8. Hover plugin

- A waveform hover cursor is displayed with millisecond-level time labels
- The hover line and label are rendered via the Wavesurfer Hover plugin

### 9. Spectrogram plugin

- A Spectrogram panel is rendered below the waveform in a shared `#spectrogram` container
- The first WaveSurfer instance loads the Spectrogram plugin and manages rendering
- Configured with a low-contrast grayscale color map, `splitChannels: false`, and `fftSamples: 1024`
- Requires `minPxPerSec` on the WaveSurfer instance for proper rendering when the waveform is scrollable

---

## Engineering Architecture

### Component structure

- **`MultiStemWavesurfer.tsx`**:
  - Loads `stems.json`
  - Manages `stems`, `zoom`, `masterVolume`, `loopRegion`, and per-stem volumes
  - Renders transport controls, loop controls, waveform + timeline, and stem volume rows

- **WaveSurfer instances**:
  - One instance per stem rendered inside `.waveform-overlay`
  - First instance additionally attaches Timeline, Regions, and Hover plugins

### Data flow

- `stems.json` → `stems` state → `<WavesurferPlayer>` instances
- UI actions → React state updates → WaveSurfer API calls

---

## State Model

### Global state

| State          | Type            | Purpose                                 |
|----------------|-----------------|-----------------------------------------|
| `stems`        | `Stem[]`        | Loaded stem metadata                    |
| `zoom`         | `number`        | Shared zoom level                       |
| `masterVolume` | `number`        | Global volume multiplier                |
| `loopRegion`   | `{start,end}`   | A/B loop boundaries                      |

`Stem`:

```ts
type Stem = {
  id: string
  label: string
  file: string
  color: string
}
```

### Per‑stem state

| State             | Type     | Purpose                    |
|-------------------|----------|----------------------------|
| `volumes[id]`     | `number` | Per-stem volume (0–1)      |

### Derived state

- `effectiveVolume = masterVolume * volumes[stemId]`

---

## Event Flow

### Playback sync

- **Play** → `play()` on all WaveSurfer instances
- **Pause** → `pause()` on all instances
- **Stop** → `stop()` on all instances and reset time to 0

### Seeking sync

- User interaction on the first waveform updates the first instance’s current time
- The same time is applied to all other instances for synchronization

### Loop region

- `Set A` sets `loopRegion.start`
- `Set B` sets `loopRegion.end`
- `Clear Loop` removes the loop region
- On `timeupdate`, if the first instance reaches the region end, all instances jump to `loopRegion.start`

---

## Plugin Configuration

### Timeline plugin

```ts
TimelinePlugin.create({
  container: '#timeline',
  height: 28,
})
```

- Attached only to the first WaveSurfer instance
- Provides the shared timeline bar below the waveform

### Regions plugin

```ts
RegionsPlugin.create()
```

- Used to render and manage the A/B loop region on the first waveform
- The region is synchronized with React state via `region-updated`

### Hover plugin

```ts
HoverPlugin.create({
  labelBackground: '#0f172a',
  labelColor: '#f8fafc',
  lineColor: '#f8fafc55',
  formatTimeCallback: (seconds) => 'MM:SS.mmm',
})
```

- Displays a hover cursor line with millisecond-accurate time labels
- Enhances waveform interaction and scrubbing feedback

---

## Implementation Notes

- All stems are assumed to be time-aligned and loaded from `stems.json`
- WaveSurfer instances are stored in `useRef` so they can be controlled outside render
- Only the first instance manages shared plugins and interaction events
- Zoom changes are propagated to all instances via `ws.zoom(zoom)`
- Loop region state is stored in React and mirrored to the Regions plugin
- Per-stem volume controls are intentionally minimal and do not include mute/solo/collapse

---

## UI Layout Summary

1. Transport controls
2. Loop region controls
3. Shared waveform display
4. Shared timeline bar
5. Global master volume slider
6. Global zoom slider
7. Per-stem volume rows

---

## Deliverables

- `src/components/MultiStemWavesurfer.tsx` — React TypeScript component implementing this demo
- `public/stems.json` — stem configuration for demo playback
- `src/components/MultiStemWavesurfer.css` — component styling and layout

1. User clicks **Play**.  
2. Component calls `play()` on all WaveSurfer instances.  
3. All stems start playback from their current time (assumed aligned).  

Similarly for **Pause** and **Stop**:

- **Pause** → `pause()` on all instances  
- **Stop** → `stop()` on all instances (reset to 0)  

### Seeking sync

1. User clicks or drags on any waveform.  
2. WaveSurfer emits `interaction` event on that instance.  
3. Component reads `currentTime = ws.getCurrentTime()` from the source instance.  
4. Component calls `setTime(currentTime)` on all other instances.  

### Loop region (A/B markers)

1. User clicks **Set A** → `loopRegion.start = currentTime` of first instance.  
2. User clicks **Set B** → `loopRegion.end = currentTime` of first instance.  
3. Regions plugin is configured on the first instance with `[ { start, end, ... } ]`.  
4. On `timeupdate` from the first instance:
   - If `loopRegion.start` and `loopRegion.end` are set and `t >= end`, then:
     - Call `setTime(loopRegion.start)` on all instances.  

### Whole‑track loop toggle

- When loop toggle is enabled and no valid A/B loop region is active:
  - On `finish` event from any instance:
    - Call `play(0)` on all instances (restart from beginning).
- If a valid A/B region loop is active, region looping takes precedence over whole-track looping.  

---

## Plugin Configuration

### Timeline plugin (first instance only)

```js
TimelinePlugin.create({
  container: timelineRef.current,
  height: 30
})
```

- Attached only to the first WaveSurfer instance.  
- Uses the first instance’s duration as the reference timeline.  

### Regions plugin (first instance only)

```js
RegionsPlugin.create({
  regions: loopRegion.start !== null && loopRegion.end !== null
    ? [
        {
          start: loopRegion.start,
          end: loopRegion.end,
          color: "rgba(255, 204, 0, 0.2)",
          drag: true,
          resize: true
        }
      ]
    : []
})
```

- Only one region is needed (the loop region).  
- Region is draggable and resizable; updates `loopRegion` on `region-updated`.  

---

## Implementation Notes

- All stems are assumed to have identical duration and alignment.  
- WaveSurfer instances should be stored in a stable `useRef` array.  
- Only the first instance should register Timeline and Regions plugins.  
- Zoom changes must be propagated to all instances via `ws.zoom(zoomValue)`.  
- Master volume should be applied as a multiplier on top of per‑stem volume.  
- Collapsing a stem should ideally hide the waveform container but keep the instance alive; unmounting/remounting can be done but may require re‑syncing.  
- Loop region state (`loopRegion`) must be kept in React state and mirrored to the Regions plugin configuration.  

---

## UI Layout Summary

1. **Transport bar**  
2. **Loop region controls** (Set A, Set B, Clear Loop)  
3. **Timeline bar**  
4. **Master volume slider**  
5. **Zoom slider**  
6. **Stem list** (for each stem):
   - Collapse toggle  
   - Label  
   - Solo button  
   - Mute button  
   - Volume slider  
7. **Overlay waveform display** (single shared waveform area showing all stems in color)  

---

## Assumptions

- All stems share identical duration and are time‑aligned.  
- Audio files are accessible via URLs defined in `stems.json`.  
- React environment supports ES modules and `@wavesurfer/react`.  
- WaveSurfer.js v7+ is used, with Timeline and Regions plugins available as ES modules.  

---

## Deliverables

- `MultiStemWavesurfer.tsx` — React TypeScript component implementing this spec  
```
