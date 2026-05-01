# Audio visualization with Multiple Stems

Current: audio visualization using `wavesurfer` package based on a single user-uploaded audio track
Future: Upon user upload, audio file is immediately demucsed and each stem is visualized as a separate overlay

*How*? Trick is to to instantiate WavesurferPlayer for each stem and then stack them on top of each other. This way we can leverage the existing WavesurferPlayer component and its plugins (e.g. regions, spectrogram) without having to modify them to support multiple stems.
Player controls (play, pause, stop) and plugins (region, hover) should be linked to first stem (master) and then mirrored to the rest of the stems. This way we can ensure that all stems are always in sync and we don't have to worry about syncing them manually.

## Detailed Specs

- Core component specs: `wavesurfer-multi-step-specs.md`
- Demo implementation: `wavesurfer-demo-app.md`
