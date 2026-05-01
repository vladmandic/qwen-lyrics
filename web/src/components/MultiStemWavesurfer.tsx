import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import WavesurferPlayer from '@wavesurfer/react'
import type WaveSurfer from 'wavesurfer.js'
import type { GenericPlugin } from 'wavesurfer.js/dist/base-plugin.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/dist/plugins/regions.esm.js'
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.esm.js'
import SpectrogramPlugin from 'wavesurfer.js/dist/plugins/spectrogram.esm.js'
import HoverPlugin from 'wavesurfer.js/dist/plugins/hover.esm.js'
import './MultiStemWavesurfer.css'

type Stem = {
  id: string
  label: string
  file: string
  color: string
}

type StemMap<T> = Record<string, T>

type RegionRange = {
  start: number | null
  end: number | null
}

function MultiStemWavesurfer() {
  const [stems, setStems] = useState<Stem[]>([])
  const [zoom, setZoom] = useState(0)
  const [masterVolume, setMasterVolume] = useState(1)
  const [regionRange, setRegionRange] = useState<RegionRange>({ start: null, end: null })
  const [fixedRegion, setFixedRegion] = useState(false)
  const [volumes, setVolumes] = useState<StemMap<number>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const regionRef = useRef<Region | null>(null)
  const dragSelectionEnabled = useRef(false)
  const dragSelectionCleanup = useRef<() => void | null>(null)

  const playersRef = useRef<StemMap<WaveSurfer>>({})
  const spectrogramRef = useRef<HTMLDivElement | null>(null)
  const timelinePlugin = useMemo(() => {
    console.log('Initializing Timeline plugin')
    return TimelinePlugin.create({ container: '#timeline', height: 28 })
  }, [])
  const hoverPlugin = useMemo(
    () => {
      console.log('Initializing Hover plugin')
      return HoverPlugin.create({
        labelBackground: '#0f172a',
        labelColor: '#f8fafc',
        lineColor: '#f8fafc55',
        formatTimeCallback: (seconds: number) => {
          const minutes = Math.floor(seconds / 60)
          const wholeSeconds = Math.floor(seconds % 60)
          const milliseconds = Math.floor((seconds % 1) * 1000)
          return `${minutes}:${wholeSeconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`
        },
      })
    },
    [],
  )
  const regionsPlugin = useMemo(() => {
    console.log('Initializing Regions plugin')
    return RegionsPlugin.create()
  }, [])
  const firstStemId = stems[0]?.id
  const [spectrogramPlugin, setSpectrogramPlugin] = useState<GenericPlugin | null>(null)

  useEffect(() => {
    console.log('MultiStemWavesurfer mounted')
  }, [])

  const overlayPlugins = useMemo(() => {
    const plugins: GenericPlugin[] = [timelinePlugin, regionsPlugin, hoverPlugin]
    if (spectrogramPlugin) {
      plugins.push(spectrogramPlugin)
    }
    return plugins
  }, [timelinePlugin, regionsPlugin, hoverPlugin, spectrogramPlugin])
  const noPlugins = useMemo(() => [] as GenericPlugin[], [])

  useEffect(() => {
    if (spectrogramPlugin || loading || !stems.length || !spectrogramRef.current) {
      return
    }

    console.log('Creating Spectrogram plugin', {
      containerExists: !!spectrogramRef.current,
      loading,
      stemsLoaded: stems.length,
    })

    const plugin = SpectrogramPlugin.create({
      container: '#spectrogram',
      height: 100,
      labels: false,
      colorMap: 'roseus',
      splitChannels: false,
      scale: 'mel',
      frequencyMax: 12000,
      frequencyMin: 0,
      fftSamples: 512,
      useWebWorker: true,
    })

    setSpectrogramPlugin(plugin)

    return () => {
      plugin.destroy()
      setSpectrogramPlugin(null)
    }
  }, [loading, stems.length])

  useEffect(() => {
    const unsubUpdated = regionsPlugin.on('region-updated', (region: Region) => {
      setRegionRange({ start: region.start, end: region.end })

      const firstPlayer = getFirstPlayer()
      if (firstPlayer) {
        firstPlayer.setTime(region.start)
        syncAllTo(region.start, firstStemId, true)
      }
    })

    const unsubCreated = regionsPlugin.on('region-created', (region: Region) => {
      const currentRegion = regionRef.current
      if (currentRegion && currentRegion.id !== region.id) {
        currentRegion.remove()
      }

      regionRef.current = region
      setRegionRange({ start: region.start, end: region.end })

      const firstPlayer = getFirstPlayer()
      if (firstPlayer) {
        firstPlayer.setTime(region.start)
        syncAllTo(region.start, firstStemId, true)
      }

      if (fixedRegion) {
        const duration = getFirstPlayer()?.getDuration() ?? region.end
        region.setOptions({
          resize: false,
          drag: true,
          end: Math.min(region.start + 20, duration ?? region.end),
        })
      }
    })

    const unsubRemoved = regionsPlugin.on('region-removed', () => {
      regionRef.current = null
      setRegionRange({ start: null, end: null })
    })

    return () => {
      unsubUpdated()
      unsubCreated()
      unsubRemoved()
    }
  }, [regionsPlugin, fixedRegion])

  useEffect(() => {
    let cancelled = false

    async function loadStems() {
      try {
        setLoading(true)
        const response = await fetch('/stems.json')

        if (!response.ok) {
          throw new Error(`Failed to load stems.json (${response.status})`)
        }

        const loadedStems = (await response.json()) as Stem[]

        if (cancelled) return

        setStems(loadedStems)
        setVolumes(
          Object.fromEntries(loadedStems.map((stem) => [stem.id, 1])) as StemMap<number>,
        )
        setError(null)
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to load stems')
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadStems()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    Object.entries(playersRef.current).forEach(([stemId, player]) => {
      const perStemVolume = volumes[stemId] ?? 1
      player.setVolume(masterVolume * perStemVolume)
      player.setMuted(false)
    })
  }, [masterVolume, volumes])

  useEffect(() => {
    Object.values(playersRef.current).forEach((player) => {
      player.zoom(zoom)
    })
  }, [zoom])


  const allPlayers = useCallback((): WaveSurfer[] => Object.values(playersRef.current), [])

  const syncAllTo = useCallback((time: number, sourceStemId?: string, force = false) => {
    Object.entries(playersRef.current).forEach(([stemId, player]) => {
      if (stemId === sourceStemId) return

      const currentTime = player.getCurrentTime()
      if (force || Math.abs(currentTime - time) > 0.03) {
        player.setTime(time)
      }
    })
  }, [])

  const playAll = useCallback(() => {
    console.log('Play all stems')
    allPlayers().forEach((player: WaveSurfer) => {
      void player.play()
    })
  }, [allPlayers])

  const pauseAll = useCallback(() => {
    console.log('Pause all stems')
    allPlayers().forEach((player: WaveSurfer) => {
      player.pause()
    })
  }, [allPlayers])

  const stopAll = useCallback(() => {
    console.log('Stop all stems')
    allPlayers().forEach((player: WaveSurfer) => {
      player.stop()
    })
  }, [allPlayers])

  const getFirstPlayer = useCallback(() => {
    if (!firstStemId) return null
    return playersRef.current[firstStemId] ?? null
  }, [firstStemId])

  const handleReady = useCallback(
    (stemId: string, player: WaveSurfer) => {
      console.log('Player ready', { stemId, zoom, volume: masterVolume * (volumes[stemId] ?? 1) })
      playersRef.current[stemId] = player
      player.setVolume(masterVolume * (volumes[stemId] ?? 1))
      player.setMuted(false)
      player.zoom(zoom)

      const firstPlayer = getFirstPlayer()
      if (firstPlayer && stemId !== firstStemId) {
        player.setTime(firstPlayer.getCurrentTime())
      }

      if (stemId === firstStemId) {
        if (!dragSelectionEnabled.current) {
          console.log('Enabling regions drag selection on first stem')
          dragSelectionCleanup.current = regionsPlugin.enableDragSelection({
            color: 'rgba(255, 190, 11, 0.22)',
            drag: true,
            resize: !fixedRegion,
          })
          dragSelectionEnabled.current = true
        }

        if (spectrogramPlugin) {
          console.log('Spectrogram plugin available for first stem')

          player.on('spectrogram-ready' as any, () => {
            console.log('Spectrogram has finished rendering')
          })

          player.on('spectrogram-click' as any, (relativeX: number) => {
            console.log('Spectrogram clicked', { relativeX })
            player.setTime(relativeX * player.getDuration())
          })
        }
      }
    },
    [masterVolume, volumes, zoom, getFirstPlayer, firstStemId, regionsPlugin, spectrogramPlugin, fixedRegion],
  )

  const handleInteraction = useCallback(
    (player: WaveSurfer) => {
      const currentTime = player.getCurrentTime()
      console.log('Waveform interaction', { currentTime })
      syncAllTo(currentTime, firstStemId)
    },
    [firstStemId, syncAllTo],
  )

  const handleTimeupdate = useCallback(
    (player: WaveSurfer, currentTime: number) => {
      if (regionRange.start !== null && regionRange.end !== null && currentTime >= regionRange.end) {
        const target = regionRange.start
        player.setTime(target)
        syncAllTo(target, firstStemId, true)
        return
      }

      syncAllTo(currentTime, firstStemId)
    },
    [firstStemId, regionRange.start, regionRange.end, syncAllTo],
  )

  const handleFinish = useCallback(() => {
    const target = regionRange.start ?? 0
    allPlayers().forEach((player: WaveSurfer) => {
      player.setTime(target)
      void player.play()
    })
  }, [allPlayers, regionRange.start])

  useEffect(() => {
    return () => {
      dragSelectionCleanup.current?.()
    }
  }, [])

  useEffect(() => {
    const region = regionRef.current
    if (!region) {
      return
    }

    region.setOptions({
      resize: !fixedRegion,
      drag: true,
    })

    if (fixedRegion) {
      const duration = getFirstPlayer()?.getDuration() ?? region.end
      region.setOptions({
        end: Math.min(region.start + 20, duration ?? region.end),
      })
    }
  }, [fixedRegion])

  if (loading) {
    return <p className="status">Loading stems.json...</p>
  }

  if (error) {
    return <p className="status error">{error}</p>
  }

  if (!stems.length) {
    return <p className="status">No stems configured in stems.json.</p>
  }

  return (
    <div className="multi-stem-root">
      <div className="controls-group transport">
        <button type="button" onClick={playAll}>
          Play
        </button>
        <button type="button" onClick={pauseAll}>
          Pause
        </button>
        <button type="button" onClick={stopAll}>
          Stop
        </button>
      </div>

      <div className="controls-group loop-actions">
        <label className="toggle">
          <input
            type="checkbox"
            checked={fixedRegion}
            onChange={() => {
              setFixedRegion((prev) => !prev)
            }}
          />
          Fixed region (20s)
        </label>
        <span className="loop-readout">
          Region: {regionRange.start?.toFixed(2) ?? '-'} → {regionRange.end?.toFixed(2) ?? '-'}
        </span>
      </div>

      <div className="controls-group sliders">
        <label>
          Master Volume
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={masterVolume}
            onChange={(event) => {
              const value = Number(event.target.value)
              console.log('Master volume changed', { value })
              setMasterVolume(value)
            }}
          />
          <span>{masterVolume.toFixed(2)}</span>
        </label>
        <label>
          Zoom
          <input
            type="range"
            min="0"
            max="200"
            step="1"
            value={zoom}
            onChange={(event) => {
              const value = Number(event.target.value)
              console.log('Zoom changed', { value })
              setZoom(value)
            }}
          />
          <span>{zoom}</span>
        </label>
      </div>

      <div className="waveform-stack">
        <div className="waveform-overlay">
          {stems.map((stem, index) => {
            const volume = volumes[stem.id] ?? 1
            const opacity = 0.1 + 0.5 * volume

            return (
              <div
                key={stem.id}
                className="waveform-layer"
                style={{
                  zIndex: index === 0 ? stems.length : stems.length - index,
                  pointerEvents: index === 0 ? 'auto' : 'none',
                  opacity,
                }}
              >
                <WavesurferPlayer
                  key={stem.id}
                  url={stem.file}
                  height={160}
                  waveColor={stem.color}
                  progressColor={stem.color}
                  cursorWidth={1}
                  interact={index === 0}
                  normalize
                  hideScrollbar
                  minPxPerSec={20}
                  plugins={index === 0 ? overlayPlugins : noPlugins}
                  onReady={(player) => handleReady(stem.id, player)}
                  {...(index === 0
                    ? {
                        onInteraction: handleInteraction,
                        onTimeupdate: handleTimeupdate,
                        onFinish: handleFinish,
                      }
                    : {})}
                />
              </div>
            )
          })}
        </div>

        <div id="timeline" className="timeline" />
        <div id="spectrogram" ref={spectrogramRef} className="spectrogram" />
      </div>

      <div className="stems">
        {stems.map((stem) => {
          return (
            <div key={stem.id} className="stem-row">
              <div className="stem-controls">
                <div className="stem-label">
                  <span className="stem-color-dot" style={{ background: stem.color[0] }} />
                  <strong>{stem.label}</strong>
                </div>
                <div className="stem-volume-row">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={volumes[stem.id] ?? 1}
                    onChange={(event) => {
                      const value = Number(event.target.value)
                      console.log('Stem volume changed', { stemId: stem.id, value })
                      setVolumes((prev) => ({ ...prev, [stem.id]: value }))
                    }}
                  />
                  <span>{(volumes[stem.id] ?? 1).toFixed(2)}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default MultiStemWavesurfer
