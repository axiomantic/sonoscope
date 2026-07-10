# sonoscope

**Machine-listening QA for audio plugins — close the render → listen → decide loop without human ears.**

[![CI](https://github.com/axiomantic/sonoscope/actions/workflows/ci.yml/badge.svg)](https://github.com/axiomantic/sonoscope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

sonoscope closes the loop `(plugin, stimulus, param-set) → wav → versioned analysis JSON`,
pairing deterministic ground-truth features with probe-gated, advisory perception. It gives
an LLM (or a developer) **machine-readable ground truth** about what a plugin actually renders
— features, integrity flags, PASS/RED tripwires, and noise-floor-gated iterate verdicts, plus
an optional advisory natural-language description — so an agent can develop, render, listen,
and decide **without human ears**.

<!-- Demo: a terminal recording of `sonoscope analyze` on a real plugin belongs here.
     Record with VHS (.tape) or asciinema + svg-term-cli and drop the SVG/GIF in assets/. -->

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). sonoscope is not published to PyPI;
install from source:

```bash
git clone https://github.com/axiomantic/sonoscope.git
cd sonoscope
uv sync                    # install pinned core dependencies
uv run sonoscope doctor    # verify your environment is healthy
```

The **`perception`** extra (optional, advisory) pulls in `torch` + `transformers` and downloads
the ~16 GB Qwen2-Audio model on first use. Skip it unless you want natural-language descriptions:

```bash
uv run --extra perception sonoscope analyze --perception --plugin "<plugin>.vst3" --spec <spec>.json
```

## Quickstart

Every command emits one JSON object on stdout and a typed exit code
(`0` ok/finding · `1` usage · `2` input · `3` render · `4` analysis · `5` environment). A plugin
defect is reported as a *finding* with exit `0`, not a crash — so CI can branch on the JSON, not
on process failure.

```bash
uv run sonoscope doctor                                   # is my environment healthy?
uv run sonoscope analyze --plugin "<plugin>.vst3" --spec <spec>.json   # what did it render?
uv run sonoscope analyze --wav path/to/render.wav         # analyze an existing wav (honest provenance)
uv run sonoscope iterate --plugin "<plugin>.vst3" \
  --baseline <a>.json --candidate <b>.json \
  --metric deterministic.summary.spectral_centroid_hz --direction decrease  # did my change land?
```

### Example: analyze a render

A `--spec` bundles the stimulus and the param-set. Point `analyze` at a built plugin and a spec:

```bash
uv run sonoscope analyze \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --spec specs/example.json
```

It renders the audio and prints a versioned analysis report (trimmed):

```json
{
  "schema_version": "1.4.0",
  "deterministic": {
    "summary": {
      "rms_dbfs": -22.03,
      "peak_dbfs": -13.69,
      "spectral_centroid_hz": 2809.06,
      "spectral_flatness": 0.00051,
      "onset_count": 1
    },
    "integrity": { "is_silent": false, "has_nan": false, "clip_count": 0 }
  },
  "tripwires": {
    "expected_audio": true,
    "results": [
      { "id": "silent-output", "verdict": "PASS", "detail": "rms_dbfs -22.0 > -80.0 dBFS" },
      { "id": "nan-inf", "verdict": "PASS" },
      { "id": "clipping", "verdict": "PASS", "detail": "clip_fraction 0.0" }
    ],
    "overall": "PASS"
  }
}
```

### Example: analyze an existing wav

`analyze --wav` analyzes a standalone file at its native rate, then resamples the analyzed slice
to 48 kHz — and records exactly what it did in an `input_provenance` block, so a resampled-from-44.1k
signal is never reported as faked-pristine. Output is a JSON array, one entry per chunk (trimmed):

```json
[
  {
    "schema_version": "1.4.0",
    "kind": "wav-chunk-analysis",
    "input_provenance": {
      "original_sample_rate": 44100,
      "n_channels": 1,
      "source_subtype": "PCM_16",
      "resample_res_type": "soxr_hq",
      "chunk_index": 0,
      "n_chunks": 1
    },
    "deterministic": { "summary": { "rms_dbfs": -12.73, "spectral_centroid_hz": 54.55 } }
  }
]
```

`doctor` gives you a fast, gated readiness check before any of that:

```
sonoscope doctor:
  [OK  ] pins: 5 pinned dependencies match
  [OK  ] lockfile: uv.lock in sync
  [OK  ] surge_xt: Surge XT install + factory content verified
  [OK  ] backend: backend pedalboard-vst3 v0.9.23 loaded
  [OK  ] perception: perception available: Qwen2-Audio-7B-Instruct (transformers)
  => OK
```

## Features

- **Plugin hosting**: renders VST3 via [pedalboard](https://github.com/spotify/pedalboard) and CLAP via a bundled C host — real audio, not simulation.
- **Honest-provenance WAV analysis**: `analyze --wav` analyzes existing files at native rate, resamples the slice to 48 kHz, and records source rate/subtype/resampler in an `input_provenance` block — never faked-pristine.
- **Spec-driven matrix**: one JSON `--spec` pins the stimulus, param-set, and patch class; swap specs to sweep params and stimuli reproducibly.
- **Versioned analysis JSON**: a schema-versioned (`1.4.0`), extra-forbidding pydantic report — stable enough to diff, regression-test, and feed to an LLM.
- **Deterministic feature ground truth**: [librosa](https://librosa.org/)-backed RMS/peak, spectral centroid/flatness/rolloff, onsets, tempo, and MFCCs, plus NaN/Inf/clip/DC integrity flags.
- **PASS/RED tripwires**: silent-output, NaN/Inf, denormal, and clipping checks that turn "did it break?" into a machine-readable verdict.
- **Noise-floor-gated iterate**: compare a baseline vs. candidate on one metric (`iterate`) or on descriptor terms (`iterate-descriptors`), gated against measured nondeterminism floors — no chasing noise.
- **Optional Qwen2-Audio perception**: opt-in, clearly-labelled *advisory* natural-language descriptions, never treated as ground truth.

## Documentation

**[docs/using-sonoscope.md](docs/using-sonoscope.md)** is the full guide — an LLM- and
developer-oriented walkthrough with a mental model, quickstart, and many worked examples
(every command and output captured from real runs against Surge XT and the Qwen2-Audio model).
**Start there.**

**[docs/music-vocabulary.md](docs/music-vocabulary.md)** is the controlled descriptor
vocabulary and grounding contract — the taxonomy sonoscope uses to describe audio, tagging
each term by how much is measured versus opinion.

Beyond the commands above, the CLI also exposes `render`, `analyze-midi`, `determinism`,
`probe`, `schema`, and `corpus`. Run `uv run sonoscope <command> --help` for details.
