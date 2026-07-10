# Using sonoscope (for LLMs & developers)

This guide teaches an LLM agent (or a developer) how to use `sonoscope` to
**build, debug, and regression-test audio plugins without human ears.** Every
command, output snippet, and number below was captured from a real run on a
provisioned machine (Surge XT + a test plugin under test + the
Qwen2-Audio model present). Nothing here is invented.

> Run the CLI with `uv run sonoscope ...`. For the perception (Qwen) path use
> `uv run --extra perception sonoscope ...`.

---

## 1. The mental model

sonoscope answers one question deterministically: **"what did this plugin
actually produce?"** It closes the loop

```
(plugin, stimulus, param-set)  ->  wav  ->  versioned analysis JSON
```

and hands you back **machine-readable ground truth** plus **PASS/RED verdicts**
and **exit codes** you can branch on. There are four things in the output, in
strict order of trust:

1. **Deterministic features (GROUND TRUTH).** Librosa-computed numbers —
   `rms_dbfs`, `spectral_centroid_hz`, `spectral_flatness`, MFCCs, onsets, etc.
   Always on. This layer drives everything else. Trust it.
2. **Integrity flags + tripwires (VERDICTS).** Booleans (`is_silent`, `has_nan`,
   `clip_count`, …) rolled up into named tripwires (`silent-output`, `nan-inf`,
   `denormal`, `clipping`) with a `PASS`/`RED` verdict and an `overall` roll-up.
   This is your automated gate.
3. **Iterate significance (DECISIONS).** A/B two renders on one feature and get a
   `PASS`/`FAIL`/`INCONCLUSIVE` verdict, gated by the plugin's own measured noise
   floor so a change smaller than the plugin's run-to-run jitter can never be a
   false pass.
4. **Perception (ADVISORY — NEVER ground truth).** An optional Qwen2-Audio
   natural-language description of what the audio "sounds like." It is
   structurally separated, carries a disclaimer, and **can never fail the run**.
   Use it as a hint, never as a gate.

**Why this is LLM-friendly:** every command emits a single JSON object on stdout
and sets a **typed process exit code**. You never parse prose to make a decision —
you read `tripwires.overall`, a `verdict` field, or `$?`.

**The golden rule:** *deterministic layer + tripwires are truth; perception is
advisory.* If they ever disagree, the numbers win.

---

## 2. Quickstart

```bash
# 1. Is my environment healthy? (pins, lockfile, Surge, backend, model)
uv run sonoscope doctor            # human report -> stderr, exit 0 if OK

# 2. Does my plugin produce correct sound?
uv run sonoscope analyze \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --spec specs/surge_lowpass_open.json          # JSON report -> stdout

# 3. Did my change move the sound the way I intended?
uv run sonoscope iterate \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --baseline specs/surge_lowpass_open.json \
  --candidate specs/surge_lowpass_closed.json \
  --metric deterministic.summary.spectral_centroid_hz \
  --direction decrease                          # iterate-delta -> stdout
```

Every command writes **one JSON object to stdout** (except `doctor`'s
human-readable report, which goes to **stderr**; add `--json` for the machine
copy on stdout). Errors are emitted as a `fatal-error` JSON object on stdout and
signalled by the exit code (§10).

---

## 3. A spec is your input contract

A spec is a small versioned JSON file: which stimulus, which patch class, and
which plugin parameters to set. Parameters are addressed **by name** (names come
from runtime plugin introspection — never hardcode them). Example
(`specs/surge_lowpass_open.json`):

```json
{
  "spec_version": "1.0.0",
  "stimulus": { "kind": "midi", "ref": "corpus/midi/c3_sustain_2s.mid" },
  "patch_class": "noisy",
  "params": { "by_name": { "a_filter_1_type": 0.0303, "a_filter_1_cutoff": 0.9 } }
}
```

`patch_class` is either `noise_free` (bit-identical renders expected) or `noisy`
(the plugin legitimately jitters run-to-run, e.g. noise oscillators). It selects
which determinism-floor cache key is used.

---

## 4. Worked examples

Each example gives the **goal**, the **exact command**, a **real output slice**,
and the **interpretation + next action**.

### 4.1 "Is my environment healthy?" — `doctor`

```bash
uv run sonoscope doctor
```

Real output (to **stderr**):

```
sonoscope doctor:
  [OK  ] pins: 5 pinned dependencies match
  [OK  ] lockfile: uv.lock in sync
  [OK  ] surge_xt: Surge XT install + factory content verified
  [OK  ] backend: backend pedalboard-vst3 v0.9.23 loaded
  [OK  ] perception: perception available: Qwen2-Audio-7B-Instruct (transformers)
  [WARN] latency:deterministic_feature_extraction: 1.718s (target 0.500s)
  => OK
```

Exit code: **0**. For the machine-readable copy, add `--json` (prints to stdout):

```bash
uv run sonoscope doctor --json
```

```json
{ "kind": "doctor", "ok": true,
  "checks": [ {"name":"pins","severity":"ok","detail":"5 pinned dependencies match"}, ... ],
  "latencies": [ {"metric":"deterministic_feature_extraction","measured_s":1.31,"target_s":0.5,"over_target":true} ] }
```

**Interpretation:** `ok:true` -> exit 0. A latency `over_target` is a **soft
warning only** (it does NOT fail the run). An `error`-severity check would set
`ok:false` and exit **5** (environment). **Next action:** if `ok:false`, stop and
fix the environment before trusting any analysis.

### 4.2 "How do I read the results?" — `schema`

```bash
uv run sonoscope schema --kind analysis          # also: iterate-delta, determinism-floors, fatal-error
```

Emits the draft-2020-12 JSON Schema for the report kind. Top-level required keys
you will key on:

| kind | top-level required |
|------|--------------------|
| `analysis` | `input`, `render`, `deterministic`, `tripwires`, `perception` (+ `errors[]`) |
| `iterate-delta` | `baseline`, `candidate`, `expectation`, `delta`, `verdict` |
| `determinism-floors` | `binary_sha256`, `patch_class`, `floors`, `is_bit_identical`, … |
| `fatal-error` | `error` (`.code`, `.message`, `.component`) |

**Next action:** validate reports against this schema in CI; the models forbid
unknown keys, so any drift is a hard failure.

### 4.3 "What can I test with?" — `corpus list` / `corpus verify`

```bash
uv run sonoscope corpus list       # names, paths, kinds, sha256 of pinned stimuli
uv run sonoscope corpus verify     # recompute + compare every hash
```

`list` returns 7 pinned items (MIDI: `c3_sustain`, `phrase_4note`; signals:
`impulse`, `sweep`, `pink_noise`, `tone`, `silence`). `verify` returns
`{"kind":"corpus","action":"verify","ok":true, ...}` and exit **0**. A drifted or
missing item is a hard **INPUT** error (exit **2**) — pins are law.

### 4.4 "Does my plugin produce sound / is it broken?" — `analyze`

**Healthy case (Surge, filter open):**

```bash
uv run sonoscope analyze \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --spec specs/surge_lowpass_open.json
```

Real slice:

```json
"tripwires": {
  "expected_audio": true,
  "results": [
    {"id":"silent-output","verdict":"PASS","detail":"rms_dbfs -22.0 > -80.0 dBFS"},
    {"id":"nan-inf","verdict":"PASS"},
    {"id":"denormal","verdict":"PASS"},
    {"id":"clipping","verdict":"PASS","detail":"clip_fraction 0.0"}
  ],
  "overall": "PASS"
}
"deterministic": {
  "summary": {"rms_dbfs": -22.03, "peak_dbfs": -13.69, "spectral_centroid_hz": 2809.06,
              "spectral_flatness": 0.00051, "onset_count": 1, ...},
  "integrity": {"is_silent": false, "has_nan": false, "clip_count": 0, ...}
}
```

Exit code: **0**. **The LLM decision rule:** `tripwires.overall == "PASS"` AND
exit `0` -> the render is structurally sound. Read
`deterministic.summary.rms_dbfs` (−22 dBFS here) to confirm a sane level.

**Broken case (a test plugin under test — a real silent-output bug):**

```bash
uv run sonoscope analyze \
  --plugin "/path/to/plugin.vst3" \
  --spec specs/dogfood_note.json
```

Real slice:

```json
"tripwires": {
  "results": [
    {"id":"silent-output","verdict":"RED",
     "detail":"rms_dbfs -240.0 <= -80.0 dBFS (audio expected, output silent)"},
    ...
  ],
  "overall": "RED"
}
"integrity": { "is_silent": true }   // rms_dbfs -240.0
```

Exit code: **0**. **This is the crucial LLM pattern:** a plugin bug is a
**FINDING, not a crash.** The tool renders, measures, reports `overall:"RED"` and
`is_silent:true`, and **exits 0** — because sonoscope ran successfully and told
the truth. **Branch on the report, not the exit code, for plugin verdicts:**

```bash
overall=$(uv run sonoscope analyze --plugin "$P" --spec "$S" | jq -r '.tripwires.overall')
[ "$overall" = "PASS" ] || echo "plugin defect detected -> go fix the DSP"
```

### 4.4b "I have a wav file, not a plugin" — `analyze --wav`

Everything above renders a plugin first. `analyze --wav` skips the render and
analyzes an **existing wav on disk** at its **native rate**: it loads the file,
resolves an optional native-unit slice (`--offset`/`--length`/`--unit`, unit is
`samples` or `seconds`), tiles the region into chunks (one chunk by default;
`--max-chunk-seconds` forces auto-chunking above that native-seconds
threshold), and resamples **each chunk independently** to 48 kHz (`soxr_hq`)
before running the same deterministic feature/descriptor pipeline `analyze
--plugin` uses.

```bash
uv run sonoscope analyze --wav samples/pad.wav
```

Real output (a mono 44.1 kHz PCM_16 file, whole-file, no slice) — the report is
**always a JSON array**, one entry per chunk:

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
      "soxr_version": "1.1.0",
      "analyzed_window": {
        "native_offset_samples": 0,
        "native_length_samples": 96000,
        "native_sample_rate": 44100,
        "analyzed_samples_48k": 104490
      },
      "max_chunk_seconds": 600.0,
      "chunk_index": 0,
      "n_chunks": 1
    },
    "deterministic": { "summary": { "rms_dbfs": -12.73, "spectral_centroid_hz": 54.55, ... }, "integrity": { "is_silent": false, ... } },
    "descriptors": { "measured": {...}, "hybrid": {...}, "advisory": {...}, "summary": [...], "library": {...} },
    "descriptor_gate": null
  }
]
```

`input_provenance` is **honest, never faked-pristine**: it always names the
file's true native sample rate/subtype and the real resample that happened
(`resample_res_type`/`soxr_version` are `null` only for a genuine 48 kHz
no-op input). `chunk_index`/`n_chunks` tell you which slice of a possibly
multi-chunk array you're looking at even with a single element.

**Slicing/chunking (`--wav`-only flags; rejected as `INPUT_WAV_FLAG_CONFLICT`/
`INPUT_ANALYZE_FLAG_CONFLICT` if mixed with the wrong source):**

```bash
uv run sonoscope analyze --wav long_take.wav \
  --offset 2 --length 5 --unit seconds \
  --max-chunk-seconds 1.0
```

`--offset` enables the slice (native units, half-open `[offset, offset+length)`,
clamped to EOF); `--length` defaults to "to end of file"; `--unit` defaults to
`samples`. `--max-chunk-seconds` overrides the frozen 600 s auto-chunk
threshold — a 5 s region with `--max-chunk-seconds 1.0` above tiles into
independent ~1 s chunks, each with its own `chunk_index`/`descriptors`.

**Gating with `--expect-descriptors`/`--fail-on-red`** works exactly like
`analyze --plugin` (§4.13), but the aggregate verdict is **cross-chunk**:
`{"verdict":"GREEN"|"RED", "red_chunks":[...], "reasons":[...]}` on stderr,
`RED` iff **any** chunk's own gate is `RED` (richer than the plugin path's
`{verdict, reasons}` — each reason is prefixed `"chunk[i] "` for attribution).
Real RED (the ramp above does not emit `bright`):

```bash
uv run sonoscope analyze --wav pad.wav \
  --expect-descriptors expect.json --fail-on-red
```

`expect.json`: `{"expect_present": ["bright"]}`. stderr:

```json
{"verdict":"RED","red_chunks":[0],"reasons":["chunk[0] DESC_MISSING: bright"]}
```

Exit code **4** (`ANALYSIS`) because of `--fail-on-red`; the stdout array still
prints in full (report-then-gate, never a swallowed report), and chunk 0's own
`descriptor_gate` in that array carries `{"verdict":"RED","reasons":["DESC_MISSING: bright"],"spec_sha256":"..."}`.

### 4.5 "Did my change make it brighter/darker?" — `iterate`

Compare a baseline render vs a candidate render on **one feature**, asserting a
**direction**. Here: closing the lowpass should **decrease** spectral centroid.

```bash
uv run sonoscope iterate \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --baseline specs/surge_lowpass_open.json \
  --candidate specs/surge_lowpass_closed.json \
  --metric deterministic.summary.spectral_centroid_hz \
  --direction decrease
```

Real output:

```json
"delta": {
  "metric": "deterministic.summary.spectral_centroid_hz",
  "baseline_value": 2808.82, "candidate_value": 180.70,
  "abs_delta": -2628.12,
  "measured_floor": 2.045, "noise_threshold_multiplier": 3.0, "noise_threshold": 6.14,
  "significant": true, "matches_expectation": true
},
"verdict": "PASS"
```

Exit code: **0**. **Interpretation:** the centroid fell from 2809 Hz to 181 Hz.
The change (2628 Hz) is hundreds of times the `noise_threshold` (6.14 Hz =
`3 × measured_floor`), so `significant:true`; it moved in the asserted direction,
so `matches_expectation:true`; verdict `PASS`.

**The significance gate (why a tiny change is INCONCLUSIVE, not a false pass).**
Run the same metric with an identical candidate (no real change):

```bash
uv run sonoscope iterate ... --baseline specs/surge_lowpass_open.json \
  --candidate specs/surge_lowpass_open.json \
  --metric deterministic.summary.spectral_centroid_hz --direction change
```

```json
"delta": { "abs_delta": 0.097, "noise_threshold": 6.14, "significant": false, "matches_expectation": false },
"verdict": "INCONCLUSIVE"
```

The 0.097 Hz wobble is below the 6.14 Hz noise threshold, so sonoscope refuses to
call it a change: **`INCONCLUSIVE` is an honest third outcome, never a green.**
This is the anti-false-positive backbone: a "movement" direction
(`increase`/`decrease`/`change`) only ever `PASS`es on a **supra-floor** change.

### 4.6 "My refactor shouldn't change the sound" — the regression assertion

The **inverted** assertion — "this change must NOT move the metric" — is
`--direction stable`: it `PASS`es when the metric stayed **within** the noise
floor and `FAIL`s when it moved beyond it. Use it to prove a refactor is
sound-preserving. Here we render the *same* spec as both baseline and candidate
(the honest floor case: no real change) and assert `stable`:

```bash
uv run sonoscope iterate \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --baseline specs/surge_lowpass_open.json \
  --candidate specs/surge_lowpass_open.json \
  --metric deterministic.summary.spectral_centroid_hz \
  --direction stable
```

Real output:

```json
"expectation": { "metric": "deterministic.summary.spectral_centroid_hz",
                 "direction": "stable", "min_effect": null },
"delta": {
  "metric": "deterministic.summary.spectral_centroid_hz",
  "baseline_value": 2808.80, "candidate_value": 2808.57,
  "abs_delta": -0.22,
  "measured_floor": 1.402, "noise_threshold_multiplier": 3.0, "noise_threshold": 4.21,
  "significant": false, "matches_expectation": true
},
"verdict": "PASS"
```

Exit code: **0**. **Interpretation:** the centroid moved only 0.22 Hz between the
two renders — **below** the 4.21 Hz `noise_threshold` (`3 × measured_floor`), so
`significant:false`. For `stable`, staying sub-floor is exactly the asserted
outcome, so `matches_expectation:true` and verdict `PASS`. The regression held.

**The inverse (FAIL).** Point `--candidate` at a spec that genuinely moves the
metric — e.g. `specs/surge_lowpass_closed.json`, which drops the centroid by
~2628 Hz (§4.5). That is hundreds of times the floor, so `significant:true`; the
metric moved when `stable` asserted it must not, so `matches_expectation:false`
and verdict **`FAIL`**. `stable` is the mirror image of the movement directions:
it fails on a supra-floor change and passes only when the sound is preserved.

### 4.7 "How deterministic/noisy is my plugin?" — `determinism`

```bash
uv run sonoscope determinism \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --spec specs/surge_lowpass_open.json --repeats 5
```

Renders N times and derives a **per-feature noise floor** (default method
`range`), caching it keyed on `(binary_sha256, patch_class)`. Real slice:

```json
"patch_class": "noisy", "repeats": 5, "is_bit_identical": false,
"floors": {
  "deterministic.summary.rms_dbfs":            {"floor": 0.00123, "unit": "dbfs",  "method": "range", "repeats": 5},
  "deterministic.summary.spectral_centroid_hz":{"floor": 1.402,   "unit": "hz",    "method": "range", "repeats": 5},
  "deterministic.summary.spectral_flatness":   {"floor": 1.2e-05, "unit": "ratio", "method": "range", "repeats": 5}
  /* ... rms, peak, crest, dc_offset, bandwidth, rolloff, zcr, onsets, 13x mfcc_mean, 13x mfcc_std ... */
}
```

Exit code: **0**. **How floors feed `iterate`:** `iterate` reads this cache
(measuring + caching it on first use if absent) and multiplies the relevant
feature's floor by `noise_threshold_multiplier` (default 3) to get the
`noise_threshold` that gates significance in §4.5. A larger floor -> a larger
change is required before `iterate` will call it real. `--repeats` must be ≥ 2.

### 4.8 "What does my plugin SOUND like?" — `analyze --perception` (advisory)

```bash
uv run --extra perception sonoscope analyze --perception \
  --plugin "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3" \
  --spec specs/surge_lowpass_open.json
```

Real `perception` block:

```json
"perception": {
  "status": "ok", "grounding": "advisory-freetext",
  "adapter": {"id":"qwen-local","model":"Qwen2-Audio-7B-Instruct","runtime":"transformers","quant":"none",
              "model_sha256":"b6cc0530..."},
  "description": "The sound has a bright quality with a medium pitch. It's not noisy but it's also not particularly clear. The volume of the sound is moderate.",
  "grounding_map": {},
  "disclaimer": "Advisory only. Not ground truth. May be inaccurate or hallucinated."
}
```

Exit code: **0**; `tripwires.overall` remains `PASS` (still the ground truth).
Without `--perception`, the block degrades to `{"status":"disabled","grounding":"none"}`.

> **STRESS: perception is advisory, never ground truth.** The `description` is a
> hint for a human/LLM reader — it carries a `disclaimer` and is structurally
> separated from `deterministic`/`tripwires`. A perception crash or timeout
> degrades to `status:"error"` (or `"unavailable"`), the loop continues, and the
> exit code stays **0** — perception can never fail a run or override the numbers.
> **Never gate a decision on `perception`; gate on `tripwires` + `deterministic`.**

### 4.9 "Is my perception adapter actually discriminating?" — `probe`

`probe` is a self-test for the **perception** layer: it runs the Qwen adapter
over a pinned A/B fixture set (10 wavs in `corpus/qwen_probe/`, 5
contrastive pairs — `cutoff`, `noisiness`, `pitch`, `timbre`, `loudness`) and
scores how many pairs the model orders correctly. It needs the model and the
perception extra:

```bash
uv run --extra perception sonoscope probe
```

Real output (JSON on **stdout**; model-load + inference logs go to **stderr**):

```json
{ "kind": "probe", "status": "ok", "verdict": "PASS",
  "n_correct": 4, "m_total": 5, "ratio": 0.8,
  "pairs": [ {"key":"cutoff","correct":true}, {"key":"noisiness","correct":true},
             {"key":"pitch","correct":true}, {"key":"timbre","correct":true},
             {"key":"loudness","correct":false} ] }
```

Exit code: **0**. **Interpretation:** the adapter ordered 4 of 5 axes correctly
(`ratio 0.8`) — it reliably grounds cutoff/pitch/timbre/noisiness but misses
loudness here, consistent with the guide's rule that perception's one reliably-
grounded axis is *brightness ↔ `spectral_centroid_hz`* while loudness is not
(§7). The fixtures ship in-repo, so `probe` works out of the box.

If the fixture wavs are absent (or `--fixtures <dir>` points nowhere), `probe`
fails cleanly with a **typed** error rather than a stack trace:

```json
{ "kind": "fatal-error",
  "error": { "code": "PROBE_FIXTURES_NOT_FOUND", "component": "perception",
             "message": "probe fixtures not found under <dir>: 10 fixture wav(s) missing. ..." } }
```

Exit code: **2** (INPUT).

### 4.10 The full render -> analyze -> iterate LOOP (how an LLM scripts it)

The develop -> render -> listen -> decide loop, expressed as an agent would
script it (edit a spec param, re-render+analyze, compare vs baseline, read the
verdict, decide):

```bash
set -e
PLUGIN="/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"

# (a) make the change: candidate spec with the filter closed (cutoff 0.9 -> 0.1)
#     -- edit specs/*.json by hand or with jq; params are by_name.

# (b) sanity-check the candidate renders & is structurally sound
overall=$(uv run sonoscope analyze --plugin "$PLUGIN" --spec specs/surge_lowpass_closed.json \
          | jq -r '.tripwires.overall')
[ "$overall" = "PASS" ] || { echo "candidate render is broken ($overall)"; exit 1; }

# (c) assert the intended effect vs the baseline, gated by the noise floor
verdict=$(uv run sonoscope iterate --plugin "$PLUGIN" \
            --baseline specs/surge_lowpass_open.json \
            --candidate specs/surge_lowpass_closed.json \
            --metric deterministic.summary.spectral_centroid_hz \
            --direction decrease | jq -r '.verdict')

# (d) decide
case "$verdict" in
  PASS)         echo "change works as intended" ;;
  FAIL)         echo "change moved the WRONG way -> revert/rethink" ;;
  INCONCLUSIVE) echo "change too small to distinguish from noise -> no real effect" ;;
esac
```

`render` on its own (for inspecting a wav) prints a render summary and, with
`--out PATH`, writes the wav:

```bash
uv run sonoscope render --plugin "$PLUGIN" --spec specs/surge_lowpass_open.json --out /tmp/open.wav
# {"kind":"render","wav_path":"/tmp/open.wav","backend":"pedalboard-vst3",
#  "render_meta":{"sample_rate_hz":48000,"channels":2,"duration_s":2.0,"render_wall_ms":10,"wav_sha256":"a0e2da58...",...}}
```

### 4.11 "Did my plugin emit the RIGHT NOTES?" — `analyze-midi`

`analyze-midi` is the **MIDI-domain sibling** of `analyze`. Where `analyze` asks
"what audio did this plugin PLAY?", `analyze-midi` asks **"what MIDI event stream
did this note-effect plugin (or `.mid` file) produce?"** — for sequencers,
arpeggiators, and MIDI-effect plugins that emit notes rather than sound. It takes
one of two sources (exactly one; both/neither is a typed `InputError`, exit 2):

- `--plugin <.clap> --spec <capture.json>` — capture a **CLAP note-effect**
  plugin, driven through the C host over the spec's transport window.
- `--file <.mid> --sample-rate <Hz>` — load a **standalone `.mid`** file.

It runs the deterministic **MIDI tripwires** — the **stuck-note firewall**
(every `note_on` must get a `note_off`) plus, when `--expected` is given, an
**expected-vs-actual diff** against a golden — and emits a versioned
`midi-analysis` report with the same trust model as audio: a `verdict`
(`PASS`/`RED`), a structured `reasons[]` array, and a typed exit code.

**The command surface** (`analyze-midi --help`, epilog):

```
Examples:
  # Capture a CLAP note-effect plugin against a golden and gate CI on RED:
  sonoscope analyze-midi --plugin 'ReferenceSequencer.clap' --spec capture.json \
    --expected golden.json --fail-on-red
  # Analyze a standalone .mid, windowing beats [1, 3):
  sonoscope analyze-midi --file phrase.mid --sample-rate 48000 \
    --offset 1 --length 2 --unit beats
```

#### File-source analysis (no plugin needed)

```bash
uv run sonoscope analyze-midi --file corpus/midi/phrase_4note.mid --sample-rate 48000
```

Real output (the `midi` block; events trimmed with `…`):

```json
"input": { "source": "file",
  "file": {"path":"corpus/midi/phrase_4note.mid","file_sha256":"e564909e..."},
  "transport": {"sample_rate":48000,"block_size":0,"tempo_bpm":120.0,
                "duration_beats":4.0,"tsig_num":4,"tsig_den":4,"playing":true} }
"midi": {
  "capture_meta": {"sample_rate":48000,"duration_samples":96000,"ppq":960,
                   "tempo_bpm":120.0,"source":"file","timing_tolerance_samples":1, ...},
  "events": [
    {"t_samples":0,     "t_ticks":0,    "type":"note_on",  "channel":0, "note":48, "velocity":100},
    {"t_samples":24000, "t_ticks":960,  "type":"note_off", "channel":0, "note":48, "velocity":0},
    {"t_samples":24000, "t_ticks":960,  "type":"note_on",  "channel":0, "note":52, "velocity":100},
    … /* 55@48000, 60@72000, off60@96000 — a 4-note C3→E3→G3→C4 phrase, 1 beat each */
  ],
  "expected_vs_actual": null,
  "integrity": {"every_note_on_has_off": true, "stuck_notes": [], "dangling_offs": []},
  "verdict": "PASS", "reasons": []
}
```

Exit code: **0**. **The `midi` block, top to bottom:** `capture_meta` is the
timing frame (`sample_rate`, `duration_samples`, `ppq`, `tempo_bpm`,
`timing_tolerance_samples`); `events[]` is the decoded stream, each event carrying
BOTH a sample-domain (`t_samples`) and a tick-domain (`t_ticks`) timestamp;
`integrity` is the stuck-note firewall's finding (`every_note_on_has_off` +
`stuck_notes[]`/`dangling_offs[]`); `verdict`+`reasons[]` are the roll-up. Here
all four notes open and close cleanly, so the firewall is green and `verdict` is
`PASS`. **File-source defaults:** `--sample-rate` is **required** (it derives the
sample timing axis) and the note-on-velocity-0 policy defaults to
`offvel0=normalize` (a `note_on` with velocity 0 is folded to a `note_off`, the
running-status convention most `.mid` files use); `--plugin` capture instead
defaults to `offvel0=red` (it treats a velocity-0 note-on as a defect). Override
either with `--offvel0 {red,normalize}`.

#### Regression diff against a golden — `--expected`

Point `--expected` at a golden event list (JSON) to diff the actual stream
against it. A **self-consistent** golden (here the capture's own events) is the
regression PASS:

```json
"expected_vs_actual": {"matched": 8, "missing": [], "extra": [], "mistimed": [], "wrong_field": []},
"verdict": "PASS", "reasons": []
```

The **RED** case — the same `.mid` diffed against a *different* golden
(`specs/refseq_demo_correct1.json`, an 8-event note-36/72 pattern that the C3
phrase does not match):

```bash
uv run sonoscope analyze-midi --file corpus/midi/phrase_4note.mid --sample-rate 48000 \
  --expected specs/refseq_demo_correct1.json
```

```json
"expected_vs_actual": {
  "matched": 0,
  "missing": [ /* 7 golden events with no actual counterpart: off36@6000, on72@6000, … */ ],
  "extra":   [ /* 7 actual events with no golden counterpart: off48@24000, on52@24000, … */ ],
  "mistimed": [],
  "wrong_field": [
    {"expected": {"t_samples":0,"type":"note_on","channel":0,"note":36,"velocity":100},
     "actual":   {"t_samples":0,"type":"note_on","channel":0,"note":48,"velocity":100},
     "field": "note"}
  ]
},
"verdict": "RED",
"reasons": ["missing-or-extra: 7 missing, 7 extra", "wrong-field: note@0"]
```

Exit code: **0** (a RED is a *finding*, not a crash — same rule as audio §4.4).
**Interpretation:** the diff is structured into five buckets — `matched` (count),
`missing` (in the golden, absent from actual), `extra` (in actual, absent from
golden), `mistimed` (right event, wrong `t_samples` beyond
`timing_tolerance_samples`), and `wrong_field` (same slot, one field differs — here
the first note-on lines up in time but is note **48** where the golden expects
**36**). The `reasons[]` array is the machine-parseable summary an agent branches
on: `missing-or-extra: 7 missing, 7 extra` + `wrong-field: note@0`.

#### Gating CI/agents on RED — `--fail-on-red`

By default RED prints and exits **0**. Add `--fail-on-red` to map a RED verdict to
the **ANALYSIS** exit code (**4**) *after* printing the report, so a CI gate both
fails AND keeps the RED report on stdout:

```bash
uv run sonoscope analyze-midi --file corpus/midi/phrase_4note.mid --sample-rate 48000 \
  --expected specs/refseq_demo_correct1.json --fail-on-red
echo $?   # -> 4
```

#### Windowed inspection — `--offset/--length/--unit`

Slice a sub-window for focused inspection. Units: `samples` (default), `seconds`,
`beats`, `ticks`. The window is **literal and rebased** — a note whose partner
event falls outside the window shows up as a dangling-off / stuck-note, and
`t_samples` restarts at 0 for the sub-window. Windowing beats `[1, 3)` of the
4-note phrase (each note is exactly one beat, so the window straddles two notes):

```bash
uv run sonoscope analyze-midi --file corpus/midi/phrase_4note.mid --sample-rate 48000 \
  --offset 1 --length 2 --unit beats
```

```json
"events": [
  {"t_samples":0,     "type":"note_off", "channel":0, "note":48, "velocity":0},   /* on was pre-window */
  {"t_samples":0,     "type":"note_on",  "channel":0, "note":52, "velocity":100},
  {"t_samples":24000, "type":"note_off", "channel":0, "note":52, "velocity":0},
  {"t_samples":24000, "type":"note_on",  "channel":0, "note":55, "velocity":100}  /* off is post-window */
],
"integrity": {"every_note_on_has_off": false,
  "stuck_notes":   [{"channel":0,"note":55,"t_samples":24000, ...}],
  "dangling_offs": [{"channel":0,"note":48,"t_samples":0, ...}]},
"verdict": "RED", "reasons": ["stuck-note: ch0 n55@24000"]
```

Exit code: **0**. **Interpretation:** the window `[24000, 72000)` samples has been
**rebased** so its first event sits at `t_samples:0`. Note 48's `note_off` and
note 55's `note_on` cross the window edges, so the firewall honestly reports a
`dangling_off` and a `stuck_note` and returns `RED` — the firewall is working *on
the sub-window*. Slice to a window that fully contains its notes when you want a
green sub-capture; use the straddling behavior to prove the firewall catches a
note left open across a boundary.

#### Plugin-source capture + the determinism-bug catch (the tool's purpose)

The `--plugin` form is why `analyze-midi` exists — machine-listening QA on a live
MIDI-generating plugin, catching defects no human ear would spot:

```bash
uv run sonoscope analyze-midi --plugin path/to/ReferenceSequencer.clap \
  --spec  specs/refseq_demo_capture.json \
  --expected specs/refseq_demo_correct1.json --fail-on-red
```

It drives the CLAP note-effect through the C host over the spec's transport
window, decodes the emitted stream, runs the firewall, and diffs against the
golden. **A real bug it caught:** capturing the live **ReferenceSequencer.clap DEMO**
surfaced a Reference Sequencer **scheduler determinism defect** — Reference Sequencer *floors* (truncates)
the beat→sample offset instead of rounding to nearest. Consequences the tool
measured: (A) a trailing beat-1 downbeat lands at `t_samples=23999` instead of
`24000`, so it falls **inside** the half-open `[0, 24000)` capture window — a
**spurious 9th event** where the golden expects 8, tripping the stuck-note
firewall; (B) ±1-sample, block-size-dependent shifts (e.g. `6000` at block 512 vs
`5999` at block 128). The sonoscope host was verified correct (integer transport +
exact `event.time`→`t_samples` mapping), so the defect is Reference Sequencer's; it was
reported to ap8. **T1** (`tests/backends/test_midi_capture_integration.py`)
encodes it as `xfail(strict=True)` asserting the FAITHFUL 8-event behavior — so
the day ap8 rounds-to-nearest, the strict xfail flips to a suite failure, the
signal to remove the marker. This is the whole point: a deterministic MIDI verdict
caught a sub-sample timing bug automatically.

#### Using `analyze-midi` to verify/debug a MIDI plugin (the LLM/dev loop)

`analyze-midi` gives an LLM or developer the same deterministic ground-truth loop
as audio, in the MIDI domain — no ears, no eyeballing a piano roll:

- **A ground-truth verdict.** `midi.verdict` is `PASS`/`RED` and `midi.reasons[]`
  is a structured, machine-parseable list (`"stuck-note: ch0 n55@24000"`,
  `"missing-or-extra: 7 missing, 7 extra"`, `"wrong-field: note@0"`). An agent
  branches on these strings, never on prose.
- **CI/agent gating.** `--fail-on-red` maps RED to exit **4**, so a plugin defect
  fails a pipeline (or an autonomous agent's step) while the RED report stays on
  stdout for the post-mortem.
- **Regression goldens.** `--expected golden.json` diffs the live stream against a
  frozen expectation; the five diff buckets (`matched`/`missing`/`extra`/
  `mistimed`/`wrong_field`) tell the agent *exactly what drifted*, not just *that*
  it drifted.
- **Windowed inspection.** `--offset/--length/--unit` narrows the analysis to the
  measure or beat range under investigation (rebased), so an agent can bisect a
  large capture down to the offending event.

Same golden rule as the audio loop (§1): **gate on the deterministic verdict +
`reasons[]` + exit code.** The MIDI stream is ground truth; there is no advisory
perception layer to second-guess it.

### 4.12 "What is this render, in words I can gate on?" — the `descriptors` block

Every `analyze` report now carries an additive **`descriptors`** block: a
self-describing, **grounding-separated** vocabulary layer derived from the same
`deterministic.summary` numbers you already trust. It answers "what *kind* of
sound is this?" in three trust tiers that mirror §1's golden rule — and it never
changes a verdict or an exit code.

- **`measured`** — deterministic ground-truth terms. Each fires from a single
  librosa metric crossing a frozen absolute threshold (`bright` when
  `spectral_centroid_hz > 2500`, `loud` when `rms_dbfs > -18`, …), plus two
  **readouts** (`rhythmic-density`, `tempo-audio`). Same trust as
  `deterministic.summary`. Gate-eligible — except `estimated` readouts (tempo).
- **`hybrid`** — metric-anchored *opinion*. A named feel-term (`driving`,
  `punchy`, `warm`) computed as a weighted composite of measured metrics into a
  `[0,1]` score; it carries its `anchor_metric`/`anchor_value` so the opinion is
  traceable to numbers. Advisory-strength, not a gate.
- **`advisory`** — a labeled LALM opinion. The Qwen2-Audio freeform description
  (§4.8) is passed through a deterministic curated map onto a bounded vocabulary
  (`cosmic`, `hypnotic`, `techno`, …). **Never fatal** — a mapping crash or a
  disabled perception layer degrades to measured-only and the run still exits 0.

The `library` sub-block is provenance: `thresholds_sha256` pins the frozen
threshold set (a **sibling** of `params_sha256`, versioned independently),
`deriver_version` pins the deriver logic, and `advisory_coverage`/
`advisory_dropped` report how much of the LALM description mapped onto the vocab.

**Real output** (the `descriptors` block from an `analyze` report on a bright,
loud, driving render). Captured from a live `derive_descriptors` run over a
representative `deterministic.summary`, with the advisory row produced from the
canned Qwen description `"spacey hypnotic pad"` — reproducible from the deriver,
not hand-written:

```json
"descriptors": {
  "measured": [
    {"term": "bright",           "value": 3200.0, "metric": "spectral_centroid_hz", "direction": "high",  "threshold": 2500.0, "estimated": false, "confidence": null},
    {"term": "loud",             "value": -12.0,  "metric": "rms_dbfs",              "direction": "high",  "threshold": -18.0,  "estimated": false, "confidence": null},
    {"term": "busy",             "value": 9.0,    "metric": "onset_rate_hz",         "direction": "high",  "threshold": 8.0,    "estimated": false, "confidence": null},
    {"term": "rhythmic-density", "value": 9.0,    "metric": "onset_rate_hz",         "direction": "value", "threshold": null,   "estimated": false, "confidence": null},
    {"term": "tempo-audio",      "value": 128.0,  "metric": "tempo_bpm",             "direction": "value", "threshold": null,   "estimated": true,  "confidence": 0.82}
  ],
  "hybrid": [
    {"term": "driving", "anchor_metric": "driving_composite", "anchor_value": 0.7097058823529411, "direction": "high", "confidence": 0.7097058823529411}
  ],
  "advisory": [
    {"term": "cosmic",   "source": "lalm-mapped", "confidence": 0.6},
    {"term": "hypnotic", "source": "lalm-mapped", "confidence": 0.6}
  ],
  "summary": "measured: bright, loud, busy, driving, 9.0 onsets/s, 128 BPM; advisory: cosmic, hypnotic",
  "library": {
    "thresholds_sha256": "8a30a4cb477803982949d7cb9f4f22a6c5980241c626e6a9d2e2e39325bcd3d3",
    "deriver_version": "1.0.0",
    "advisory_coverage": 1.0,
    "advisory_dropped": 0
  }
}
```

Exit code: **0**; `tripwires.overall` is untouched — the descriptors block is
purely additive and can never fail a run.

**Interpretation, tier by tier:**

- **`measured`** — five ground-truth terms fired from this render's numbers:
  `bright` (centroid 3200 Hz `>` 2500), `loud` (−12 dBFS `>` −18), `busy` (9.0
  onsets/s `>` 8.0), and the two readouts `rhythmic-density` (9.0 onsets/s) and
  `tempo-audio` (128 BPM, `estimated: true` + `confidence`). Note `tempo-audio`
  carries `estimated: true` — an inferential term, so it is **not** gate-eligible
  even though it lives in `measured`. Terms that did *not* clear their thresholds
  (`dark`, `quiet`, `compressed`, `dynamic`, `spare`, `dense`) are simply absent —
  a term's presence *is* the assertion.
- **`hybrid`** — `driving` fired with `anchor_value` 0.7097, a weighted composite
  of onset rate, tempo, and level. It is metric-anchored opinion: read
  `anchor_metric`/`anchor_value` to see *why*, but do not gate on it.
- **`advisory`** — the LALM said `"spacey hypnotic pad"`; the curated map
  resolved `spacey → cosmic` and `hypnotic → hypnotic` (`advisory_coverage: 1.0`,
  `advisory_dropped: 0` — both surface forms mapped). Every advisory term carries
  `source: "lalm-mapped"` and a fixed base confidence. **Never gate on these.**
- **`summary`** — a one-line human read in a fixed grammar:
  gated-measured terms, then hybrids, then readouts (`… 9.0 onsets/s, 128 BPM`),
  then a `; advisory: …` clause only when advisory is non-empty.

**The LLM decision rule:** treat `descriptors.measured` (minus `estimated`
readouts) with the same trust as `deterministic.summary` — it *is* those numbers,
named. Treat `hybrid` and `advisory` exactly like `perception` (§4.8, §7):
useful hints, never gates. When advisory disagrees with measured, **the numbers
win.**

> **Thresholds are calibration-pending (placeholder) for C1.** The absolute
> firing thresholds in `DERIVER_THRESHOLDS` (and therefore the exact set of terms
> that fire for a given render) are **provisional placeholders** in cycle 1. They
> freeze only after the calibration-corpus manifest
> (`corpus/descriptors-calibration-manifest.toml`) is ratified by the maintainer and a
> real calibration run derives separating boundaries; at that point
> `thresholds_sha256` is re-pinned. The *shape* of the block — the three tiers,
> the record fields, the grammar — is stable; the *numbers* are not yet frozen.

### 4.13 "Fail my CI when the sound stops matching" — the descriptor gate

§4.12 *describes* a render; the **descriptor gate** *asserts* against it. You hand
`analyze` an expectation spec and it turns the `descriptors.measured` terms into a
`PASS`/`RED` verdict you can gate CI or an agent loop on — the same "a term's
presence *is* the assertion" idea, now enforced.

```bash
sonoscope analyze --plugin 'Surge XT.vst3' --spec render.json \
  --expect-descriptors expect.json --fail-on-red
```

**Output contract (three surfaces):**

- **stdout** — the analysis report JSON, **exactly one document**: structurally the
  same report an un-gated `analyze` emits, now with the `descriptor_gate` field
  populated instead of `null` (see below).
- **stderr** — one compact single-line JSON verdict object, byte-stable for CI
  matching: `{"verdict":"RED","reasons":[...]}` (only emitted when a spec is given).
- **exit code** — with `--fail-on-red`, a `RED` verdict returns **exit 4**
  (`ANALYSIS`); a `PASS` returns **0**. *Without* `--fail-on-red` a `RED` still
  prints its verdict but exits **0** (report-only). `--fail-on-red` with **no**
  `--expect-descriptors` is a usage error (**exit 1**,
  `USAGE_FAIL_ON_RED_REQUIRES_SPEC`). A malformed / ineligible spec, or a report
  with no `descriptors` block, is **exit 2** (`INPUT`) — the spec is validated
  *before* the render, so a bad spec never wastes a render.

**The `--expect-descriptors` spec grammar** is a JSON object with three optional
arrays:

```json
{
  "expect_present": ["bright", "loud"],
  "expect_absent":  ["dark", "quiet"],
  "expect_value": [
    {"term": "rhythmic-density", "min": 0.001},
    {"term": "loud", "equals": -12.0, "tolerance": 1.5}
  ]
}
```

- `expect_present` — each term MUST have fired.
- `expect_absent` — each term MUST NOT have fired.
- `expect_value` — a `{term, ...}` bound on a term's measured value. Exactly one
  bound form: an **`equals`** point (optional finite non-negative `tolerance`,
  default `0.0`) **or** a **band** (`min` and/or `max`, at least one side, all
  finite, `min <= max`). Any other shape is a spec error (exit 2, e.g.
  `expect_value_no_bound`, `min_gt_max`).

**Only gate-eligible terms may be asserted.** Gate-eligible = a **measured,
non-estimated** term. The audio set is the ten measured gated/readout terms:
`bright`, `dark`, `loud`, `quiet`, `compressed`, `dynamic`, `busy`, `spare`,
`dense`, `rhythmic-density`. Asserting on a hybrid/advisory term (`driving`,
`cosmic`, …), the estimated readout `tempo-audio`, a MIDI term against an audio
report, or a typo is a **spec error at load** (exit 2,
`DESCRIPTORS_EXPECTED_SPEC_INVALID`) with an exact `detail.reason` —
`term_not_gate_eligible` / `unknown_term` / `cross_context_term`. (Load-time
rejection is deliberate: it closes the vacuous-`expect_absent` green mirage, where
asserting the absence of a term that could never fire would always PASS.)

**A PASS** (stderr line) against the §4.12 render:

```json
{"verdict":"PASS","reasons":[]}
```

**A RED** — say the spec required `dense` and demanded `rhythmic-density >= 12`:

```json
{"verdict":"RED","reasons":["DESC_MISSING: dense","DESC_VALUE_OUT_OF_RANGE: rhythmic-density value=9.0 not in [12.0, inf]"]}
```

Reason ids are priority-ordered and self-describing. The formats (exact strings the
comparator emits):

| Reason id | Fires when | Example |
|---|---|---|
| `DESC_MISSING: <term>` | `expect_present` term did not fire | `DESC_MISSING: dense` |
| `DESC_UNEXPECTED: <term>` | `expect_absent` term fired | `DESC_UNEXPECTED: bright` |
| `DESC_VALUE_ABSENT: <term>` | `expect_value` term did not fire at all | `DESC_VALUE_ABSENT: rhythmic-density` |
| `DESC_VALUE_NONFINITE: <term> value=<v>` | term's value is NaN/Inf | `DESC_VALUE_NONFINITE: loud value=nan` |
| `DESC_VALUE_OUT_OF_RANGE: <term> value=<v> not in [<lo>, <hi>]` | band miss | `… value=9.0 not in [12.0, inf]` |
| `DESC_VALUE_OUT_OF_RANGE: <term> value=<v> not within <tol> of <eq>` | equals miss | `… loud value=-12.0 not within 1.0 of -6.0` |

**Gating a silent / stopped-transport render.** A silent render emits **no**
`rhythmic-density` readout (the deriver suppresses it at `onset_rate_hz == 0`), so
`{"term": "rhythmic-density", "min": 0.001}` fires `DESC_VALUE_ABSENT` → `RED` — a
one-line assertion that "this render must actually make rhythmic sound." This is
the audio value-readout gate you have **today**. The MIDI analog is **forward-
looking (arrives with C2)**: once the MIDI descriptor producer lands, the six
gate-eligible MIDI terms (`note-density`, `register`, `pitch-range`, `polyphony`,
`velocity-dynamics`, `ioi`) become assertable on `analyze-midi` reports, and
`{"term": "note-density", "min": 0.001}` catches a Reference Sequencer that emitted nothing
because the host transport was stopped. **`note-density` = unique onsets per second
— a chord struck at a single timestamp counts as 1 onset**, not one per voice.

**The persisted verdict (`descriptor_gate`).** As of `SCHEMA_VERSION` **1.3.0** the
verdict is *also* written into the report JSON on stdout (in addition to the stderr
line), so a saved report carries its own audit trail. For the RED example above —
the exact `expect.json` being:

```json
{
  "expect_present": ["dense"],
  "expect_value": [
    {"term": "rhythmic-density", "min": 12}
  ]
}
```

the report carries:

```json
"descriptor_gate": {
  "verdict": "RED",
  "reasons": ["DESC_MISSING: dense", "DESC_VALUE_OUT_OF_RANGE: rhythmic-density value=9.0 not in [12.0, inf]"],
  "spec_sha256": "77ee094037ab43b2783c2fe69ac782939e9d8f8cc43775d333fc48c58d68aa87"
}
```

`spec_sha256` is the sha256 of the **raw expectation-spec file bytes** — provenance
that ties the verdict to the exact spec that produced it (running
`sha256sum expect.json` on the exact bytes shown above reproduces
`77ee0940…aa87`). The field is
additive-**optional**: it is `null`/absent on an un-gated `analyze` and on any
pre-1.3.0 report, so old readers and old JSON stay valid. `reasons` is `[]` for a
`PASS`; `spec_sha256` is `null` when no spec was supplied.

**Regression across two renders — `iterate-descriptors`.** Where the gate asserts
one report against an authored spec, `iterate-descriptors` *observes* what moved
between **two** reports (e.g. before/after a refactor that should not change the
sound). Run `analyze` twice, save each report, then diff:

```bash
sonoscope analyze --plugin 'Surge XT.vst3' --spec base.json > baseline.json
sonoscope analyze --plugin 'Surge XT.vst3' --spec cand.json > candidate.json
sonoscope iterate-descriptors --baseline baseline.json --candidate candidate.json
```

It prints one single-line JSON `DescriptorTermDiff` to stdout (exit 0). **Real
output** (captured from two hand-built reports where the candidate lost `bright`,
gained `dense`, flipped `busy` from `high`→`low`, and drifted `loud` −12→−9.5 /
`rhythmic-density` 9.0→9.3):

```json
{"added":["dense"],"removed":["bright"],"direction_changed":["busy"],"value_drift":[{"term":"busy","baseline_value":9.0,"candidate_value":6.0},{"term":"loud","baseline_value":-12.0,"candidate_value":-9.5},{"term":"rhythmic-density","baseline_value":9.0,"candidate_value":9.3}]}
```

The **regression signal is `added` + `removed` + `direction_changed`** — a term
appearing, disappearing, or flipping firing direction is a real change in *what
kind* of sound the plugin makes. `value_drift` is a **separate, advisory** list:
raw measured values are expected to churn across renders, so drift is banded by
`--value-tolerance` (default `0.0` → report any drift). Re-running with
`--value-tolerance 0.5` drops the sub-threshold `rhythmic-density` drift (0.3) while
keeping `busy` (3.0) and `loud` (2.5):

```json
{"added":["dense"],"removed":["bright"],"direction_changed":["busy"],"value_drift":[{"term":"busy","baseline_value":9.0,"candidate_value":6.0},{"term":"loud","baseline_value":-12.0,"candidate_value":-9.5}]}
```

The diff is computed over **non-estimated measured terms only** (`tempo-audio` is
excluded — its raw value is expected to churn) and is deliberately block-kind-
agnostic: it *observes* rather than *asserts*, so it does not re-apply the gate's
eligibility rules. A missing/unparseable report, or one with no `descriptors`
block, is a typed `INPUT` error (**exit 2**) naming the failing `side`.

---

### 4.13 "What kind of PHRASE is this?" — the MIDI `descriptors` block

`analyze-midi` (§4.11) now carries the **MIDI-domain sibling** of the audio
`descriptors` block (§4.12). Every `midi-analysis` report grows an additive
`descriptors` block that reduces the decoded event stream into **exactly six
MEASURED value-readouts** — the same three-tier record shape as audio, but for
these six terms the block is **measured-only**: `hybrid` and `advisory` are
always empty in C2. It answers "what *shape* is this phrase — how dense, how
wide, how many voices?" purely from the notes, and — like audio — it never
changes a `verdict` or an exit code.

The six terms, in frozen emission order, each `direction: "value"` (a readout,
never a threshold crossing) and `estimated: false` (all six are gate-eligible
ground truth, none inferential):

| Term | Metric | Reads |
|------|--------|-------|
| `note-density` | `notes_per_second` | unique onsets per second (a chord counts as **one** onset) |
| `register` | `mean_note` | mean MIDI note number across all note-ons |
| `pitch-range` | `note_span_semitones` | highest minus lowest note, in semitones |
| `polyphony` | `max_concurrent_notes` | peak simultaneously-sounding voices |
| `velocity-dynamics` | `velocity_std` | population std-dev of note-on velocities |
| `ioi` | `median_ioi_seconds` | median inter-onset interval, in seconds |

**Real output — a populated block.** Analyzing any MIDI source attaches the
block:

```bash
uv run sonoscope analyze-midi --file corpus/midi/phrase_4note.mid --sample-rate 48000
```

The exact numbers depend on the notes. For a worked, fully-pinned fixture, take a
2.0-second window (`sample_rate = 48000`, `window_samples = 96000`) holding an
opening triad `n60 v80` + `n64 v90` + `n67 v100` (all at `t_samples=0`, released
at `24000`), then `n72 v40` at `48000` (off `72000`), then `n48 v110` at `72000`
(off `95000`). Unique onsets `{0, 48000, 72000}`; velocities `[80,90,100,40,110]`;
notes `[60,64,67,72,48]`. That input reduces deterministically to:

```json
"descriptors": {
  "measured": [
    {"term": "note-density",      "value": 1.5,                "metric": "notes_per_second",     "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "register",          "value": 62.2,               "metric": "mean_note",            "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "pitch-range",       "value": 24.0,               "metric": "note_span_semitones",  "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "polyphony",         "value": 3.0,                "metric": "max_concurrent_notes", "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "velocity-dynamics", "value": 24.166091947189145, "metric": "velocity_std",         "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "ioi",               "value": 0.75,               "metric": "median_ioi_seconds",   "direction": "value", "threshold": null, "estimated": false, "confidence": null}
  ],
  "hybrid": [],
  "advisory": [],
  "summary": "measured: 1.50 notes/s, note 62.2, 24 st, 3 voices, velocity std 24.2, 0.750s IOI",
  "library": {
    "thresholds_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "deriver_version": "midi-1.0.0",
    "advisory_coverage": null,
    "advisory_dropped": null
  }
}
```

Exit code: **0**; the `midi.verdict` firewall (§4.11) is untouched — the
descriptors block is purely additive and can never fail a run.

**Reading the `summary`, term by term** (the one-line grammar is
`measured: <note-density>, <register>, <pitch-range>, <polyphony>,
<velocity-dynamics>, <ioi>`):

- **`1.50 notes/s`** — three unique onsets across the 2.0-second window; the
  opening triad is **one** onset, not three, so density measures *attack events*,
  not note count.
- **`note 62.2`** — the mean note number `(60+64+67+72+48)/5 = 62.2`, a register
  sitting just above middle C (MIDI 60).
- **`24 st`** — the span from the lowest note (48) to the highest (72) is 24
  semitones, i.e. two octaves.
- **`3 voices`** — the opening triad is the peak concurrency; after it releases
  the phrase is monophonic, so the max is 3.
- **`velocity std 24.2`** — the population std-dev of `[80,90,100,40,110]` is
  `24.166091947189145` (rendered to one decimal); a wide dynamic spread.
- **`0.750s IOI`** — inter-onset gaps are `[1.0 s, 0.5 s]` (onsets at
  `0 / 48000 / 72000`); their median is `0.75 s`.

**The empty / stopped-transport block.** A capture with no note-ons — the
**primary** Reference Sequencer stopped-transport mode (per the host-sync contract: Reference Sequencer
emits MIDI only while the host transport is playing) — still emits all six rows,
each a `0.0` sentinel (never NaN, never omitted):

```json
"descriptors": {
  "measured": [
    {"term": "note-density",      "value": 0.0, "metric": "notes_per_second",     "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "register",          "value": 0.0, "metric": "mean_note",            "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "pitch-range",       "value": 0.0, "metric": "note_span_semitones",  "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "polyphony",         "value": 0.0, "metric": "max_concurrent_notes", "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "velocity-dynamics", "value": 0.0, "metric": "velocity_std",         "direction": "value", "threshold": null, "estimated": false, "confidence": null},
    {"term": "ioi",               "value": 0.0, "metric": "median_ioi_seconds",   "direction": "value", "threshold": null, "estimated": false, "confidence": null}
  ],
  "hybrid": [],
  "advisory": [],
  "summary": "measured: 0.00 notes/s, note 0.0, 0 st, 0 voices, velocity std 0.0, 0.000s IOI",
  "library": {
    "thresholds_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "deriver_version": "midi-1.0.0",
    "advisory_coverage": null,
    "advisory_dropped": null
  }
}
```

**The honesty model.** All six terms are **MEASURED value-readouts**, not
opinions: each is a deterministic function of the note events, so identical input
yields a byte-identical block, and `direction: "value"` marks every one as a plain
readout (there are no thresholds to cross — `threshold` is `null` on all six).
There is **no advisory or hybrid tier** in C2: the block is measured-only, so
there is nothing here to second-guess. The `library` sub-block is provenance:
`deriver_version` is `"midi-1.0.0"` (the MIDI deriver logic, versioned
independently of the audio deriver), and `thresholds_sha256` is
`44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` — the SHA-256
of the **empty** threshold set, an honest signal that this deriver uses *no*
thresholds at all.

**Scope (what C2 deliberately does NOT do).** These six numeric readouts are the
whole capability today. Deferred to future cycles, and honestly absent here:

- **Gated qualitative MIDI terms + calibration** (e.g. a `busy` / `sparse` firing
  vocabulary with tuned boundaries) — a future cycle, contingent on a calibration
  corpus, exactly as the audio thresholds remain calibration-pending (§4.12).
- **IOI-swing** (a swing/groove ratio built on inter-onset timing) — Cycle 4.
- **A MIDI advisory tier** (an LALM-style opinion layer for phrases) — Cycle 7.

Until then, treat `descriptors.measured` in a MIDI report with the same trust as
the `midi` block's ground truth: it *is* the notes, reduced to six numbers.

---

## 5. Reading the report (field reference)

The `analysis` report is the one an LLM reads most. Key fields, by trust tier:

**`tripwires` (verdicts — read these first):**
- `tripwires.overall` — `PASS` | `RED` | `ERROR`. Your primary gate.
- `tripwires.results[]` — per-tripwire `{id, verdict, detail}`; ids:
  `silent-output`, `nan-inf`, `denormal`, `clipping`.
- `tripwires.expected_audio` — whether audio was expected (drives `silent-output`).

**`deterministic` (ground truth):**
- `deterministic.summary.rms_dbfs`, `peak_dbfs`, `crest_factor_db` — levels.
- `deterministic.summary.spectral_centroid_hz` — brightness (the one axis Qwen
  also grounds reliably).
- `deterministic.summary.spectral_flatness` — noisiness (0 tonal … 1 noise-like).
- `spectral_bandwidth_hz`, `spectral_rolloff_hz`, `zero_crossing_rate`,
  `onset_count`, `onset_rate_hz`, `tempo_bpm`/`tempo_confidence` (nullable),
  `mfcc_mean[13]`, `mfcc_std[13]`.
- `deterministic.integrity` — booleans/counts: `is_silent`, `has_nan`, `has_inf`,
  `has_denormal`, `clip_count`, `clip_fraction`, `dc_offset_exceeds` (+ thresholds).

**`render`:** `sample_rate_hz`, `block_size`, `channels`, `duration_s`,
`wav_sha256`, `render_wall_ms`, and `render.determinism`
(`is_bit_identical`, `patch_class`, `noise_floor_measured`, embedded `floors`).

**`input`:** provenance — `plugin.binary_sha256`, `stimulus.ref`/`ref_sha256`,
`param_set.spec_sha256`/`resolved_sha256`. Use these to prove reproducibility.

**`perception` (advisory):** `status` (`ok`/`disabled`/`unavailable`/`error`),
`grounding`, `description`, `disclaimer`, `adapter`. Ignore for gating.

**`errors[]`:** non-fatal issues collected during the run (each
`{code, message, severity, component}`). An empty array is the healthy case. A
**fatal** error is NOT here — it replaces the whole report with a `fatal-error`
object and a non-zero exit code.

**`iterate-delta` report:** `delta.{baseline_value, candidate_value, abs_delta,
measured_floor, noise_threshold, significant, matches_expectation}` and the
top-level `verdict` (`PASS`/`FAIL`/`INCONCLUSIVE`).

---

## 6. Exit codes (control flow)

Errors are emitted as a single `fatal-error` JSON object on stdout **and**
signalled by the process exit code. Branch on `$?`:

| Exit | Name | Meaning | LLM action |
|-----:|------|---------|------------|
| **0** | OK | Ran successfully. **Includes a RED tripwire / silent-output finding** — the tool worked, the plugin has a defect. | Read `tripwires.overall` / `verdict` from the JSON to get the plugin verdict. |
| **1** | USAGE | Bad flag, unknown command, unsupported mode (e.g. `analyze --wav`). | Fix the invocation. |
| **2** | INPUT | Input-contract failure: invalid/unreadable spec, unknown param name, corpus hash drift. | Fix the spec/stimulus/params. |
| **3** | RENDER | Backend load/crash, GUI-init required without raw_state. | Investigate the plugin/backend. |
| **4** | ANALYSIS | Deterministic feature layer failed (unreadable wav, numeric fault). | Likely a tool/plugin-output problem. |
| **5** | ENVIRONMENT | Pin/lockfile drift or a required runtime hard-failed (`doctor` failed). | Run `doctor`; fix the environment. |

**The single most important distinction for an LLM:** a **plugin defect**
(silence, NaN, clipping) is exit **0** with `overall:"RED"` — a *finding*, not a
crash. A **non-zero** exit means *sonoscope itself could not complete* (bad
input, environment, or an internal fault). Real fatal-error examples observed:

```
analyze --wav /no/such.wav     -> INPUT_WAV_UNREADABLE          exit 2
analyze --plugin /no/such.vst3 -> PLUGIN_PATH_NOT_FOUND         exit 2
analyze --spec bad.json        -> INPUT_SPEC_INVALID            exit 2
analyze (unknown param name)   -> PARAM_UNKNOWN_NAME            exit 2
corpus verify (hash drift)     -> INPUT_CORPUS_DRIFT            exit 2
```

---

## 7. Caveat: perception is advisory, deterministic is truth

To restate the one rule that keeps this loop trustworthy:

- **Gate on `tripwires.overall`, `deterministic.*`, and `iterate` `verdict`.**
  These are deterministic, reproducible, and floor-gated.
- **Never gate on `perception.description`.** It is a fallible LALM hint with a
  built-in disclaimer; it is structurally quarantined from ground truth and can
  never fail a run (a perception failure degrades to `status:"error"`, exit 0).
- When perception and the numbers disagree, **the numbers win.** Perception's
  only reliably-grounded axis is *brightness ↔ `spectral_centroid_hz`*; noisiness
  is weak and loudness is unreliable, so read those from `deterministic` instead.
