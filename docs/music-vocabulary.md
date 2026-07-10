# Music-description vocabulary (sonoscope) — v1

**Status:** design/docs draft (v1 for eek's refinement). No `src/` written.
**Scope:** the controlled vocabulary sonoscope may use to *describe* audio and
MIDI, and the discipline that keeps every term honest about how much of it is
measured versus opinion.

This is a **taxonomy + grounding contract**, not an implementation. It names the
descriptor terms, groups them, and — most importantly — tags each one by how it
is grounded, so a sonoscope description can always say *"this much is measured,
this much is opinion."*

---

## 1. The spine: three grounding tags

sonoscope's core discipline (AGENTS.md, design §2.1): **the deterministic layer
is GROUND TRUTH; the perception/AI layer is LABELED ADVISORY and never fatal.**
This vocabulary inherits that split at the level of the individual word. Every
term below carries exactly one grounding tag:

| Tag | Meaning | Trust | What the doc gives you |
|---|---|---|---|
| **measured** | A defined 1:1 mapping to a computed metric (audio or MIDI feature). Ground truth, cross-checkable, QA-defensible. | Deterministic | the metric + mapping direction |
| **hybrid** | A metric gives a *partial* anchor; the AI/perception layer adds feel the metric cannot fully justify. | Anchored opinion | the proxy metric (partial) + which part is advisory |
| **advisory** | Genuinely subjective. No honest metric. The perception layer's labeled opinion. | Opinion | the loose lean, if any, but no claim of measurement |

The tag split **is the point.** It is what lets sonoscope emit
`measured: bright, driving, 128 BPM` alongside `advisory: cosmic, hypnotic`
without pretending the second half is ground truth.

**Availability marker.** Measured/hybrid terms also carry how computable they are
*today*:

- **`now`** — grounded in a feature sonoscope already computes (see §2).
- **`new`** — needs added analysis. Two sub-kinds:
  - **`new-reduce`** — a straightforward reduction over data already captured
    (e.g. MIDI note-stream stats). Low risk.
  - **`new-dsp`** — needs a genuinely new analysis stage (key detection, chroma,
    swing quantification). Higher risk; flagged in §12 open questions.

Direction notation: `↑term` = metric high ⇒ term applies; `↓term` = metric low ⇒
term applies.

---

## 2. Measurement substrate (what sonoscope can actually measure)

Grounded terms MUST map to one of these. This is the whole honest anchor set.

### 2a. Audio features — `deterministic.summary` + `integrity`

Computed by `features/librosa_features.py` (frozen, hashed param set: 48 kHz,
n_fft 2048, hop 512, 13 MFCC) and `features/integrity.py`. Per-channel, reduced
by mean (peak by max). These are the **timbre / level / rhythm** anchor.

| Metric (field) | What it measures | Grounds (terms) |
|---|---|---|
| `spectral_centroid_hz` | spectral center of mass | brightness, pitch-register (proxy), warmth |
| `spectral_bandwidth_hz` | spread around the centroid | thin/full, thick/thin |
| `spectral_rolloff_hz` (85%) | frequency below which 85% of energy sits | airy, boomy, high-freq extension |
| `spectral_flatness` | tonal vs noise-like (geometric/arithmetic mean ratio) | noisiness, harsh, clean/dirty, metallic |
| `zero_crossing_rate` | sign-change rate (noise/HF proxy) | grit, harshness, noisiness (secondary) |
| `mfcc_mean` (13) | timbral envelope shape | warm, hollow, nasal, woody (shape cues) |
| `mfcc_std` (13) | timbral variability over time | static vs evolving timbre |
| `rms_dbfs` | RMS level | loud/quiet (**deterministic-only**; see §11) |
| `peak_dbfs` | true peak (max across channels) | headroom, transient peak |
| `crest_factor_db` | peak − RMS | dynamic/compressed, punchy |
| `onset_count`, `onset_rate_hz` | detected onsets, onsets/sec | rhythmic density, busy/sparse, driving/frantic |
| `tempo_bpm`, `tempo_confidence` | estimated tempo (gated: ≥4 onsets, 40–300 BPM, else `null`) | tempo/BPM (audio, **estimated**) |
| `dc_offset`, `is_silent`, `has_nan/inf/denormal`, `clip_*` | integrity tripwires | not descriptive — QA gates (context only) |

Not yet computed but standard librosa (would unlock harmony/key from audio):
`chroma_stft` / `chroma_cqt` (pitch-class energy). Flagged `new-dsp`.

### 2b. MIDI features — the note stream (`MidiBlock` / `MidiEvent`)

Captured by the v2 MIDI path. The event list is
GROUND TRUTH (like librosa). Each `MidiEvent` = `{t_samples, t_ticks (960 PPQ),
type (note_on/off), channel 0–15, note 0–127, velocity 0–127}`. Transport
metadata (`MidiCaptureMeta`) carries **host-provided** tempo/time-signature — not
estimated, exact.

| Source | What it measures | Grounds (terms) |
|---|---|---|
| `capture_meta.tempo_bpm` | host transport tempo (exact) | tempo/BPM (MIDI, **exact**) |
| `capture_meta.tsig_num/den` | host time signature (exact) | time signature |
| note-on count / duration | note density (notes/sec) | rhythmic density, busy/sparse |
| `note` min/max/mean/spread | pitch range & register | register, range, pitch-height |
| concurrent-note count | polyphony / voice count | dense/thick, chordal vs monophonic |
| distinct `channel` set | multitimbral layering | layered, multitimbral |
| `velocity` distribution | symbolic dynamics | dynamic/flat, accented, velocity-dynamics |
| inter-onset intervals (Δ`t_ticks`) | rhythm grid & spacing | syncopation, tight/loose, swing (derived) |
| pitch-class histogram of `note` | key / mode / harmony | key, mode, consonance (**needs K-S profiling**) |
| onset offset vs quantized grid | micro-timing | swung/straight, pushing/dragging, human |

**Important:** the descriptive MIDI *reductions* above (density, register,
polyphony, velocity stats, IOI) are **not computed today** — the v2 MIDI layer
computes tripwires/integrity/diff, not descriptive statistics. They are
`new-reduce` (simple passes over the captured list). Key/mode/harmony from
pitch-class content is `new-dsp` (needs a Krumhansl-Schmuckler-style profiler).

### 2c. Prior seed vocab (G2 probe)

The B3/R6 probe tested three candidate structured terms against Qwen2-Audio:

- **`brightness` ↔ `spectral_centroid_hz`: reliable** — the clean one.
- **`noisiness` ↔ `spectral_flatness`: weak** — directionally right, soft magnitude.
- **`dynamics`/loudness: deterministic-only** — the LALM has no absolute-level
  reference; loudness stays RMS ground truth, never a Qwen-grounded descriptor.

This is why brightness is confidently `measured`, noisiness is `hybrid`, and
loudness is `measured` but **deterministic-only** (§11).

---

## 3. Category A — Objective music theory (mostly measured)

The QA-defensible spine. Most map cleanly; a few need new analysis.

| Term | Definition | Tag | Metric · direction · availability |
|---|---|---|---|
| tempo / BPM | beats per minute | **measured** | MIDI `capture_meta.tempo_bpm` (exact, `now`); audio `tempo_bpm` (estimated + gated, `now`) |
| time signature | metrical grouping (4/4, 3/4…) | **measured** | MIDI `tsig_num/den` (exact, `now`); audio → `new-dsp` |
| note density | notes per second | **measured** | MIDI note-ons / duration (`new-reduce`) |
| rhythmic density | event rate of the audio | **measured** | audio `onset_rate_hz` (`now`) |
| pitch range | lowest→highest pitch span | **measured** | MIDI `note` max−min (`new-reduce`) |
| register | overall pitch height (bass…treble) | **measured** | MIDI mean `note` (`new-reduce`); audio `spectral_centroid_hz` proxy (`now`, hybrid-lean) |
| polyphony / voice count | simultaneous notes | **measured** | MIDI concurrent-note max/mean (`new-reduce`) |
| key | tonal center (C, F#…) | **measured** | pitch-class histogram + K-S profile — MIDI or audio chroma (`new-dsp`) |
| mode | major / minor / modal | **measured** | same profiler, major-vs-minor template (`new-dsp`) |
| dynamic range | loud-to-soft span | **measured** | audio `crest_factor_db` + peak−RMS (`now`); MIDI velocity spread (`new-reduce`) |
| syncopation | accent off the metric grid | **hybrid** | MIDI off-beat IOI/accent weight (`new-dsp`); metric anchors, "feel" is advisory |
| harmonic content / consonance | interval/chord consonance | **hybrid** | MIDI pitch-class intervals; audio chroma (`new-dsp`); consonance judgment part-advisory |

---

## 4. Category B — Timbre / tone color (measured + hybrid)

Audio-only (MIDI carries no timbre). Anchored to centroid, flatness, rolloff,
zcr, bandwidth, MFCC shape. The richest *measured* descriptive set.

| Term | Definition | Tag | Metric · direction · availability |
|---|---|---|---|
| bright / dark | high vs low spectral center | **measured** | `↑spectral_centroid_hz` = bright (`now`) — the reliable anchor |
| warm / cold | full low-mids, gentle highs vs thin/edgy | **hybrid** | low-ish centroid + low-mid MFCC energy (`now`); "warm" feel is advisory |
| harsh / smooth | edgy/abrasive vs even | **hybrid** | `↑flatness` + `↑zcr` + high rolloff = harsh (`now`); abrasiveness part-advisory |
| thin / full | narrow vs broad spectral body | **hybrid** | `↑spectral_bandwidth_hz` + low-freq energy = full (`now`) |
| hollow | scooped mids / comb-notched | **hybrid** | MFCC dip signature (`now`); largely shape-inferred |
| airy | extended, delicate highs | **hybrid** | `↑spectral_rolloff_hz` + moderate flatness (`now`) |
| boomy | low-frequency dominance | **hybrid** | `↓rolloff` + `↓centroid` + high low energy (`now`) |
| nasal | honky mid emphasis | **advisory** | MFCC mid-band lean (weak proxy); mostly opinion |
| metallic | inharmonic bright partials | **hybrid** | `↑centroid` + `↑flatness` + `↑zcr` (`now`); inharmonicity not directly measured |
| glassy | bright *and* clean | **hybrid** | `↑centroid` + `↓flatness` (`now`) |
| woody | resonant, organic mids | **advisory** | MFCC shape lean; mostly opinion |
| gritty | textured distortion | **hybrid** | `↑flatness` + `↑zcr` (`now`); "grit" character part-advisory |
| clean / dirty | pure vs saturated/noisy | **hybrid** | `↓flatness` + high crest = clean (`now`) |

---

## 5. Category C — Texture / space (hybrid)

Anchored to polyphony, note-density, spectral spread. "Space" cues (reverb,
depth) have **no metric today** and stay advisory.

| Term | Definition | Tag | Metric · direction · availability |
|---|---|---|---|
| dense / sparse | many vs few simultaneous events | **measured** | audio `onset_rate_hz` / MIDI note-density + polyphony (`now` audio, `new-reduce` MIDI) |
| minimal | deliberately few elements | **hybrid** | low density + low polyphony (`now`/`new-reduce`); intent is advisory |
| busy | high event rate | **measured** | `↑onset_rate_hz` (`now`) |
| spare | wide gaps between events | **measured** | `↓onset_rate_hz` / long IOIs (`now`/`new-reduce`) |
| thick / thin | heavy vs light layered body | **hybrid** | polyphony + `spectral_bandwidth_hz` (`now`/`new-reduce`) |
| layered | multiple stacked parts | **measured** | MIDI polyphony + distinct channels (`new-reduce`) |
| wall-of-sound | saturated, gapless mass | **advisory** | high polyphony + high RMS + wide bandwidth *lean* (`now`); the gestalt is opinion |
| spacious / intimate | sense of room / distance | **advisory** | no reverb/depth metric today — pure opinion |

---

## 6. Category D — Dynamics / energy (measured + hybrid)

Anchored to RMS, crest factor, onset rate, tempo, MIDI velocity.

| Term | Definition | Tag | Metric · direction · availability |
|---|---|---|---|
| loud / quiet | absolute level | **measured** | `rms_dbfs` (`now`, **deterministic-only** — never AI-grounded, §11) |
| compressed | small peak-to-average | **measured** | `↓crest_factor_db` (`now`) |
| dynamic | wide peak-to-average | **measured** | `↑crest_factor_db` + MIDI velocity variance (`now`/`new-reduce`) |
| punchy | strong transients | **hybrid** | `↑crest_factor_db` + fast onset transients (`now`); "punch" part-advisory |
| driving | relentless forward push | **hybrid** | steady high `onset_rate_hz` + tempo + RMS (`now`); feel is advisory |
| frantic | fast, agitated | **hybrid** | `↑onset_rate_hz` + `↑tempo_bpm` (`now`); agitation advisory |
| gentle | soft, unhurried | **hybrid** | `↓rms_dbfs` + `↓onset_rate_hz` (`now`) |
| laid-back | relaxed, behind-the-beat | **hybrid** | low onset rate + micro-timing behind grid (`now`/`new-dsp`) |
| aggressive (energy) | forceful, hard-hitting | **hybrid** | `↑rms` + `↑crest` + high HF energy + fast onsets (`now`); see also mood (§8) |

---

## 7. Category E — Groove / feel (hybrid + advisory)

Anchored to inter-onset-interval regularity, micro-timing offsets, velocity
variance — almost entirely **MIDI-derived** and mostly `new` analysis.

| Term | Definition | Tag | Metric · direction · availability |
|---|---|---|---|
| tight / loose | low vs high timing scatter | **hybrid** | IOI deviation from grid variance (`new-dsp`); "loose" feel advisory |
| on-the-grid | quantized to the beat | **measured** | grid-quantization error ≈ 0 (`new-dsp`) |
| swung / straight | uneven vs even subdivision | **measured** | off-beat 8th placement ratio / swing quotient (`new-dsp`) |
| pushing / dragging | ahead of / behind the beat | **hybrid** | mean signed micro-timing offset (`new-dsp`) |
| syncopated | accents off the strong beats | **hybrid** | off-beat accent weight (`new-dsp`); see §3 |
| mechanical / human | rigid vs micro-varied | **hybrid** | micro-timing + velocity variance near zero = mechanical (`new-dsp`); "human" advisory |
| hypnotic / repetitive | strong repetition, low variation | **advisory** | low pattern entropy / high self-similarity *lean* (`new-dsp`); "hypnotic" is opinion |

---

## 8. Category F — Affect / mood (mostly advisory)

Framed on the **valence–arousal** 2-axis model:

- **Arousal** (calm↔excited): loosely proxied — `tempo_bpm`, `rms_dbfs`,
  `onset_rate_hz` all lean high = high arousal. A *loose* measured anchor.
- **Valence** (negative↔positive): weakly proxied by **mode** (major≈positive,
  minor≈negative) once key detection exists (`new-dsp`); otherwise no metric.

All terms below are **advisory** — the proxy is a lean, not a mapping.

| Term | Valence / arousal | Loose lean (advisory only) |
|---|---|---|
| tense | neg / high | dissonance + fast + loud |
| calm | neutral / low | slow + quiet + sparse |
| dark (mood) | neg / — | minor mode + low register |
| uplifting | pos / high | major + bright + moderate-fast |
| melancholy | neg / low | minor + slow + sparse |
| aggressive (mood) | neg / high | loud + fast + distorted timbre |
| playful | pos / mid | major + bouncy syncopation + mid tempo |
| ominous | neg / low-mid | minor + low register + slow |
| serene | pos / low | consonant + slow + smooth timbre |
| anxious | neg / high | fast + irregular + rising |
| triumphant | pos / high | major + loud + bright/brassy |
| nostalgic | mixed / low | (no honest proxy — pure opinion) |
| euphoric | pos / high | major + fast + bright + loud |

---

## 9. Category G — Evocative / aesthetic adjectives (advisory)

The rich descriptive layer. All **advisory**. A loose lean is noted only where a
metric weakly co-varies; it is never a claim of measurement.

| Term | Loose lean (advisory only) |
|---|---|
| ethereal | sparse + high register + smooth |
| cosmic | slow + sparse + long-decay + high register |
| dreamy | soft + smooth + moderate density |
| lush | high polyphony + wide bandwidth + warm |
| glassy (aesthetic) | bright + clean (echoes §4) |
| retro / vintage | band-limited + narrower bandwidth |
| futuristic | bright + synthetic + wide spectrum |
| campy | (no honest proxy) |
| cinematic | wide dynamic range + layered |
| psychedelic | evolving timbre (high `mfcc_std`) + modulation |
| gritty (aesthetic) | high flatness/zcr (echoes §4) |
| warm (aesthetic) | low-mid energy (echoes §4) |
| organic / synthetic | timbral regularity / inharmonicity lean |
| alien | inharmonic + unusual intervals |
| epic | loud + wide bandwidth + layered + slow-building |

Terms that echo a measured/hybrid sibling (glassy, gritty, warm) are listed in
both places deliberately: the **timbre** entry is the anchored one; the
**aesthetic** entry is the mood-flavored opinion. sonoscope should prefer the
anchored sense when a metric supports it.

---

## 10. Category H — Genre / style (advisory, controlled taxonomy)

Genre is **advisory** — an AI classification. Some measured features weakly
correlate (tempo band, density) but never determine genre. Keep a controlled
tree so labels are consistent; do not let the model invent freeform genres.

| Family | Representative styles | Weak measured correlates |
|---|---|---|
| Ambient / drone | ambient, drone, dark ambient, new age | very low onset rate, no/low tempo, sustained |
| Electronic — 4/4 | techno, house, trance, minimal | tempo 120–140, steady onset grid |
| Electronic — breaks | DnB, jungle, breakbeat, garage | tempo 160–175, high onset rate, syncopated |
| Bass / hip-hop | hip-hop, trap, lo-fi, dubstep | tempo 70–160 (half-time feel), sparse-heavy |
| Rock / guitar | rock, indie, punk, post-rock | mid tempo, high RMS, high flatness (distortion) |
| Metal | metal, djent, doom, black | high flatness + high RMS + fast onsets |
| Jazz | jazz, fusion, swing, bebop | swing timing, wide harmony, dynamic |
| Classical / orchestral | classical, romantic, chamber, orchestral | wide dynamic range, acoustic timbre, rubato |
| Cinematic / score | film score, trailer, underscore | layered, wide dynamics, slow builds |
| Folk / acoustic | folk, singer-songwriter, country | acoustic timbre, moderate density |
| Pop | pop, synth-pop, dance-pop | 4/4, tempo 100–130, compressed |
| Experimental | noise, musique concrète, glitch, IDM | high flatness, irregular onsets, non-tonal |

---

## 11. Special case: loudness stays deterministic-only

Per the B3/R6 probe: a single-clip LALM has **no absolute-loudness reference**
(it called a −52 dBFS clip "loud"). Therefore **loud/quiet, compressed, dynamic
range** are `measured` from `rms_dbfs`/`crest_factor_db` and **must never be
sourced from the perception model.** This is the cleanest example of the
architecture: deterministic owns level, perception owns timbre.

---

## 12. How this plugs into sonoscope

Three layers, matching the schema's grounding split
(`Grounding = advisory-freetext | structured-vocab | none`, `models.py`):

1. **Measured subset = the structured-vocab core (G2).** Every `measured` term
   maps 1:1 to a metric and is cross-checkable against
   `deterministic.summary` / the MIDI reductions. This is the QA-defensible
   grounding — the natural payload for a `grounding="structured-vocab"`
   `grounding_map` (design §3.3). Terms here can gate, delta, and tripwire.

2. **Hybrid layer = anchored advisory.** Emitted as advisory, but sonoscope can
   attach the anchoring metric so a reader sees *why* (e.g. `harsh` +
   `spectral_flatness=0.31`). The metric bounds the claim; the adjective adds the
   feel.

3. **Advisory layer = constrained perception vocabulary.** Mood, aesthetic, and
   genre terms drawn from a **fixed list** (§8–§10), labeled advisory, carrying
   the standard perception disclaimer. Constraining the model to this list (vs
   freeform) keeps outputs consistent and auditable.

**A combined description would read:**

```
measured : bright, dynamic, 128 BPM, 4/4, key A minor, dense
hybrid   : harsh (flatness 0.31), driving
advisory : cosmic, hypnotic, techno
```

The reader can trust the first line as ground truth, treat the second as
metric-anchored opinion, and the third as labeled perception. That transparency
is the whole product.

---

## 13. Application & output — the vocab is USED and RETURNED

The vocabulary is not a glossary; it is applied per-track and **returned in the
analysis JSON** as a `descriptors` block. This is the implementation that
*follows* this spec — two producers (a deterministic deriver + a constrained
advisory path), one output shape — wired into **both** the audio `analyze` and
the MIDI `analyze-midi` outputs.

### 13a. Measured producer — the deterministic descriptor-deriver

A pure function over `deterministic.summary` (audio) and the MIDI reductions
(§2b). Each `measured` term **fires when its mapped metric crosses a defined
threshold or falls in a defined range.** Examples (thresholds are illustrative —
calibration is an open question, §16.7):

- `bright` fires when `spectral_centroid_hz > ~2500`; `dark` when `< ~800`.
- `dense` fires when `onset_rate_hz > ~8` (audio) or notes/sec `> ~6` (MIDI).
- `compressed` fires when `crest_factor_db < ~6`; `dynamic` when `> ~15`.
- `swung` fires when the swing quotient departs from 0.5 beyond a threshold.

Each fired term is returned as a **ground-truth descriptor** carrying its
evidence: `{term, value, metric, direction}`. Because the metric is attached,
every measured descriptor is **cross-checkable** — re-run the deriver on the
same summary and it must reproduce. This IS the `grounding="structured-vocab"`
(G2) payload from `models.py`: the deriver populates the `grounding_map`
(term → metric) directly.

Because it is deterministic, the measured descriptor set can gate, delta, and
tripwire, and it ships with RED/GREEN tests (AGENTS.md discipline): a summary
just over threshold fires the term (GREEN), just under does not (RED).

### 13b. Advisory producer — perception constrained to the vocab

The advisory descriptors come from the perception layer **constrained to the
controlled vocabulary** (§8–§10), never freeform:

- **Audio timbre/mood** → the Qwen2-Audio LALM (`qwen_local.py`), selecting from
  the timbre/mood/aesthetic terms.
- **Symbolic / genre / evocative / mood-from-structure** → a text LLM given the
  **feature + MIDI summary** (not raw audio), selecting genre (§10), aesthetic
  (§9), and mood (§8) terms.

Output is a set of **labeled-advisory descriptors** `{term, confidence?}`,
carrying the standard perception disclaimer, `grounding="advisory-freetext"` (or
a new `constrained-vocab` grounding if the selection is list-restricted). A
crash/timeout degrades to measured-only (advisory omitted, exit 0) — advisory is
never fatal (design §2.1).

### 13c. Returned output shape — the `descriptors` block

A new block in the analysis JSON (proposed; additive, schema-versioned):

```jsonc
"descriptors": {
  "measured": [
    { "term": "bright",  "value": 2840.0, "metric": "spectral_centroid_hz", "direction": "high" },
    { "term": "driving", "value": 9.2,    "metric": "onset_rate_hz",        "direction": "high" },
    { "term": "128 BPM", "value": 128.0,  "metric": "tempo_bpm",            "direction": "value" }
  ],
  "advisory": [
    { "term": "cosmic",   "confidence": 0.7 },
    { "term": "hypnotic", "confidence": 0.6 },
    { "term": "techno",   "confidence": 0.8 }
  ],
  "summary": "measured: bright, driving, 128 BPM; advisory: cosmic, hypnotic, techno"
}
```

Every returned term is tagged by grounding via the block it sits in
(`measured` vs `advisory`). The one-line `summary` is the human-readable render
of the same split. The block plugs into both report kinds: audio `AnalysisReport`
gets measured (audio metrics) + advisory (Qwen); `MidiAnalysisReport` gets
measured (MIDI reductions) + advisory (text-LLM from the MIDI summary).

**Sequencing:** measured runs first and always (ground truth); advisory runs
second and optionally (labeled, degradable). The measured block never depends on
the advisory block.

---

## 14. Prior art & open sources

### 14a. Pandora Music Genome Project — inspiration, IP off-limits

The **Pandora Music Genome Project** is the direct design inspiration:
musicologists hand-annotated ~450 controlled musical attributes per song, and
that per-track controlled-vocabulary annotation drives recommendation and
pairing. That "controlled vocabulary, applied per track, used downstream" model
is exactly what §13 emulates.

**Constraint:** Pandora's attribute set and annotation database are
**proprietary trade secrets** — not licensable, not scrapeable, not usable. We
take the *model* (controlled vocab applied per track), never their data or their
attribute list. Our vocabulary is built independently from the open sources
below and from sonoscope's own measurement substrate.

### 14b. Open analogues — term set and grounding candidates

Several currently-`advisory` terms could become `measured`/`hybrid` by wiring an
open MIR model. These are **"could-ground-via-open-MIR"** candidates:

| Open source | What it offers | sonoscope use |
|---|---|---|
| **Essentia / AcousticBrainz** | open-source MIR: high-level mood, genre, danceability, timbre descriptors from audio via open pretrained models | promote mood (§8), danceability, genre (§10), some timbre (§4) from advisory → **measured/hybrid** by running an open Essentia model as a *deterministic* producer. Tagged **`open-MIR-groundable`** below. |
| **Spotify audio-features scheme** | documented descriptor *design*: danceability, energy, valence, acousticness, instrumentalness, speechiness | reference for the measured/hybrid **axis design** (not their data) — e.g. an `energy` axis (RMS + onset + HF) and a `danceability` axis (tempo + beat regularity) |
| **Russell valence–arousal** | 2-axis affect model | the mood frame in §8 |
| **GEMS (Geneva Emotional Music Scale)** | music-specific emotion taxonomy (9 factors: wonder, transcendence, tenderness, nostalgia, peacefulness, power, joyful activation, tension, sadness) | a principled, citable controlled mood vocabulary to expand/replace ad-hoc §8 terms |
| **Timbre-descriptor studies** (e.g. brightness/roughness/warmth acoustic correlates) | validated term↔acoustic-feature mappings | evidence backing the §4 hybrid anchors (centroid↔brightness, roughness↔flatness) |

**`open-MIR-groundable` candidates** (advisory today → measured/hybrid if an open
Essentia/AcousticBrainz model is wired as a deterministic producer):

- **mood** (§8): Essentia mood models (happy/sad/aggressive/relaxed/party).
- **danceability** (new axis): Essentia danceability.
- **genre** (§10): Essentia/AcousticBrainz genre classifiers (still advisory-ish
  — genre classification is soft — but *reproducible* if the model is pinned).
- **timbre** (§4): several Essentia timbral descriptors (brightness, roughness).

Note: an Essentia model wired as a producer is **deterministic** (same input →
same output, pinnable) even when the *concept* is soft — that reproducibility is
what lets it move from `advisory` to `measured`/`hybrid`. Whether to add Essentia
as a dependency is an open question (§16.8).

---

## 15. Extensibility — adding a term

1. **Pick the tag.** Is there an honest metric?
   - Clean 1:1 mapping → **measured**. Give the metric + direction + `now`/`new`.
   - Partial anchor + feel → **hybrid**. Give the proxy + name the advisory part.
   - No honest metric → **advisory**. Put it in §8–§10; it inherits the
     disclaimer. Note a loose lean only if one genuinely co-varies.
2. **If measured/hybrid and the metric doesn't exist yet**, mark `new-reduce`
   (simple stat over captured data) or `new-dsp` (new analysis stage) and file
   it against the open questions (§16) — do NOT tag it `measured`/`now` until the
   metric ships and a RED/GREEN test proves the mapping (AGENTS.md testing
   discipline).
3. **Advisory terms always carry the disclaimer** and should be added to the
   controlled list, not invented ad hoc by the model at inference time.

Every `measured` term is a promise that a deterministic metric backs it. Do not
make that promise without the metric and its test.

---

## 16. Open questions (for eek)

1. **Which `new` metrics to build, in what order?** The `new-reduce` MIDI stats
   (density, register, polyphony, velocity dynamics, IOI) are cheap and unlock a
   lot of Category A/C/D. The `new-dsp` set is heavier:
   - **key/mode detection** (pitch-class histogram + Krumhansl-Schmuckler) —
     unlocks key, mode, and the valence proxy. Audio (chroma) vs MIDI
     (pitch-class) — build one or both?
   - **swing/micro-timing quantification** — unlocks all of Category E. MIDI-only
     and well-defined (offset vs quantized grid), but needs a grid-inference
     step.
   - **chord/harmony detection** — the heaviest; needed for consonance/harmonic
     content. Defer?
2. **Audio chroma.** Add `chroma_*` to the frozen feature set to enable audio-side
   key/harmony? It changes `params_sha256` (a deliberate, versioned change).
3. **How large should the advisory sets be?** Genre (§10) and aesthetic (§9) are
   currently modest. Bounded controlled vocabulary (auditable, consistent) vs
   open freeform (expressive, but drifts and can't be validated)? Recommendation:
   **bounded** — it matches the whole "no false green" ethos.
4. **Perception term-emission policy.** Should the model be *restricted* to the
   controlled advisory list (structured selection) or allowed freeform prose that
   is then mapped back onto the list? The B3 probe suggests freeform is what Qwen
   does naturally; a mapping step keeps it auditable.
5. **Register overlap: audio-centroid vs MIDI-pitch.** "Register" has both an
   audio proxy (centroid) and an exact MIDI source (mean note). When both exist,
   which wins, and do they cross-check each other (a useful tripwire)?
6. **Genre confidence.** Should advisory genre carry a confidence/top-k rather
   than a single label, given how weak the measured correlates are?
7. **Threshold calibration (§13a).** The measured-deriver thresholds (`bright`
   when centroid > X, `dense` when onset rate > Y, …) need calibration against a
   real corpus. Fixed thresholds, or percentile/relative thresholds? These become
   part of the frozen, hashed param set (like `FROZEN_PARAMS`) so any change is
   detectable.
8. **Essentia dependency (§14b).** Wire an open Essentia/AcousticBrainz model to
   promote mood/danceability/genre/timbre from advisory → measured/hybrid? It
   adds a dependency and a model pin, but buys reproducible high-level
   descriptors. Which axes are worth it first?
9. **`descriptors` block schema (§13c).** Land the `descriptors` block as an
   additive, schema-versioned addition to both `AnalysisReport` and
   `MidiAnalysisReport`? Confirm the field shapes (`{term, value, metric,
   direction}` / `{term, confidence?}`) and whether a new `constrained-vocab`
   `Grounding` literal is warranted.

---

*v1 — organized, comprehensive, honestly tagged. Refine freely.*
