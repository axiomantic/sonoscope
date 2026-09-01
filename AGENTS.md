# sonoscope

Machine-listening / audio-QA harness. sonoscope consumes *built* plugins (wrapped VST3
today; native CLAP later) and runs the closed loop
`(plugin, stimulus, param-set) -> wav -> versioned analysis JSON`. It is a **CLI-first tool**
for **regression / tripwire QA** — silence, NaN/Inf, clipping, denormals, gross spectral
deltas, feature-delta sanity, nondeterminism floors — **not** aesthetic "does it sound good"
judgment. Its job is to catch real render failures reproducibly and tell the truth about them.

The project's guiding ethos: pinned + checksummed acquisition of every external artifact,
no false-green test results, and "do it right" over "ship fast."

## Tech stack (decided)

- **Python 3.12** (single-minor pin — MPS/wheel matching is version-sensitive), **uv-managed**,
  `src/` layout.
- Runtime deps, hard-pinned: `pedalboard==0.9.23`, `librosa==0.10.2`, `numpy==1.26.4`,
  `soundfile==0.12.1`, `mido==1.3.2`, `pydantic>=2.7,<3`.
- Perception deps (local Qwen2-Audio runtime) are shipped as the optional extra
  `sonoscope[perception]` (`pyproject.toml`), so the deterministic core installs without the
  ~16 GB model stack.
- **Pins are law.** Every external artifact (deps, Surge XT binary + factory content, model
  weights, corpus items) is version-pinned and sha256-verified; drift is a **hard fail**.
  `uv.lock` is committed and fully hashed; the pinned versions are re-exported in
  `src/sonoscope/pins.py` so `doctor`/CI can hard-fail on runtime drift.

## Architecture invariants

Four decoupled layers behind stable interfaces (design §2.1):

- **Deterministic Librosa layer is GROUND TRUTH.** It is always on and drives all tripwires,
  deltas, and any automated gating. This is the anti-hallucination backbone.
- **Perception (LALM) output is LABELED ADVISORY and is NEVER fatal.** A perception crash or
  timeout degrades to deterministic-only (`status:"error"`, loop continues, exit 0). It is
  structurally separated from ground truth in the JSON and carries a disclaimer.
- **`RenderBackend` and `PerceptionAdapter` are pluggable interfaces.** pedalboard->VST3 is the
  v1 backend; a native CLAP backend and a cloud perception adapter drop in without a schema
  change.
- **Never hardcode plugin parameter names.** Param names come from runtime `backend.probe()`
  introspection; specs address params `by_name` (preferred) or `by_index` (fallback).
- **Every render runs isolated in a spawned subprocess** so a plugin/JUCE crash cannot take
  down the long-lived process.
- The Pydantic models under `src/sonoscope/schema/` are the **single source of truth** for the
  versioned JSON contract; the JSON Schema is generated from them, never hand-maintained.

## Build / run

```bash
uv sync                          # install the deterministic core
# uv sync --extra perception     # perception extra (local Qwen2-Audio runtime)
uv run pytest                    # full suite
uv run pytest -m "not integration"   # default/unit run (no external artifacts)
uv run pytest -m integration     # integration run (Surge XT / test plugin / model present)
uv run sonoscope ...             # CLI entrypoint
```

## Testing discipline (BINDING)

Green-mirage test discipline, expressed in pytest:

- **RED + GREEN for every check.** Every check / tripwire (silence, NaN/Inf, clip, denormal,
  DC offset, feature-delta, nondeterminism-floor) MUST ship a **RED test** proving it catches a
  real failure AND a **GREEN test** proving healthy input passes.
- **The green-mirage rule:** a check that only ever passes proves nothing. A check without a
  paired RED test is treated as unverified.
- **Exact-equality assertions only (Level 4+):** `assert result == expected` — compare full
  objects / exact values / exact enum verdicts. **Forbidden:** substring matches, length-only
  checks, `assert x` truthiness, `in` membership, `mock.ANY` / `ANY` on the value under test.
- **Integration tests are marked `@pytest.mark.integration`** and skip (with an explicit reason
  string, never a silent pass) when their external artifact is absent — Surge XT, a test
  plugin, or the perception model. Default/unit CI runs `pytest -m "not integration"`; the
  integration suite is invoked explicitly on a provisioned machine. `--strict-markers` is on,
  so the marker must stay registered.

## Gotchas (macOS Apple Silicon)

- **Python 3.12 comes from uv**, not necessarily on `PATH` as `python3.12`. Use `uv run` /
  `uv sync`; do not assume a system interpreter.
- **A mirror-configured uv silently rewrites `uv.lock`.** If `~/.config/uv/uv.toml`
  points at a local index (e.g. proxpi), a plain `uv run` rewrites every registry URL
  in the committed, fully-hashed `uv.lock` to the mirror — no error, no output, just a
  dirty tree. Pins are law, so committing that is a real defect. Prefix uv commands
  with `UV_NO_CONFIG=1 UV_DEFAULT_INDEX=https://pypi.org/simple`, and check
  `git status --porcelain uv.lock` is empty before committing; restore with
  `git checkout -- uv.lock` (that exact path, never a bare `.`).
- **MPS lacks float64.** Keep deterministic analysis on CPU / librosa; the model path may use
  MPS but validate MPS-vs-CPU numerics and keep a CPU fallback.
- **pedalboard has no CLAP loader** — it hosts a wrapped VST3/AU, which is why sonoscope
  consumes CLAP-wrapper VST3 output for CLAP-native plugins.
- **Some plugins tie state to the GUI thread.** Handle `raw_state` carefully: re-inject only
  when its stamped hash matches the plugin `binary_sha256`, else hard-error (never proceed to
  silent default state). **v1 raw_state scope:** the `PedalboardVST3Backend` implements
  only `raw_state` re-injection + hash-stamp validation. The one-time interactive **capture**
  tooling (open on the main thread, `show_editor()`, dump state stamped to `binary_sha256`) is
  **DEFERRED and NOT built in v1** — Surge XT renders non-silent headless (−22 dBFS with the
  default init patch), so no GUI init is required. If a future plugin genuinely needs GUI init,
  capture becomes a separate operator-driven task, never silently built inside the backend.
- **Surge XT install needs `sudo`** (`installer -pkg`). This step is operator-driven — only
  verify/test steps run unattended.
- **macOS Surge XT install locations (dev convenience).** The pinned Surge XT (currently 1.3.4)
  drops the VST3 at `/Library/Audio/Plug-Ins/VST3/Surge XT.vst3` — this is what sonoscope renders
  via pedalboard — and the standalone CLI at `/Applications/Surge XT.app/Contents/MacOS/surge-xt-cli`.
  sonoscope discovers/pins the plugin via the manifest (not a hard-coded path);
  `scripts/verify_surge_xt.sh` confirms the install matches `pins/surge_xt.manifest.toml`.

## Out of v1 scope (deferred to v2)

inverse-synth (param recovery), cloud (Gemini) perception adapter, native CLAP render
backend, MIDI-stream QA, and an MCP wrapper are **out of v1 scope**. The cloud adapter is a
conditionally-promoted contingency only if the local-model probe gate fails and the maintainer
approves it.

## Perception grounding (v1)

Perception grounding is **`advisory-freetext` only** in v1 — labeled with the disclaimer,
never ground truth. **Structured-vocab grounding is NOT built in v1; it is DEFERRED to v2.**

A feasibility probe showed Qwen2-Audio's structured discrimination is **mixed, not cleanly
discriminating**. Per-term reliability of the three candidate vocab terms:

- **`brightness` ↔ `spectral_centroid_hz`: reliable** — the one axis cleanly grounded.
- **`noisiness` ↔ `spectral_flatness`: weak** — directionally correct, magnitude soft.
- **`dynamics` / loudness: unreliable / failed** — Qwen has no absolute-loudness reference, so
  loudness stays deterministic ground truth via RMS (never a Qwen-grounded descriptor).

Structured-vocab grounding is built only if a probe flags structured-vocab as discriminating;
it did not, so it is correctly deferred. **v2 path:** if the structured axes prove out (esp.
brightness), author the full descriptor vocabulary + 1:1 `grounding_map` (design §3.3 / §10.1
S3) and build the `grounding="structured-vocab"` path in `qwen_local.py`.

## PR review bots — sequence, don't parallelize

Two review bots run on PRs here:
- **`gemini-code-assist[bot]`** — auto-reviews on PR open and on push; re-trigger with a
  `/gemini review` comment. (Being sunset in 2026.)
- **`axiomantic-momus[bot]`** — reviews on PR open/reopen; re-trigger with an `/ai-review`
  comment. Note: the comment-triggered run reports via its status **comment**, not the PR
  `review` status check (that check only updates on open/reopen).

**RULE:** when running the PR review cycle, take **one** bot through to completion (all
findings addressed, bot clean/approved) **before** starting the other — do not run both cycles
concurrently. Default order: gemini clean **first**, then momus. Exception: if one bot is
broken/unavailable, proceed with the other.

Rationale: concurrent bot cycles cause overlapping re-review churn and wasted cycles; sequencing
keeps each cycle clean and the diff stable per bot.
