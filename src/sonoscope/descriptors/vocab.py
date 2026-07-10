"""Bounded advisory vocabulary + curated synonym map (by design, Task T5).

Single source of truth for the cycle-1 advisory vocabulary. ``BOUNDED_ADVISORY_VOCAB``
is FROZEN to exactly 67 terms (by design, F9): affect (Category F, 13), evocative
(Category G, 15), genre/style (Category H, 36), plus three advisory-lean hybrids
(``hypnotic``, ``spacious``, ``wall-of-sound``). The advisory producer's curated map
(this module) and T6's constrained-classifier ``allowed_labels`` both source this set
so they cannot diverge; a structural test asserts the exact membership so drift is a
hard fail.

``CURATED_SYNONYM_MAP`` maps freeform Qwen surface forms onto canonical bounded terms;
every value is a member of ``BOUNDED_ADVISORY_VOCAB`` (structurally tested).
"""

from __future__ import annotations

# --- Category F — affect / mood (13) ----------------------------------------
_AFFECT: tuple[str, ...] = (
    "tense",
    "calm",
    "dark",
    "uplifting",
    "melancholy",
    "aggressive",
    "playful",
    "ominous",
    "serene",
    "anxious",
    "triumphant",
    "nostalgic",
    "euphoric",
)

# --- Category G — evocative / aesthetic (15) ---------------------------------
_EVOCATIVE: tuple[str, ...] = (
    "ethereal",
    "cosmic",
    "dreamy",
    "lush",
    "glassy",
    "retro/vintage",
    "futuristic",
    "campy",
    "cinematic",
    "psychedelic",
    "gritty",
    "warm",
    "organic/synthetic",
    "alien",
    "epic",
)

# --- Category H — genre / style, controlled taxonomy (36) --------------------
# The design's genre tree "…" is CLOSED for C1; additions re-freeze this set.
_GENRE: tuple[str, ...] = (
    "ambient",
    "drone",
    "techno",
    "house",
    "trance",
    "minimal",
    "dnb",
    "jungle",
    "breakbeat",
    "garage",
    "hip-hop",
    "trap",
    "lo-fi",
    "dubstep",
    "rock",
    "indie",
    "punk",
    "post-rock",
    "metal",
    "djent",
    "doom",
    "black",
    "jazz",
    "fusion",
    "swing",
    "bebop",
    "classical",
    "orchestral",
    "film-score",
    "folk",
    "country",
    "pop",
    "synth-pop",
    "noise",
    "glitch",
    "idm",
)

# --- advisory-lean hybrids (3) ----------------------------------------------
_HYBRIDS: tuple[str, ...] = (
    "hypnotic",
    "spacious",
    "wall-of-sound",
)

BOUNDED_ADVISORY_VOCAB: frozenset[str] = frozenset(
    _AFFECT + _EVOCATIVE + _GENRE + _HYBRIDS
)
"""FROZEN 67-term bounded advisory vocabulary (by design, F9)."""


# Freeform surface form -> canonical bounded term. Every value is a member of
# BOUNDED_ADVISORY_VOCAB (structurally tested by test_map_all_values_in_vocab).
# This is the SEED map (by design, examples + worked-example coverage); it grows
# per the living playbook. NON-FROZEN for cycle 1 (F6).
CURATED_SYNONYM_MAP: dict[str, str] = {
    "spacey": "cosmic",
    "outer space": "cosmic",
    "galactic": "cosmic",
    "cosmic": "cosmic",
    "hypnotic": "hypnotic",
    "trance-like": "hypnotic",
    "four on the floor": "techno",
}


# Fixed confidence assigned to every mapped advisory term (Qwen freeform carries
# no per-term confidence in C1; by design). Frozen constant.
ADVISORY_BASE_CONFIDENCE: float = 0.6


# ── C2 MIDI measured value-readouts (additive; audio/advisory terms untouched) ──
# Canonical EMISSION ORDER for the six C2 MIDI measured value-readouts (frozen
# by design). A frozenset is unordered, so this ordered tuple is the authoritative
# emission sequence; the membership frozenset below is DERIVED from it.
MIDI_TERM_ORDER: tuple[str, ...] = (
    "note-density",
    "register",
    "pitch-range",
    "polyphony",
    "velocity-dynamics",
    "ioi",
)

# CONTRACT CAVEAT: MIDI_MEASURED_TERMS holds ONLY gate-eligible (estimated=False)
# MEASURED MIDI terms. The consumer gate treats EVERY member as gate-eligible.
# Any FUTURE inferential MIDI term (estimated=True) MUST go in a SEPARATE set and
# stay OUT of this frozenset (so the gate never wrongly gates an estimated
# term). All six current terms are estimated=False.
MIDI_MEASURED_TERMS: frozenset[str] = frozenset(MIDI_TERM_ORDER)
