"""C2 MIDI vocab: canonical term source (design §4.3, Fixture 11)."""
from sonoscope.descriptors.vocab import (
    BOUNDED_ADVISORY_VOCAB,
    MIDI_MEASURED_TERMS,
    MIDI_TERM_ORDER,
)

_EXPECTED_ORDER = (
    "note-density",
    "register",
    "pitch-range",
    "polyphony",
    "velocity-dynamics",
    "ioi",
)


def test_midi_term_order_is_frozen_sequence():
    assert MIDI_TERM_ORDER == _EXPECTED_ORDER


def test_midi_measured_terms_set_equality():
    assert MIDI_MEASURED_TERMS == frozenset(
        {
            "note-density",
            "register",
            "pitch-range",
            "polyphony",
            "velocity-dynamics",
            "ioi",
        }
    )


def test_midi_measured_terms_derives_from_order():
    assert MIDI_MEASURED_TERMS == frozenset(MIDI_TERM_ORDER)


def test_midi_terms_disjoint_from_advisory_vocab():
    # Additive: no MIDI term leaks into the existing audio/advisory vocab.
    assert MIDI_MEASURED_TERMS & BOUNDED_ADVISORY_VOCAB == frozenset()
