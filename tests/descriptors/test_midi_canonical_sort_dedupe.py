"""_canonical_sort has ONE shared definition (design-neutral refactor)."""
from sonoscope.schema.models import MidiEvent


def _ev(t, typ, note, ch=0):
    return MidiEvent(t_samples=t, t_ticks=0, type=typ, channel=ch, note=note, velocity=0)


def test_single_shared_definition():
    import sonoscope.backends.midi_capture as cap
    import sonoscope.midi_input as mi

    # Identity: exactly one function object is reused by both modules.
    assert mi._canonical_sort is cap._canonical_sort


def test_class_staticmethod_delegates_to_shared_definition():
    import sonoscope.backends.midi_capture as cap

    # The MidiCaptureBackend staticmethod is a thin wrapper (a distinct object,
    # not an identity rebind), so it DELEGATES to the single module-level
    # definition: both call surfaces produce identical output over the same input.
    sample_events = [
        _ev(100, "note_on", 60), _ev(100, "note_off", 62),
        _ev(0, "note_on", 64), _ev(100, "note_off", 61),
        _ev(50, "note_on", 67), _ev(50, "note_off", 67),
    ]
    expected = [
        _ev(0, "note_on", 64),
        _ev(50, "note_off", 67), _ev(50, "note_on", 67),
        _ev(100, "note_off", 62), _ev(100, "note_off", 61),
        _ev(100, "note_on", 60),
    ]
    wrapper_out = cap.MidiCaptureBackend._canonical_sort(sample_events)
    module_out = cap._canonical_sort(sample_events)
    # Wrapper delegates to the single shared logic: same output, and it is the
    # correct canonical order (off-before-on at coincident t, stable otherwise).
    assert wrapper_out == module_out == expected


def test_tie_break_off_before_on_and_stable():
    from sonoscope.midi_input import _canonical_sort

    events = [
        _ev(100, "note_on", 60), _ev(100, "note_off", 62),
        _ev(0, "note_on", 64), _ev(100, "note_off", 61),
    ]
    ordered = _canonical_sort(events)
    # @0 first; at t=100 both note_offs (input order kept: 62 then 61) precede the on.
    assert ordered == [
        _ev(0, "note_on", 64),
        _ev(100, "note_off", 62), _ev(100, "note_off", 61),
        _ev(100, "note_on", 60),
    ]
