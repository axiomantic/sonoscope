"""C2 window-length resolution (design §7). resolve_window_samples lives in
midi_input.py, co-located with _unit_to_samples (called directly, same module)."""
import pytest

from sonoscope.midi_input import resolve_window_samples


def test_no_slice_uses_full_duration():
    assert resolve_window_samples(
        None, None, None,
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 96000


def test_sliced_explicit_length_samples():
    assert resolve_window_samples(
        0.0, 48000.0, "samples",
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 48000


def test_sliced_length_seconds_unit():
    assert resolve_window_samples(
        0.0, 1.0, "seconds",
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 48000


def test_sliced_length_beats_unit():
    # 2 beats @120bpm == 1.0 s == 48000 samples.
    assert resolve_window_samples(
        0.0, 2.0, "beats",
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 48000


def test_sliced_length_none_to_end():
    assert resolve_window_samples(
        24000.0, None, "samples",
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 96000 - 24000


def test_offset_past_end_clamps_to_zero():
    assert resolve_window_samples(
        120000.0, None, "samples",
        sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
    ) == 0


def test_length_set_without_unit_raises_value_error():
    # -O strips asserts; the missing-unit guard must raise ValueError, not assert.
    # offset non-None so the consolidated (offset-is-set) guard is reached.
    with pytest.raises(ValueError, match="unit must be provided when offset is set"):
        resolve_window_samples(
            0.0, 48000.0, None,
            sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
        )


def test_offset_to_end_without_unit_raises_value_error():
    # offset set, length None, unit None: the offset-to-end branch must also
    # validate unit rather than silently mis-interpreting offset as ticks.
    with pytest.raises(ValueError, match="unit must be provided when offset is set"):
        resolve_window_samples(
            24000.0, None, None,
            sample_rate=48000, tempo_bpm=120.0, full_duration_samples=96000,
        )
