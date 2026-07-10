"""C2 MIDI deriver (design §5/§6/§9). Shared helpers _ev/_meta defined here."""
from sonoscope.descriptors.midi_deriver import (
    MIDI_DERIVER_THRESHOLDS,
    MIDI_DERIVER_VERSION,
    derive_midi_descriptors,
)
from sonoscope.descriptors.thresholds import thresholds_sha256
from sonoscope.schema.models import (
    DescriptorsLibrary,
    MeasuredDescriptor,
    MidiCaptureMeta,
    MidiEvent,
)


def _ev(t_samples, typ, note, velocity, channel=0):
    """Inline synthetic MidiEvent (t_ticks is ignored by the deriver)."""
    return MidiEvent(
        t_samples=t_samples,
        t_ticks=0,
        type=typ,
        channel=channel,
        note=note,
        velocity=velocity,
    )


def _meta(sample_rate=48000):
    """Minimal valid MidiCaptureMeta. The deriver reads ONLY .sample_rate;
    all other fields are filler to satisfy pydantic validation."""
    return MidiCaptureMeta(
        sample_rate=sample_rate,
        block_size=512,
        duration_samples=0,   # deriver ignores; window_samples is passed separately
        tempo_bpm=120.0,
        start_position_beats=0.0,
        duration_beats=0.0,
        tsig_num=4,
        tsig_den=4,
        source="file",
    )


_PINNED_EMPTY_DIGEST = (
    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
)


def _row(term, value, metric):
    return MeasuredDescriptor(
        term=term,
        value=value,
        metric=metric,
        direction="value",
        threshold=None,
        estimated=False,
        confidence=None,
    )


def test_provenance_constants_exact():
    assert MIDI_DERIVER_VERSION == "midi-1.0.0"
    assert MIDI_DERIVER_THRESHOLDS == {}
    assert thresholds_sha256(MIDI_DERIVER_THRESHOLDS) == _PINNED_EMPTY_DIGEST


def test_empty_input_emits_six_zero_rows():
    block = derive_midi_descriptors([], _meta(sample_rate=48000), window_samples=96000)
    assert block.measured == [
        _row("note-density", 0.0, "notes_per_second"),
        _row("register", 0.0, "mean_note"),
        _row("pitch-range", 0.0, "note_span_semitones"),
        _row("polyphony", 0.0, "max_concurrent_notes"),
        _row("velocity-dynamics", 0.0, "velocity_std"),
        _row("ioi", 0.0, "median_ioi_seconds"),
    ]
    assert block.hybrid == []
    assert block.advisory == []
    assert block.library == DescriptorsLibrary(
        thresholds_sha256=_PINNED_EMPTY_DIGEST,
        deriver_version="midi-1.0.0",
        advisory_coverage=None,
        advisory_dropped=None,
    )


def test_emission_order_matches_vocab():
    from sonoscope.descriptors.vocab import MIDI_TERM_ORDER

    block = derive_midi_descriptors([], _meta(), window_samples=0)
    assert [m.term for m in block.measured] == list(MIDI_TERM_ORDER)


def _measured(block, term):
    return [m for m in block.measured if m.term == term]


# ── note-density (Fixture 2/3/5-map §10.3 rows 5,6) ──
def test_note_density_green_two_onsets_over_two_seconds():
    events = [
        _ev(0, "note_on", 60, 80), _ev(24000, "note_off", 60, 0),
        _ev(48000, "note_on", 62, 80), _ev(72000, "note_off", 62, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "note-density") == [
        _row("note-density", 2.0 / 2.0, "notes_per_second")  # 2 onsets / 2.0 s == 1.0
    ]


def test_note_density_red_new_onset_raises_density():
    # RED vs the zero-stub, self-contrasting with the dedup test below (1 unique
    # onset -> 0.5): TWO distinct t_samples -> 2 onsets / 2.0 s == 1.0.
    events = [_ev(0, "note_on", 60, 80), _ev(12000, "note_on", 62, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "note-density")[0].value == 2.0 / 2.0  # 2 onsets / 2.0 s == 1.0


def test_note_density_dedup_same_onset_does_not_change():
    # Onset-dedup proof (§6.9): a 2nd note at an EXISTING t_samples -> still 1 onset.
    events = [_ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "note-density")[0].value == 1.0 / 2.0  # 1 onset / 2.0 s


def test_note_density_zero_window_guard():
    events = [_ev(0, "note_on", 60, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=0)
    assert _measured(block, "note-density")[0].value == 0.0  # no ZeroDivisionError


# ── register (§6.3) ──
def test_register_green_mean_note():
    events = [_ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "register") == [
        _row("register", 62.0, "mean_note")  # (60+64)/2
    ]


def test_register_red_changing_a_note_shifts_mean():
    events = [_ev(0, "note_on", 60, 80), _ev(0, "note_on", 66, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "register")[0].value == 63.0  # (60+66)/2, != 62.0


# ── pitch-range (§6.4) ──
def test_pitch_range_green_span():
    events = [_ev(0, "note_on", 48, 80), _ev(0, "note_on", 72, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "pitch-range") == [
        _row("pitch-range", 24.0, "note_span_semitones")
    ]


def test_pitch_range_single_note_is_zero():
    block = derive_midi_descriptors(
        [_ev(0, "note_on", 60, 80)], _meta(48000), window_samples=96000
    )
    assert _measured(block, "pitch-range")[0].value == 0.0


def test_pitch_range_red_raising_max_widens_span():
    events = [_ev(0, "note_on", 48, 80), _ev(0, "note_on", 84, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "pitch-range")[0].value == 36.0  # != 24.0


# ── polyphony (§6.5) ──
def test_polyphony_green_chord_is_three():
    events = [
        _ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 90), _ev(0, "note_on", 67, 100),
        _ev(24000, "note_off", 60, 0), _ev(24000, "note_off", 64, 0),
        _ev(24000, "note_off", 67, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony") == [
        _row("polyphony", 3.0, "max_concurrent_notes")
    ]


def test_polyphony_red_drop_one_chord_note():
    events = [
        _ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 90),
        _ev(24000, "note_off", 60, 0), _ev(24000, "note_off", 64, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 2.0  # 3.0 -> 2.0


def test_polyphony_same_pitch_retrigger_stacks():
    # §6.5: a 2nd note_on before its off stacks a 2nd voice (+1), E1-consistent.
    events = [
        _ev(0, "note_on", 60, 80), _ev(100, "note_on", 60, 80),
        _ev(200, "note_off", 60, 0), _ev(300, "note_off", 60, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 2.0


def test_polyphony_window_boundary_dangling_off_seeds():
    # Fixture 5: note_off with no in-window on (slice cut the START) -> seed counts it.
    events = [_ev(1000, "note_off", 60, 0)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 1.0  # seed=1, sweep 1 -> 0


def test_polyphony_window_boundary_stuck_note():
    # Fixture 6: note_on with no in-window off (slice cut the END) -> sounds to end.
    events = [_ev(0, "note_on", 60, 80)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 1.0


def test_polyphony_legato_boundary_is_one():
    # Fixture 10(a): A_off@100 / B_on@100 -> off-before-on hands off -> peak 1.
    events = [
        _ev(0, "note_on", 60, 80), _ev(100, "note_off", 60, 0),
        _ev(100, "note_on", 62, 80), _ev(200, "note_off", 62, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 1.0


def test_polyphony_zero_length_over_sustained_is_two():
    # Fixture 10(b): V sustained 0..200; Z zero-length @100 (on72@100, off72@100).
    #
    # DOCUMENTED SEMANTICS UPDATE (Gemini seed/sweep consistency): value was 1.0
    # under the old inconsistent tie-break; it is 2.0 under the consistent
    # off-before-on convention. From windowed events alone, "off72@100, on72@100"
    # is INDISTINGUISHABLE from a pre-window dangling off (a voice sounding since
    # window start, ending at t=100) followed by a fresh note starting at t=100.
    # The off-before-on convention resolves the ambiguity uniformly: a note_off
    # at t closes any voice open just before t, and an unmatched off implies a
    # voice sounding since window start. So off72@100 (no note72 open before it)
    # seeds one such voice, which overlaps sustained V(60) -> peak 2. This is the
    # SAME rule that makes the pre-window-retrigger cases correct; treating the
    # zero-length note as momentarily sounding is the consistent, defensible read.
    events = [
        _ev(0, "note_on", 60, 80), _ev(200, "note_off", 60, 0),
        _ev(100, "note_on", 72, 80), _ev(100, "note_off", 72, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony")[0].value == 2.0


def test_polyphony_prewindow_dangling_off_coincident_retrigger_counts_two():
    # Gemini HIGH: a dangling note_off (its matching note_on is PRE-window; the
    # slice cut the START) coincident at the same t_samples with a same-(channel,
    # note) legato retrigger, plus a second voice starting at the boundary.
    #
    # _dangling_off_seed pairs on-before-off (_pairing_key), so the in-window
    # retrigger note_on ABSORBS the dangling note_off -> seed=0 (the pre-window
    # voice is NOT counted). But _max_concurrent's sweep processes note_off
    # BEFORE note_on (_sweep_key) at the coincident t, so running dips to -1 and
    # every later event is undercounted by 1.
    #
    # Truth: note60-pre sounds from before the window up to t=1000 (1 voice); at
    # t=1000 it hands off (legato, off-before-on) to note60-retrig while note67
    # also starts -> 2 simultaneous voices from t=1000 onward. Correct peak = 2.
    events = [
        _ev(1000, "note_off", 60, 0),   # dangling off: matching note_on pre-window
        _ev(1000, "note_on", 60, 80),   # legato retrigger, same (channel, note)
        _ev(1000, "note_on", 67, 90),   # second voice starting at the boundary
        _ev(2000, "note_off", 67, 0),
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony") == [
        _row("polyphony", 2.0, "max_concurrent_notes")
    ]


def test_polyphony_two_prewindow_voices_one_coincident_retrigger_counts_two():
    # Gemini HIGH (seed/sweep tie-break consistency): TWO voices are sounding at
    # window start. Note 60's pre-window voice hands off (legato) to a coincident
    # same-(channel, note) retrigger at t=1000; note 67's pre-window voice sustains
    # until its off at t=2000. Both sound simultaneously across [1000, 2000) -> 2.
    #
    # This is the case the old code undercounted: _dangling_off_seed paired
    # on-before-off, so note60's in-window retrigger ABSORBED note60's dangling
    # off (seed counted only note67 -> seed=1). The sweep (off-before-on) then
    # never dipped below 0 (note60's retrigger kept running non-negative), so the
    # negative-dip self-correction never fired and the peak came out 1 (WRONG).
    # With seed and sweep sharing the off-before-on order, seed=2 by construction.
    events = [
        _ev(1000, "note_off", 60, 0),   # dangling off: note 60's on is pre-window
        _ev(1000, "note_on", 60, 80),   # same-(channel, note) retrigger at same t
        _ev(2000, "note_off", 67, 0),   # dangling off: note 67's on is pre-window
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "polyphony") == [
        _row("polyphony", 2.0, "max_concurrent_notes")
    ]


# ── velocity-dynamics (§6.7) ──
def test_velocity_single_note_is_zero():
    block = derive_midi_descriptors(
        [_ev(0, "note_on", 60, 80)], _meta(48000), window_samples=96000
    )
    assert _measured(block, "velocity-dynamics") == [
        _row("velocity-dynamics", 0.0, "velocity_std")
    ]


def test_velocity_green_pstdev():
    import statistics

    vels = [80, 90, 100, 40, 110]
    events = [_ev(0, "note_on", 60, v) for v in vels]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "velocity-dynamics")[0].value == statistics.pstdev(vels)


def test_velocity_red_all_equal_is_zero():
    events = [_ev(0, "note_on", 60, 90), _ev(0, "note_on", 64, 90)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "velocity-dynamics")[0].value == 0.0


# ── ioi (§6.8) ──
def test_ioi_green_median_seconds():
    events = [
        _ev(0, "note_on", 60, 80),
        _ev(48000, "note_on", 62, 80),   # +1.0 s
        _ev(72000, "note_on", 64, 80),   # +0.5 s
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "ioi") == [
        _row("ioi", 0.75, "median_ioi_seconds")  # median([1.0, 0.5])
    ]


def test_ioi_single_onset_is_zero():
    block = derive_midi_descriptors(
        [_ev(0, "note_on", 60, 80)], _meta(48000), window_samples=96000
    )
    assert _measured(block, "ioi")[0].value == 0.0


def test_ioi_red_moving_onset_shifts_median():
    # RED that genuinely trips: onsets 0/48000/96000 -> deltas [1.0, 1.0] ->
    # median == 1.0, which differs from the GREEN median (0.75).
    events = [
        _ev(0, "note_on", 60, 80),
        _ev(48000, "note_on", 62, 80),   # +1.0 s
        _ev(96000, "note_on", 64, 80),   # +1.0 s
    ]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "ioi")[0].value == 1.0  # median([1.0, 1.0]) != 0.75


def test_ioi_red_collapse_to_one_onset_is_zero():
    events = [_ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 80)]  # 1 unique onset
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "ioi")[0].value == 0.0


# ── multichannel incl. percussion (Fixture 4, §6.3/§6.4) ──
def test_register_spans_all_channels_incl_percussion():
    events = [_ev(0, "note_on", 36, 80, channel=9), _ev(0, "note_on", 84, 80, channel=0)]
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "register")[0].value == 60.0          # (36+84)/2
    assert _measured(block, "pitch-range")[0].value == 48.0       # 84-36


# ── offvel0 normalization (Fixture 7, §6.1) ──
def test_offvel0_note_on_excluded_from_note_ons():
    # A note_on with velocity 0 is a note_off; it must NOT count as an onset,
    # must NOT affect register, and must NOT enter velocity pstdev.
    events = [_ev(0, "note_on", 60, 80), _ev(24000, "note_on", 60, 0)]  # 2nd is off
    block = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    assert _measured(block, "note-density")[0].value == 1.0 / 2.0   # 1 onset
    assert _measured(block, "register")[0].value == 60.0            # only the vel80 on
    assert _measured(block, "velocity-dynamics")[0].value == 0.0    # single velocity
    assert _measured(block, "polyphony")[0].value == 1.0            # on then off -> peak 1


# ── determinism / order-perturbation (Fixture 8, §9) ──
def test_determinism_order_perturbation():
    events = [
        _ev(0, "note_on", 60, 80), _ev(0, "note_on", 64, 90),
        _ev(48000, "note_on", 67, 100), _ev(24000, "note_off", 60, 0),
    ]
    a = derive_midi_descriptors(events, _meta(48000), window_samples=96000)
    b = derive_midi_descriptors(list(reversed(events)), _meta(48000), window_samples=96000)
    assert a.measured == b.measured
    assert a == b
