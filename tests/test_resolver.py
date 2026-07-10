"""Param/MIDI spec resolver tests (Task E2, design §6.2, I8).

Green-mirage discipline: every rejection path ships a RED-proving test that
fails if the resolver silently no-ops or clamps, paired with a GREEN case that a
constant stub could not fake. Assertions are exact-equality: exact resolved
values, exact ``resolved_sha256`` equality/inequality, exact error ``code`` /
``exit_code`` / ``component``.

The ``PluginInfo`` fixtures carry KNOWN param names, indices, kinds, and step
counts so a stub that ignores the plugin surface cannot pass: an unknown name is
absent from the fixture, an out-of-range value is a real number the resolver
must reject, and stepped/bool params have exact valid step positions.
"""

from __future__ import annotations

import pytest

from sonoscope.backends.base import ParamInfo, PluginInfo
from sonoscope.errors import InputError
from sonoscope.resolver import ResolvedParam, resolve
from sonoscope.schema.models import ExitCode
from sonoscope.spec import SPEC_VERSION, Spec


# --- Fixtures ----------------------------------------------------------------


def _plugin_info() -> PluginInfo:
    """A synth with KNOWN params: two floats, one bool, one 4-step stepped."""
    return PluginInfo(
        name="TestSynth",
        format="vst3",
        params=[
            ParamInfo(name="Cutoff", index=0, kind="float", num_steps=None, default=0.5),
            ParamInfo(name="Resonance", index=1, kind="float", num_steps=None, default=0.0),
            ParamInfo(name="Bypass", index=2, kind="bool", num_steps=2, default=0.0),
            ParamInfo(name="Waveform", index=3, kind="stepped", num_steps=4, default=0.0),
        ],
        input_channels=0,
        output_channels=2,
        is_instrument=True,
        needs_gui=False,
        latency_samples=0,
    )


def _midi_spec(**overrides) -> Spec:
    """A valid midi-stimulus spec; ``params``/fields overridable per test."""
    base = {
        "spec_version": SPEC_VERSION,
        "stimulus": {"kind": "midi", "ref": "corpus/midi/c3_sustain_2s.mid"},
        "params": {"by_name": {"Cutoff": 0.25}},
    }
    base.update(overrides)
    return Spec.model_validate(base)


# --- Required acceptance tests (exact names) ---------------------------------


def test_unknown_param_name_hard_errors():
    # RED-proving: a name absent from PluginInfo must raise, never silently
    # skip. The fixture has no "Nonexistent Param", so a no-op stub would return
    # an (empty) vector and fail this test.
    spec = _midi_spec(params={"by_name": {"Nonexistent Param": 0.5}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_UNKNOWN_NAME"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_out_of_range_value_hard_errors():
    # RED-proving: 1.5 is out of the I8 [0,1] domain. The resolver must raise,
    # NOT clamp to 1.0 and proceed. A clamping stub would return a value and fail
    # this assertion.
    spec = _midi_spec(params={"by_name": {"Cutoff": 1.5}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_VALUE_OUT_OF_RANGE"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_nan_value_hard_errors():
    # RED-proving: NaN is outside the I8 [0,1] domain and must hard-error. The
    # resolver relies on the ``not (0.0 <= value <= 1.0)`` idiom, where NaN makes
    # both comparisons False → not False → raise. A naive refactor to
    # ``value < 0.0 or value > 1.0`` would silently ACCEPT NaN (both comparisons
    # False → no raise) with the suite green; this test locks against that mirage.
    spec = _midi_spec(params={"by_name": {"Cutoff": float("nan")}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_VALUE_OUT_OF_RANGE"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_inf_value_hard_errors():
    # RED-proving: +inf and -inf are outside the I8 [0,1] domain and must
    # hard-error, never clamp to a boundary. Mirrors the out-of-range case for
    # the non-finite tails so a future domain-check refactor cannot start
    # admitting infinities unnoticed.
    for value in (float("inf"), float("-inf")):
        spec = _midi_spec(params={"by_name": {"Cutoff": value}})
        with pytest.raises(InputError) as exc_info:
            resolve(spec, _plugin_info())
        err = exc_info.value
        assert err.code == "PARAM_VALUE_OUT_OF_RANGE"
        assert err.exit_code == ExitCode.INPUT
        assert err.component == "resolver"


def test_resolved_sha256_stable():
    plugin = _plugin_info()
    spec_a = _midi_spec(params={"by_name": {"Cutoff": 0.25, "Resonance": 0.4}})
    spec_b = _midi_spec(params={"by_name": {"Cutoff": 0.25, "Resonance": 0.4}})
    spec_changed = _midi_spec(
        params={"by_name": {"Cutoff": 0.26, "Resonance": 0.4}}
    )

    sha_a = resolve(spec_a, plugin).resolved_sha256
    sha_b = resolve(spec_b, plugin).resolved_sha256
    sha_changed = resolve(spec_changed, plugin).resolved_sha256

    # Identical specs → EXACTLY equal hash; a single changed value → different.
    assert sha_a == sha_b
    assert sha_a != sha_changed


def test_patch_class_defaults_noisy():
    # Omitted patch_class resolves to the conservative "noisy" regime (§6.2).
    spec = _midi_spec()
    assert "patch_class" not in spec.model_dump(exclude_unset=True)
    resolved = resolve(spec, _plugin_info())
    assert resolved.patch_class == "noisy"


def test_expected_audio_derived_from_silence():
    # A silence stimulus with expected_audio=null derives False (§4.4 table),
    # reusing D3's derivation (single source of truth).
    spec = Spec.model_validate(
        {
            "stimulus": {
                "kind": "silence",
                "ref": "corpus/signals/silence_2s.wav",
            },
        }
    )
    resolved = resolve(spec, _plugin_info())
    assert resolved.expected_audio is False


# --- GREEN pairs + adjacent behavior coverage --------------------------------


def test_happy_path_resolves_named_vector():
    # GREEN: valid names/values resolve to an index-sorted vector with the exact
    # values, and a midi stimulus derives expected_audio True (§4.4).
    spec = _midi_spec(params={"by_name": {"Resonance": 0.4, "Cutoff": 0.25}})
    resolved = resolve(spec, _plugin_info())
    assert resolved.params == (
        ResolvedParam(index=0, name="Cutoff", kind="float", value=0.25),
        ResolvedParam(index=1, name="Resonance", kind="float", value=0.4),
    )
    assert resolved.expected_audio is True


def test_by_index_resolves_against_plugin():
    # GREEN: by_index addresses a param by its integer index (string key).
    spec = _midi_spec(params={"by_name": {}, "by_index": {"1": 0.75}})
    resolved = resolve(spec, _plugin_info())
    assert resolved.params == (
        ResolvedParam(index=1, name="Resonance", kind="float", value=0.75),
    )


def test_unknown_param_index_hard_errors():
    # RED-proving: index 99 is not on the plugin → hard error, no silent skip.
    spec = _midi_spec(params={"by_name": {}, "by_index": {"99": 0.5}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_UNKNOWN_INDEX"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_bool_value_must_be_binary():
    # RED-proving: a bool param accepts only 0.0/1.0. 0.5 does not resolve to a
    # valid step → hard error (no silent snap).
    spec = _midi_spec(params={"by_name": {"Bypass": 0.5}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_VALUE_NOT_ON_STEP"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_bool_value_binary_resolves():
    # GREEN pair: 1.0 is a valid bool step and resolves exactly.
    spec = _midi_spec(params={"by_name": {"Bypass": 1.0}})
    resolved = resolve(spec, _plugin_info())
    assert resolved.params == (
        ResolvedParam(index=2, name="Bypass", kind="bool", value=1.0),
    )


def test_stepped_value_off_step_hard_errors():
    # RED-proving: a 4-step param has steps {0, 1/3, 2/3, 1}. 0.5 lands between
    # steps → hard error (no silent snap to nearest).
    spec = _midi_spec(params={"by_name": {"Waveform": 0.5}})
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_VALUE_NOT_ON_STEP"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_stepped_value_on_step_resolves_to_canonical():
    # GREEN pair: 2/3 is a valid step of a 4-step param and resolves to the exact
    # canonical position.
    spec = _midi_spec(params={"by_name": {"Waveform": 2 / 3}})
    resolved = resolve(spec, _plugin_info())
    assert resolved.params == (
        ResolvedParam(index=3, name="Waveform", kind="stepped", value=2 / 3),
    )


def test_unknown_spec_version_major_hard_errors():
    # An unknown MAJOR is a hard fail (§6.2 "unknown major ⇒ hard fail").
    spec = _midi_spec(spec_version="2.0.0")
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "SPEC_VERSION_UNSUPPORTED"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_conflicting_param_ref_hard_errors():
    # RED-proving: the same param addressed by BOTH name and index must not be
    # silently overwritten — it is a hard conflict error.
    spec = _midi_spec(
        params={"by_name": {"Cutoff": 0.25}, "by_index": {"0": 0.75}}
    )
    with pytest.raises(InputError) as exc_info:
        resolve(spec, _plugin_info())
    err = exc_info.value
    assert err.code == "PARAM_CONFLICTING_REF"
    assert err.exit_code == ExitCode.INPUT
    assert err.component == "resolver"


def test_expected_audio_override_wins():
    # A spec expected_audio override beats the derived default (§4.4), consistent
    # with D3's derive_expected_audio.
    spec = _midi_spec(expected_audio=False)
    resolved = resolve(spec, _plugin_info())
    assert resolved.expected_audio is False
