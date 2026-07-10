"""Param/MIDI spec resolver (Task E2, design §6.2, I8).

Validates a versioned :class:`~sonoscope.spec.Spec` against a probed
:class:`~sonoscope.backends.base.PluginInfo` and produces a fully-resolved param
vector plus a stable ``resolved_sha256`` (the R2/iterate delta key, design §3.2).

Hard-fail contract (design §6.2 I8 — "any failure is a hard error, never a
silent no-op or clamp"). Every rejection raises the C5
:class:`~sonoscope.errors.InputError` (exit code 2, ``component="resolver"``):

- ``SPEC_VERSION_UNSUPPORTED`` / ``SPEC_VERSION_MALFORMED`` — unknown or
  unparseable ``spec_version`` MAJOR.
- ``PARAM_UNKNOWN_NAME`` / ``PARAM_AMBIGUOUS_NAME`` — a ``by_name`` reference
  absent from (or duplicated in) the probed params.
- ``PARAM_UNKNOWN_INDEX`` / ``PARAM_INDEX_MALFORMED`` — a ``by_index`` reference
  not on the plugin, or a non-integer index key.
- ``PARAM_VALUE_OUT_OF_RANGE`` — a value outside the normalized ``[0.0, 1.0]``
  domain (this also rejects NaN/Inf, whose comparisons are all false).
- ``PARAM_VALUE_NOT_ON_STEP`` — a bool/stepped value that does not land on a
  valid normalized step (no silent snap to nearest).
- ``PARAM_STEPPED_MALFORMED`` — a ``stepped`` param whose ``num_steps`` is
  missing/< 2 (cannot form step positions).
- ``PARAM_CONFLICTING_REF`` — the same param index addressed twice (by_name and
  by_index, or two names) with no single winner.

Consistency notes:

- ``expected_audio`` derivation REUSES D3's
  :func:`sonoscope.features.tripwires.derive_expected_audio` — the single source
  of truth for the §4.4 derivation table and the spec-override rule — so the
  resolver and the tripwire evaluator can never diverge.
- The value domain and step math live only here; the resolver never consults
  D1's reproducibility tolerance (``features.tolerance``), which governs a
  different concern (bit-repro), per that module's documented scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sonoscope.backends.base import ParamInfo, ParamKind, PluginInfo
from sonoscope.errors import InputError
from sonoscope.features.tripwires import derive_expected_audio
from sonoscope.schema.models import PatchClass
from sonoscope.spec import SPEC_VERSION, RenderSpec, Spec, StimulusSpec

_COMPONENT = "resolver"

#: Absolute tolerance for "does this value land on a normalized step?" — a
#: step-snap check for bool/stepped params only. Intentionally distinct from
#: D1's reproducibility tolerance (``features.tolerance``), which governs
#: bit-repro comparisons, a different concern (design §4.2 I3).
STEP_TOL = 1e-9


@dataclass(frozen=True)
class ResolvedParam:
    """One resolved param: its probed identity plus the validated normalized
    value. For stepped params ``value`` is the canonical step position."""

    index: int
    name: str
    kind: ParamKind
    value: float


@dataclass(frozen=True)
class ResolvedSpec:
    """The fully-resolved spec. ``params`` is index-sorted; ``resolved_sha256``
    hashes the resolved ``(index, value)`` vector (design §3.2 delta key)."""

    spec_version: str
    stimulus: StimulusSpec
    patch_class: PatchClass
    expected_audio: bool
    params: tuple[ResolvedParam, ...]
    render: RenderSpec
    resolved_sha256: str


def _input_error(code: str, message: str, detail: dict) -> InputError:
    return InputError(code, message, detail=detail, component=_COMPONENT)


def _check_spec_version(spec_version: str) -> None:
    """Reject an unknown MAJOR (design §6.2 "unknown major ⇒ hard fail")."""
    try:
        major = int(spec_version.split(".")[0])
    except (ValueError, IndexError):
        raise _input_error(
            "SPEC_VERSION_MALFORMED",
            f"spec_version {spec_version!r} is not a parseable semver",
            {"spec_version": spec_version},
        ) from None
    current_major = int(SPEC_VERSION.split(".")[0])
    if major != current_major:
        raise _input_error(
            "SPEC_VERSION_UNSUPPORTED",
            f"unsupported spec_version major {major}; this build understands "
            f"major {current_major} (spec_version {SPEC_VERSION})",
            {"requested_major": major, "supported_major": current_major},
        )


def _resolve_step(param: ParamInfo, value: float, num_steps: int) -> float:
    """Resolve a bool/stepped value to its canonical step position, or hard-fail.

    Valid positions are ``i / (num_steps - 1)`` for ``i`` in ``0..num_steps-1``.
    The input must already sit on a step (within :data:`STEP_TOL`); a between-step
    value is a hard error, never a silent snap to nearest.
    """
    idx = round(value * (num_steps - 1))
    canonical = idx / (num_steps - 1)
    if abs(value - canonical) > STEP_TOL:
        raise _input_error(
            "PARAM_VALUE_NOT_ON_STEP",
            f"param {param.name!r} (index {param.index}) value {value} does not "
            f"resolve to a valid step of {num_steps} (nearest {canonical})",
            {
                "name": param.name,
                "index": param.index,
                "kind": param.kind,
                "value": value,
                "num_steps": num_steps,
                "nearest_step": canonical,
            },
        )
    return canonical


def _resolve_value(param: ParamInfo, value: float) -> float:
    """Validate + resolve one normalized value for ``param`` (I8), or hard-fail."""
    # I8 domain: a plain number in [0, 1]. NaN/Inf fail these comparisons and are
    # rejected here rather than clamped.
    if not (0.0 <= value <= 1.0):
        raise _input_error(
            "PARAM_VALUE_OUT_OF_RANGE",
            f"param {param.name!r} (index {param.index}) value {value} is outside "
            "the normalized [0.0, 1.0] domain",
            {
                "name": param.name,
                "index": param.index,
                "kind": param.kind,
                "value": value,
            },
        )
    if param.kind == "float":
        return float(value)
    if param.kind == "bool":
        # A boolean is the two-step case: valid positions {0.0, 1.0}.
        return _resolve_step(param, value, num_steps=2)
    # stepped
    if param.num_steps is None or param.num_steps < 2:
        raise _input_error(
            "PARAM_STEPPED_MALFORMED",
            f"stepped param {param.name!r} (index {param.index}) has num_steps="
            f"{param.num_steps}; need >= 2 to form step positions",
            {
                "name": param.name,
                "index": param.index,
                "num_steps": param.num_steps,
            },
        )
    return _resolve_step(param, value, param.num_steps)


def _record(
    resolved: dict[int, ResolvedParam], param: ParamInfo, value: float
) -> None:
    """Record a resolved param, hard-failing on a conflicting double reference."""
    if param.index in resolved:
        raise _input_error(
            "PARAM_CONFLICTING_REF",
            f"param index {param.index} ({param.name!r}) is referenced more than "
            "once (by_name and/or by_index); refusing to silently overwrite",
            {"index": param.index, "name": param.name},
        )
    resolved[param.index] = ResolvedParam(
        index=param.index,
        name=param.name,
        kind=param.kind,
        value=value,
    )


def _resolved_sha256(vector: tuple[ResolvedParam, ...]) -> str:
    """Stable hash of the fully-resolved param vector (design §3.2 delta key).

    Canonicalizes to an index-sorted ``[[index, value], ...]`` array with compact
    separators so identical resolved vectors hash identically and any changed
    value changes the hash.
    """
    canonical = json.dumps(
        [[p.index, p.value] for p in vector], separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve(spec: Spec, plugin_info: PluginInfo) -> ResolvedSpec:
    """Resolve a spec against a probed PluginInfo (design §6.2).

    Validates ``spec_version``, every ``by_name``/``by_index`` reference, and
    every value against the I8 ``[0, 1]`` domain (and step validity), then
    derives ``expected_audio`` (reusing D3) and computes ``resolved_sha256``.
    Any validation failure is a hard :class:`~sonoscope.errors.InputError`
    (exit 2), never a silent no-op or clamp.
    """
    _check_spec_version(spec.spec_version)

    # Build lookup maps; detect duplicate probed names so a by_name reference to
    # an ambiguous name fails loudly rather than binding an arbitrary index.
    name_to_params: dict[str, list[ParamInfo]] = {}
    index_map: dict[int, ParamInfo] = {}
    for p in plugin_info.params:
        name_to_params.setdefault(p.name, []).append(p)
        index_map[p.index] = p

    resolved: dict[int, ResolvedParam] = {}

    for name, value in spec.params.by_name.items():
        matches = name_to_params.get(name, [])
        if not matches:
            raise _input_error(
                "PARAM_UNKNOWN_NAME",
                f"param name {name!r} is not present on the plugin "
                f"({plugin_info.name!r}); names come from probe(), never hardcoded",
                {"name": name, "plugin": plugin_info.name},
            )
        if len(matches) > 1:
            raise _input_error(
                "PARAM_AMBIGUOUS_NAME",
                f"param name {name!r} matches {len(matches)} probed params; "
                "address it by_index to disambiguate",
                {"name": name, "indices": [m.index for m in matches]},
            )
        param = matches[0]
        _record(resolved, param, _resolve_value(param, value))

    for index_key, value in spec.params.by_index.items():
        try:
            index = int(index_key)
        except ValueError:
            raise _input_error(
                "PARAM_INDEX_MALFORMED",
                f"by_index key {index_key!r} is not a base-10 integer index",
                {"index_key": index_key},
            ) from None
        param = index_map.get(index)
        if param is None:
            raise _input_error(
                "PARAM_UNKNOWN_INDEX",
                f"param index {index} is not present on the plugin "
                f"({plugin_info.name!r})",
                {"index": index, "plugin": plugin_info.name},
            )
        _record(resolved, param, _resolve_value(param, value))

    vector = tuple(sorted(resolved.values(), key=lambda rp: rp.index))
    expected_audio = derive_expected_audio(
        spec.stimulus.kind, spec.expected_audio
    )

    return ResolvedSpec(
        spec_version=spec.spec_version,
        stimulus=spec.stimulus,
        patch_class=spec.patch_class,
        expected_audio=expected_audio,
        params=vector,
        render=spec.render,
        resolved_sha256=_resolved_sha256(vector),
    )
