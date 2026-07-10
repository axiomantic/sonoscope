"""Stimulus-corpus tests (Task E1, design §6.1, R4).

Green-mirage discipline: the drift check ships a RED fixture that provably
catches a mutated corpus byte, paired with a GREEN clean case, plus a
regeneration test that asserts the pinned manifest hashes are byte-for-byte
reproducible from the generator script.

Assertions are exact-equality (exact booleans / exact sha256 strings / exact
name sets) so a constant stub cannot pass.

The drift / missing fixtures operate on a **temp copy** of the committed corpus
(``tmp_path``) so the pinned in-repo corpus is never mutated.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

from sonoscope import corpus

# --- Import the generator script by path (scripts/ is not a package) ---------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATE_CORPUS_PY = _REPO_ROOT / "scripts" / "generate_corpus.py"


def _load_generate_corpus():
    spec = importlib.util.spec_from_file_location(
        "generate_corpus", _GENERATE_CORPUS_PY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass type resolution (which looks the module
    # up in sys.modules) works for a module imported by file path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generate_corpus = _load_generate_corpus()


# The canonical set of corpus item names (manifest keys). Exact-equality set
# assertion below is the green-mirage guard that every spec'd generator exists.
_EXPECTED_ITEM_NAMES = {
    "impulse",
    "sweep",
    "pink_noise",
    "tone",
    "silence",
    "c3_sustain",
    "phrase_4note",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- list_items --------------------------------------------------------------


def test_list_items_covers_all_generators():
    names = {item.name for item in corpus.list_items()}
    assert names == _EXPECTED_ITEM_NAMES


def test_list_items_are_signals_and_midi():
    kinds = {item.name: item.kind for item in corpus.list_items()}
    assert kinds == {
        "impulse": "signal",
        "sweep": "signal",
        "pink_noise": "signal",
        "tone": "signal",
        "silence": "signal",
        "c3_sustain": "midi",
        "phrase_4note": "midi",
    }


# --- verify: clean pass ------------------------------------------------------


def test_verify_clean_passes():
    # GREEN side of the green-mirage pair: the committed, pinned corpus verifies
    # cleanly against its manifest (no drift, no missing files).
    result = corpus.verify()
    assert result.ok is True
    assert result.failures == ()


# --- verify: drift detection (RED-proving) -----------------------------------


def test_verify_detects_drift(tmp_path):
    # Copy the committed corpus so the pinned in-repo files are never mutated.
    corpus_root = tmp_path / "corpus"
    shutil.copytree(corpus.DEFAULT_CORPUS_ROOT, corpus_root)
    manifest_path = corpus_root / "manifest.toml"

    # Mutate exactly one byte of exactly one item (the tone wav's final sample).
    tone = corpus_root / "signals" / "tone_1k_2s.wav"
    raw = bytearray(tone.read_bytes())
    raw[-1] ^= 0x01
    tone.write_bytes(bytes(raw))

    result = corpus.verify(
        corpus_root=corpus_root, manifest_path=manifest_path
    )

    # The whole-corpus verdict is a FAILURE result (structured, not an exit code).
    assert result.ok is False

    by_name = {v.name: v for v in result.items}
    # The mutated item is flagged with the exact hash-mismatch reason.
    assert by_name["tone"].ok is False
    assert by_name["tone"].reason == corpus.REASON_HASH_MISMATCH
    # Every other item still verifies (only the mutated byte drifted).
    assert result.failures == (by_name["tone"],)


def test_verify_detects_missing_file(tmp_path):
    corpus_root = tmp_path / "corpus"
    shutil.copytree(corpus.DEFAULT_CORPUS_ROOT, corpus_root)
    manifest_path = corpus_root / "manifest.toml"

    missing = corpus_root / "midi" / "c3_sustain_2s.mid"
    missing.unlink()

    result = corpus.verify(
        corpus_root=corpus_root, manifest_path=manifest_path
    )
    assert result.ok is False
    by_name = {v.name: v for v in result.items}
    assert by_name["c3_sustain"].ok is False
    assert by_name["c3_sustain"].reason == corpus.REASON_MISSING
    assert by_name["c3_sustain"].actual_sha256 is None


# --- deterministic regeneration ----------------------------------------------


def test_regenerated_matches_manifest(tmp_path):
    # Regenerate every item into a fresh dir and assert each regenerated file's
    # sha256 EXACTLY equals the committed manifest hash (deterministic regen).
    out_dir = tmp_path / "regen"
    generated = generate_corpus.generate_all(out_dir)

    manifest_hashes = {item.name: item.sha256 for item in corpus.list_items()}
    assert {g.name for g in generated} == _EXPECTED_ITEM_NAMES

    for g in generated:
        file_bytes = (out_dir / g.path).read_bytes()
        assert _sha256_bytes(file_bytes) == manifest_hashes[g.name]
        # The generator's self-reported hash also matches its own bytes.
        assert g.sha256 == manifest_hashes[g.name]


def test_regeneration_is_byte_identical(tmp_path):
    # Directly prove determinism (design "Determinism requirement"): two
    # independent generation runs produce byte-identical files for every item.
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    items_a = generate_corpus.generate_all(dir_a)
    items_b = generate_corpus.generate_all(dir_b)

    paths_a = {i.name: i.path for i in items_a}
    paths_b = {i.name: i.path for i in items_b}
    assert paths_a == paths_b

    for name, rel in paths_a.items():
        assert (dir_a / rel).read_bytes() == (dir_b / rel).read_bytes()
