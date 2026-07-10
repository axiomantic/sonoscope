"""wav_orchestrator engine tests (design §5–§9). Synthetic wavs written to
tmp_path (no integration marker; keeps the -m "not integration" gate green)."""

import numpy as np
import soundfile as sf

from sonoscope.wav_orchestrator import _read_chunk_native, _read_metadata


def _write_wav(path, data, sr, subtype="PCM_16"):
    # soundfile.write expects frame-major (n_frames, n_channels).
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def test_read_chunk_native_is_channel_major_stereo(tmp_path):
    # RED (C1): distinct L/R content. Frame-major on disk is (n, 2); the seek-read
    # helper MUST return channel-major (2, n). Removing the `.T` returns (n, 2) ->
    # assertion fails, proving the transpose is load-bearing. _read_metadata reads
    # sr/channels/subtype from the header without pulling any audio.
    n = 4096
    left = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    right = np.linspace(0.5, -0.5, n, dtype=np.float32)
    frame_major = np.stack([left, right], axis=1)  # (n, 2)
    wav = _write_wav(tmp_path / "stereo.wav", frame_major, 44100)

    with sf.SoundFile(str(wav)) as f:
        n_native, native_sr, n_channels, subtype = _read_metadata(f)
        audio = _read_chunk_native(f, 0, n_native)

    assert n_native == n
    assert native_sr == 44100
    assert n_channels == 2
    assert audio.shape == (2, n)         # channel-major, NOT (n, 2)
    assert audio.dtype == np.float32
    assert subtype == "PCM_16"
    # Channel 0 rises, channel 1 falls -> distinguishes a real transpose from luck.
    assert audio[0, 0] < audio[0, -1]
    assert audio[1, 0] > audio[1, -1]


import pytest

from sonoscope.errors import InputError
from sonoscope.wav_orchestrator import AudioSlice, _resolve_region


def test_resolve_region_whole_file_when_no_slice():
    off, length = _resolve_region(n_native=44100, native_sr=44100, slice_spec=None)
    assert (off, length) == (0, 44100)


def test_resolve_region_seconds_half_open_rounded():
    # 0.5s offset, 0.25s length at 44100 -> round(0.5*44100)=22050, round(0.25*44100)=11025
    s = AudioSlice(offset=0.5, length=0.25, unit="seconds")
    off, length = _resolve_region(n_native=44100, native_sr=44100, slice_spec=s)
    assert off == 22050
    assert length == 11025


def test_resolve_region_samples_clamped_to_eof():
    s = AudioSlice(offset=40000, length=10000, unit="samples")  # 40000+10000 > 44100
    off, length = _resolve_region(n_native=44100, native_sr=44100, slice_spec=s)
    assert off == 40000
    assert length == 44100 - 40000  # clamped, not out-of-range


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_resolve_region_rejects_nonfinite_or_negative_offset(bad):
    s = AudioSlice(offset=bad, length=None, unit="samples")
    with pytest.raises(InputError) as ei:
        _resolve_region(n_native=44100, native_sr=44100, slice_spec=s)
    assert ei.value.code == "INPUT_WAV_SLICE_INVALID"


def test_resolve_region_offset_at_eof_out_of_range():
    s = AudioSlice(offset=44100, length=None, unit="samples")
    with pytest.raises(InputError) as ei:
        _resolve_region(n_native=44100, native_sr=44100, slice_spec=s)
    assert ei.value.code == "INPUT_WAV_SLICE_OUT_OF_RANGE"


def test_resolve_region_zero_length_out_of_range():
    s = AudioSlice(offset=0, length=0, unit="samples")
    with pytest.raises(InputError) as ei:
        _resolve_region(n_native=44100, native_sr=44100, slice_spec=s)
    assert ei.value.code == "INPUT_WAV_SLICE_OUT_OF_RANGE"


from sonoscope.wav_orchestrator import TARGET_SR, _resample_chunk


def test_resample_44100_to_48000_realized_length_golden(tmp_path):
    # RED (R5): 1s mono @44100 -> resample to 48k. Pin the REALIZED soxr output
    # length as a golden (hardcoded once observed during GREEN, per plan §15.1).
    # Mutation: slicing after resample, wrong ratio, or dropping res_type="soxr_hq"
    # -> different realized length -> RED.
    n = 44100
    chunk = np.linspace(-0.3, 0.3, n, dtype=np.float32).reshape(1, n)  # (1, n)
    out, res_type, soxr_ver = _resample_chunk(chunk, native_sr=44100)
    assert out.dtype == np.float32
    assert out.shape[0] == 1
    # GOLDEN: exact realized soxr_hq output length for 44100->48000 of 44100
    # native samples. soxr returns 48001, +1 over the ideal round(44100*48000/44100)
    # = 48000 (the documented +-1 resample-length variance, design §6/§15.1).
    assert out.shape[1] == 48001
    assert res_type == "soxr_hq"
    assert soxr_ver == "1.1.0"


def test_resample_48000_is_noop_with_null_provenance():
    n = 4096
    chunk = np.linspace(-0.3, 0.3, n, dtype=np.float32).reshape(1, n)
    out, res_type, soxr_ver = _resample_chunk(chunk, native_sr=48000)
    assert out.shape == (1, n)
    assert np.array_equal(out, chunk)   # true no-op, samples unperturbed
    assert res_type is None
    assert soxr_ver is None


def test_resample_48000_noop_returns_contiguous_copy_not_a_view():
    # chunk_input is itself a VIEW into a larger array (mimics real ingest slicing
    # audio[:, cs:ce]) to catch the case where the no-op path leaks that view.
    n = 4096
    audio = np.linspace(-0.3, 0.3, n * 2, dtype=np.float32).reshape(1, n * 2)
    chunk_input = audio[:, :n]
    out, res_type, soxr_ver = _resample_chunk(chunk_input, native_sr=48000)
    assert np.array_equal(out, chunk_input)
    assert not np.shares_memory(out, chunk_input)
    assert out.flags["C_CONTIGUOUS"]
    assert res_type is None
    assert soxr_ver is None


def test_resample_48000_noop_owned_contiguous_passes_through_no_copy():
    # gemini perf: a FRESH, owned, C-contiguous 48k input (exactly what
    # _read_chunk_native yields per chunk via ascontiguousarray(data.T)) must be
    # returned AS-IS — no redundant ~115 MB copy on the 48k no-op path.
    # RED against the unconditional-copy code (out is a fresh object, so
    # `out is arr` is False); GREEN once the copy is made conditional and skipped
    # for owned-contiguous inputs. Mutation: reinstating the unconditional
    # astype(copy=True) breaks `out is arr`.
    n = 4096
    arr = np.linspace(-0.3, 0.3, n, dtype=np.float32).reshape(1, n).copy()
    assert arr.base is None            # owned, not a view
    assert arr.flags["C_CONTIGUOUS"]
    assert arr.dtype == np.float32
    out, res_type, soxr_ver = _resample_chunk(arr, native_sr=48000)
    assert out is arr                  # pass-through: no copy, same object
    assert res_type is None
    assert soxr_ver is None


# --- Task 7: chunking model — tiling, threshold validation, remainder-fold (D2) ---
from sonoscope.wav_orchestrator import _chunk_bounds, _min_native


def test_min_native_44100_absorbs_soxr_variance():
    # ceil(2048 * 44100 / 48000) = ceil(1881.6) = 1882; +2 margin -> 1884.
    assert _min_native(44100) == 1884


def test_single_chunk_when_region_under_threshold():
    # region 44100 native @44100, threshold 600s -> one chunk [0, 44100)
    bounds = _chunk_bounds(start=0, length=44100, native_sr=44100, max_chunk_seconds=600.0)
    assert bounds == [(0, 44100)]


def test_single_chunk_at_min_native_boundary():
    # region EXACTLY min_native @44100 -> valid single chunk covering it exactly.
    m = _min_native(44100)
    bounds = _chunk_bounds(start=0, length=m, native_sr=44100, max_chunk_seconds=600.0)
    assert bounds == [(0, m)]


def test_multi_chunk_tiles_region_with_no_gap_or_overlap():
    # threshold 0.5s @44100 -> chunk_native=22050; region=88200 -> exact 4 chunks.
    bounds = _chunk_bounds(start=0, length=88200, native_sr=44100, max_chunk_seconds=0.5)
    assert bounds == [(0, 22050), (22050, 44100), (44100, 66150), (66150, 88200)]
    # tiling invariants
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 88200
    for a, b in zip(bounds, bounds[1:]):
        assert a[1] == b[0]  # no gap / no overlap


def test_multi_chunk_tiles_region_with_nonzero_start():
    # start offset must be honored (region [1000, 1000+88200)).
    bounds = _chunk_bounds(start=1000, length=88200, native_sr=44100, max_chunk_seconds=0.5)
    assert bounds == [(1000, 23050), (23050, 45100), (45100, 67150), (67150, 89200)]


def test_remainder_fold_absorbs_short_tail():
    # threshold 0.5s -> chunk_native=22050; region=88200+500 tail (< min_native).
    # last chunk should be LARGER than chunk_native (tail folded in), not degenerate.
    length = 22050 * 4 + 500  # 500-sample tail, < min_native
    bounds = _chunk_bounds(start=0, length=length, native_sr=44100, max_chunk_seconds=0.5)
    assert bounds[-1][1] == length            # covers to region end
    assert bounds[-1][1] - bounds[-1][0] > 22050  # last chunk folded the short tail
    assert len(bounds) == 4
    # no gap / no overlap preserved after fold
    assert bounds[0][0] == 0
    for a, b in zip(bounds, bounds[1:]):
        assert a[1] == b[0]


def test_remainder_at_min_native_is_kept_not_folded():
    # A trailing chunk of EXACTLY min_native is >= min_native -> NOT folded.
    m = _min_native(44100)
    length = 22050 * 2 + m  # trailing chunk exactly min_native
    bounds = _chunk_bounds(start=0, length=length, native_sr=44100, max_chunk_seconds=0.5)
    assert bounds == [(0, 22050), (22050, 44100), (44100, 44100 + m)]


def test_whole_region_too_short_raises_slice_too_short():
    # region < min_native @44100 -> INPUT_WAV_SLICE_TOO_SHORT
    tiny = _min_native(44100) - 1
    with pytest.raises(InputError) as ei:
        _chunk_bounds(start=0, length=tiny, native_sr=44100, max_chunk_seconds=600.0)
    assert ei.value.code == "INPUT_WAV_SLICE_TOO_SHORT"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_chunk_param_invalid(bad):
    with pytest.raises(InputError) as ei:
        _chunk_bounds(start=0, length=88200, native_sr=44100, max_chunk_seconds=bad)
    assert ei.value.code == "INPUT_WAV_CHUNK_PARAM_INVALID"


def test_chunk_too_small_rejected_with_effective_min_message():
    # threshold so small that chunk_native < min_native -> INPUT_WAV_CHUNK_TOO_SMALL.
    with pytest.raises(InputError) as ei:
        _chunk_bounds(start=0, length=88200, native_sr=44100, max_chunk_seconds=0.001)
    assert ei.value.code == "INPUT_WAV_CHUNK_TOO_SMALL"
    # message states the effective minimum seconds (min_native / native_sr)
    assert "second" in str(ei.value).lower()


# --- Task 8: analyze_wav assembly — per-chunk pipeline, provenance, R1 guard ---
from sonoscope.schema.models import WavAnalysisReport
from sonoscope.wav_orchestrator import WavFileSource, analyze_wav


def test_analyze_wav_44100_provenance_and_array(tmp_path):
    n = 96000  # ~2.2s @44100, comfortably above min
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "a.wav"), data, 44100, subtype="PCM_16")
    report = analyze_wav(WavFileSource(tmp_path / "a.wav"))
    assert isinstance(report, WavAnalysisReport)
    assert len(report.root) == 1
    p = report.root[0].input_provenance
    assert p.original_sample_rate == 44100
    assert p.n_channels == 1
    assert p.source_subtype == "PCM_16"
    assert p.resample_res_type == "soxr_hq"
    assert p.soxr_version == "1.1.0"
    assert p.chunk_index == 0 and p.n_chunks == 1
    # Every remaining provenance field is stamped honestly (design §10.2).
    assert p.analysis_dtype == "float32"
    assert p.channel_reduction == "mean_spectral_max_peak"
    assert p.max_chunk_seconds == 600.0
    assert p.analyzed_window.native_offset_samples == 0
    assert p.analyzed_window.native_length_samples == n
    assert p.analyzed_window.native_sample_rate == 44100
    # 96000 native @44100 -> realized 48k length. GOLDEN: observed soxr_hq output;
    # ideal round(96000*48000/44100)=104490 (soxr matches ideal here, no +-1 drift).
    assert p.analyzed_window.analyzed_samples_48k == 104490
    chunk = report.root[0]
    assert chunk.kind == "wav-chunk-analysis"
    assert chunk.deterministic.summary.sample_rate_hz == 48000
    assert chunk.descriptors is not None
    assert chunk.descriptor_gate is None


def test_analyze_wav_48k_native_noop_provenance(tmp_path):
    n = 96000
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "b.wav"), data, 48000, subtype="PCM_16")
    report = analyze_wav(WavFileSource(tmp_path / "b.wav"))
    p = report.root[0].input_provenance
    assert p.original_sample_rate == 48000
    assert p.resample_res_type is None
    assert p.soxr_version is None
    # 48k no-op: realized window equals the native window exactly.
    assert p.analyzed_window.native_length_samples == n
    assert p.analyzed_window.analyzed_samples_48k == n


def test_analyze_wav_too_short_slice_raises(tmp_path):
    # RED (R1): slice yielding < 2048 @48k -> INPUT_WAV_SLICE_TOO_SHORT
    n = 96000
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "c.wav"), data, 48000, subtype="PCM_16")
    with pytest.raises(InputError) as ei:
        analyze_wav(WavFileSource(tmp_path / "c.wav"),
                    slice_spec=AudioSlice(offset=0, length=1000, unit="samples"))
    assert ei.value.code == "INPUT_WAV_SLICE_TOO_SHORT"


def test_analyze_wav_deterministic_same_env(tmp_path):
    n = 96000
    data = np.sin(np.linspace(0, 400, n)).astype(np.float32)
    sf.write(str(tmp_path / "d.wav"), data, 44100, subtype="PCM_16")
    r1 = analyze_wav(WavFileSource(tmp_path / "d.wav"))
    r2 = analyze_wav(WavFileSource(tmp_path / "d.wav"))
    # byte-identical summary scalars (exact equality; generated_at excluded)
    assert (r1.root[0].deterministic.summary.model_dump()
            == r2.root[0].deterministic.summary.model_dump())


def test_min_native_48000_is_true_floor_no_margin():
    # 48k->48k is a genuine no-op resample (zero variance): 2048 is the true
    # floor, NOT 2048+2. Regression guard for the spurious-margin bug.
    from sonoscope.wav_orchestrator import _min_native
    assert _min_native(48000) == 2048


def test_analyze_wav_48k_exactly_2048_samples_is_accepted(tmp_path):
    # GREEN: a 48000-Hz slice of EXACTLY 2048 samples (design §8 floor) must be
    # analyzable, not rejected by a spurious no-op-resample margin.
    n = 2048
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "floor.wav"), data, 48000, subtype="PCM_16")
    report = analyze_wav(WavFileSource(tmp_path / "floor.wav"))
    assert len(report.root) == 1
    chunk = report.root[0]
    assert chunk.input_provenance.analyzed_window.analyzed_samples_48k == 2048


def test_analyze_wav_chunk_isolates_r1_realized_length_guard(tmp_path, monkeypatch):
    # RED-on-delete: isolates the R1 realized-length guard (~lines 313-322) by
    # monkeypatching _resample_chunk to return a channel-major array below
    # MIN_ANALYZE_SAMPLES_48K, bypassing the _chunk_bounds pre-check entirely
    # (which would otherwise catch this first). If the guard is deleted, this
    # goes RED because compute_summary would then run with < n_fft samples.
    import sonoscope.wav_orchestrator as wo

    n = 4096  # comfortably passes _chunk_bounds's min_native pre-check @48k
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "g.wav"), data, 48000, subtype="PCM_16")

    def _fake_resample_chunk(chunk_native, native_sr):
        short = chunk_native[:, :100]  # < MIN_ANALYZE_SAMPLES_48K (2048) @48k
        return short, None, None

    monkeypatch.setattr(wo, "_resample_chunk", _fake_resample_chunk)

    with pytest.raises(InputError) as ei:
        analyze_wav(WavFileSource(tmp_path / "g.wav"))
    assert ei.value.code == "INPUT_WAV_SLICE_TOO_SHORT"
    assert ei.value.detail["reason"] == "below_n_fft"


def test_analyze_wav_chunks_tile_and_index(tmp_path):
    n = 44100 * 3  # 3s @44100
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "e.wav"), data, 44100, subtype="PCM_16")
    report = analyze_wav(WavFileSource(tmp_path / "e.wav"), max_chunk_seconds=1.0)
    windows = [c.input_provenance.analyzed_window for c in report.root]
    n_chunks = report.root[0].input_provenance.n_chunks
    assert n_chunks == len(report.root) >= 3
    # native windows tile the region with no gap/overlap
    for a, b in zip(windows, windows[1:]):
        assert a.native_offset_samples + a.native_length_samples == b.native_offset_samples
    for i, c in enumerate(report.root):
        assert c.input_provenance.chunk_index == i
        assert c.input_provenance.n_chunks == n_chunks
        # each chunk carries its own independent realized-48k window
        assert c.input_provenance.analyzed_window.analyzed_samples_48k >= 2048


# --- T9: per-chunk descriptor gate + cross-chunk aggregation (D0, §11) --------

from sonoscope.features.descriptor_gate import load_expected_descriptors
from sonoscope.schema.models import DescriptorGateResult
from sonoscope.wav_orchestrator import aggregate_gate


def _pass_gate() -> DescriptorGateResult:
    return DescriptorGateResult(verdict="PASS", reasons=[], spec_sha256=None)


def _three_chunk_report(tmp_path):
    # 3s @44100 with max_chunk_seconds=1.0 tiles into 3 independent chunks.
    n = 44100 * 3
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "agg.wav"), data, 44100, subtype="PCM_16")
    return analyze_wav(WavFileSource(tmp_path / "agg.wav"), max_chunk_seconds=1.0)


def test_analyze_wav_sets_per_chunk_gate_pass(tmp_path):
    # A PASS expectation ("loud" IS emitted for this ramp) must persist a per-chunk
    # DescriptorGateResult carrying the threaded spec_sha256. verdict is the
    # per-chunk PASS/RED token (NOT the aggregate GREEN/RED).
    n = 96000
    sf.write(str(tmp_path / "g.wav"),
             np.linspace(-0.4, 0.4, n, dtype=np.float32), 44100, subtype="PCM_16")
    expectation = load_expected_descriptors(
        {"expect_present": ["loud"]}, block_kind="audio"
    )
    report = analyze_wav(WavFileSource(tmp_path / "g.wav"),
                         expectation=expectation, expectation_spec_sha256="deadbeef")
    assert len(report.root) == 1
    assert report.root[0].descriptor_gate == DescriptorGateResult(
        verdict="PASS", reasons=[], spec_sha256="deadbeef"
    )


def test_analyze_wav_per_chunk_gate_red_is_really_evaluated(tmp_path):
    # RED: an expectation that FAILS ("bright" is NOT emitted for this ramp) must
    # persist verdict="RED" with the exact evaluator reason. Kills the always-PASS
    # mirage: a wiring that hard-codes PASS (never calls evaluate_descriptors)
    # fails this assertion.
    n = 96000
    sf.write(str(tmp_path / "r.wav"),
             np.linspace(-0.4, 0.4, n, dtype=np.float32), 44100, subtype="PCM_16")
    expectation = load_expected_descriptors(
        {"expect_present": ["bright"]}, block_kind="audio"
    )
    report = analyze_wav(WavFileSource(tmp_path / "r.wav"),
                         expectation=expectation, expectation_spec_sha256="cafe")
    assert report.root[0].descriptor_gate == DescriptorGateResult(
        verdict="RED", reasons=["DESC_MISSING: bright"], spec_sha256="cafe"
    )


def test_analyze_wav_no_expectation_leaves_gate_none(tmp_path):
    # No expectation -> every chunk's descriptor_gate stays None (gate is opt-in).
    report = _three_chunk_report(tmp_path)
    assert [c.descriptor_gate for c in report.root] == [None, None, None]


def test_aggregate_gate_green_when_no_red_chunks(tmp_path):
    # All per-chunk verdicts PASS -> aggregate GREEN, no red chunks, no reasons.
    # The aggregate GREEN token is INTENTIONALLY distinct from the per-chunk PASS.
    report = _three_chunk_report(tmp_path)
    for chunk in report.root:
        chunk.descriptor_gate = _pass_gate()
    assert aggregate_gate(report) == ("GREEN", [], [])


# --- Provenance honesty: kill green mirages from all-PCM_16-fixture bias -----

def test_analyze_wav_non_pcm16_subtype_flows_into_provenance(tmp_path):
    # RED-on-hardcode: every other fixture in this file is PCM_16, so a hardcoded
    # source_subtype="PCM_16" would pass silently. Write FLOAT instead and assert
    # provenance reflects the real on-disk subtype.
    n = 96000
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "float.wav"), data, 44100, subtype="FLOAT")
    report = analyze_wav(WavFileSource(tmp_path / "float.wav"))
    assert report.root[0].input_provenance.source_subtype == "FLOAT"


def test_analyze_wav_soxr_version_is_runtime_read(tmp_path, monkeypatch):
    # RED-on-constant: monkeypatch soxr.__version__ (the exact source read by
    # _resample_chunk, see wav_orchestrator.py:180) and confirm the stamped
    # provenance value tracks it, proving it isn't a hardcoded literal.
    import soxr

    n = 96000  # 44100 native -> resampling actually happens
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "resample.wav"), data, 44100, subtype="PCM_16")
    monkeypatch.setattr(soxr, "__version__", "9.9.9")
    report = analyze_wav(WavFileSource(tmp_path / "resample.wav"))
    assert report.root[0].input_provenance.soxr_version == "9.9.9"


def test_analyze_wav_stereo_n_channels_flows_into_provenance(tmp_path):
    # Existing tests only cover mono (n_channels==1); prove the loader's real
    # channel count threads through to provenance for stereo input.
    n = 96000
    left = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    right = np.linspace(0.4, -0.4, n, dtype=np.float32)
    frame_major = np.stack([left, right], axis=1)  # (n, 2)
    sf.write(str(tmp_path / "stereo.wav"), frame_major, 44100, subtype="PCM_16")
    report = analyze_wav(WavFileSource(tmp_path / "stereo.wav"))
    assert report.root[0].input_provenance.n_channels == 2


# --- Partial (seek-based) loading proof: peak memory ~one chunk, not whole file --

def test_partial_load_reads_at_most_one_chunk_per_read(tmp_path, monkeypatch):
    # RED-on-old: the whole-file loader calls SoundFile.read() ONCE with no
    # frames= argument (reads every frame into memory), so frames_read == [None]
    # and len(frames_read) == 1 -> this test fails. The seek-based loader reads
    # exactly one chunk per read, so peak memory is ~one chunk, never the file.
    n = 44100 * 6  # 6s @44100 -> with max_chunk_seconds=1.0, tiles into 6 chunks
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "multi.wav"), data, 44100, subtype="PCM_16")

    frames_read: list = []
    orig_read = sf.SoundFile.read

    def spy_read(self, *args, **kwargs):
        frames_read.append(kwargs.get("frames"))
        return orig_read(self, *args, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", spy_read)

    chunk_native = round(1.0 * 44100)  # 44100 native samples per chunk
    report = analyze_wav(WavFileSource(tmp_path / "multi.wav"), max_chunk_seconds=1.0)

    assert len(report.root) == 6                        # 6 independent chunks
    assert len(frames_read) == 6                         # exactly one read per chunk
    assert all(isinstance(fr, int) for fr in frames_read)  # every read is frame-limited
    assert max(frames_read) == chunk_native             # never larger than one chunk
    assert max(frames_read) < n                          # NEVER the whole file at once
    assert sum(frames_read) == n                         # region fully covered, in pieces


def test_partial_load_offset_reads_only_region_after_offset(tmp_path, monkeypatch):
    # RED-on-old: the whole-file loader reads all n frames regardless of --offset,
    # so sum(frames_read) == n (and the single read has frames=None). The seek-based
    # loader seeks past the offset and reads ONLY the region [offset, n).
    n = 44100 * 3
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "offset.wav"), data, 44100, subtype="PCM_16")

    offset = 50000
    region = n - offset  # single chunk under the default 600s threshold

    frames_read: list = []
    orig_read = sf.SoundFile.read

    def spy_read(self, *args, **kwargs):
        frames_read.append(kwargs.get("frames"))
        return orig_read(self, *args, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", spy_read)

    report = analyze_wav(
        WavFileSource(tmp_path / "offset.wav"),
        slice_spec=AudioSlice(offset=offset, length=None, unit="samples"),
    )

    assert len(report.root) == 1
    assert all(isinstance(fr, int) for fr in frames_read)  # frame-limited reads only
    assert sum(frames_read) == region                       # region only, no pre-offset frames
    assert sum(frames_read) < n                              # strictly less than the whole file


# --- Review fixes: narrow INPUT_WAV_UNREADABLE + short-read guard --------------


def test_analyze_wav_internal_pipeline_error_propagates_raw(tmp_path, monkeypatch):
    # RED (FIX 1): on a perfectly READABLE wav, an internal analysis-stage defect
    # (here compute_summary) must propagate RAW, NOT be rewrapped as the I/O-domain
    # INPUT_WAV_UNREADABLE. Pre-refactor only the file LOAD was wrapped; the refactor
    # widened the try to swallow ALL pipeline exceptions. This test pins the narrowed
    # scope: a valid file + a sentinel-raising stage => the sentinel escapes untouched.
    import sonoscope.wav_orchestrator as wo

    n = 96000
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "readable.wav"), data, 44100, subtype="PCM_16")

    sentinel = RuntimeError("boom sentinel")

    def _boom(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(wo, "compute_summary", _boom)

    with pytest.raises(RuntimeError) as ei:
        analyze_wav(WavFileSource(tmp_path / "readable.wav"))
    assert ei.value is sentinel  # exact sentinel, not an InputError rewrap


def test_analyze_wav_short_read_is_unreadable(tmp_path, monkeypatch):
    # RED (FIX 2): libsndfile can silently return fewer frames than requested (EOF /
    # a file truncated between the metadata read and the chunk read). The chunk read
    # MUST verify the realized count and fail loud with INPUT_WAV_UNREADABLE + a
    # "short read" detail, never analyze a partially-read buffer as if complete.
    n = 96000
    data = np.linspace(-0.4, 0.4, n, dtype=np.float32)
    sf.write(str(tmp_path / "trunc.wav"), data, 44100, subtype="PCM_16")

    orig_read = sf.SoundFile.read

    def short_read(self, *args, **kwargs):
        full = orig_read(self, *args, **kwargs)
        return full[:-1]  # one frame short of the requested length

    monkeypatch.setattr(sf.SoundFile, "read", short_read)

    with pytest.raises(InputError) as ei:
        analyze_wav(WavFileSource(tmp_path / "trunc.wav"))
    assert ei.value.code == "INPUT_WAV_UNREADABLE"
    assert ei.value.detail == {"start": 0, "requested": n, "realized": n - 1}
    assert "short read" in str(ei.value)


def test_aggregate_gate_red_with_chunk_prefixed_reasons(tmp_path):
    # One RED chunk (index 1) among PASS chunks -> aggregate RED, red_chunks=[1],
    # every reason prefixed "chunk[1] " for per-chunk attribution.
    report = _three_chunk_report(tmp_path)
    report.root[0].descriptor_gate = _pass_gate()
    report.root[1].descriptor_gate = DescriptorGateResult(
        verdict="RED",
        reasons=["DESC_MISSING: bright", "DESC_UNEXPECTED: loud"],
        spec_sha256=None,
    )
    report.root[2].descriptor_gate = _pass_gate()
    assert aggregate_gate(report) == (
        "RED",
        [1],
        ["chunk[1] DESC_MISSING: bright", "chunk[1] DESC_UNEXPECTED: loud"],
    )
