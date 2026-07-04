"""Tests for mbes_tools.water_column (vessel-frame WC products).

The reassembly, gridding, and anomaly logic is unit-tested synthetically (always
run, no matplotlib, no data). A gated end-to-end test grids + detects over the
committed real fixtures when matplotlib is importable.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import water_column as wcp
from mbes_tools.wc_diagnostics import WCFrame
from mbes_tools.wcd import WaterColumnDatagram, WCBeam


# ---------------------------------------------------------------------------
# Helpers to build synthetic inputs.
# ---------------------------------------------------------------------------


def make_wcd_fragment(counter, num_datagram, datagram_num, beam_numbers, num_beams_ping):
    """A minimal WaterColumnDatagram fragment carrying the given beam numbers."""
    beams = [
        WCBeam(
            pointing_angle_deg=float(b),
            start_range_sample=0,
            number_of_samples=0,
            detected_range_samples=0,
            tx_sector_number=0,
            beam_number=int(b),
        )
        for b in beam_numbers
    ]
    return WaterColumnDatagram(
        header=None,
        counter=counter,
        serial_number=0,
        num_datagram=num_datagram,
        datagram_num=datagram_num,
        num_tx_sectors=0,
        num_beams_ping=num_beams_ping,
        num_beams_this_datagram=len(beams),
        sound_speed_m_s=1500.0,
        sample_frequency_hz=750.0,
        tx_heave_m=0.0,
        tvg_function=0,
        tvg_offset_db=0,
        beams=beams,
    )


def make_frame(amp_db, angles_deg, detected_samples, c=1500.0, fs=750.0):
    """A synthetic WCFrame; c=1500, fs=750 -> 1 m one-way range per sample."""
    amp = np.asarray(amp_db, dtype=float)
    return WCFrame(
        amp_db=amp,
        phase_deg=None,
        angles_deg=np.asarray(angles_deg, dtype=float),
        detected_samples=np.asarray(detected_samples, dtype=int),
        sound_speed_m_s=c,
        sample_freq_hz=fs,
        phase_flag=0,
        sector_freqs_hz=[],
        label="synthetic",
    )


# ---------------------------------------------------------------------------
# 1. Reassembly.
# ---------------------------------------------------------------------------


def test_merge_wcd_fragments_orders_by_datagram_num():
    # Two fragments delivered out of order; merged ping keeps datagram order.
    f2 = make_wcd_fragment(100, 2, 2, [200, 201], num_beams_ping=4)
    f1 = make_wcd_fragment(100, 2, 1, [100, 101], num_beams_ping=4)
    merged = wcp.merge_wcd_fragments([f2, f1])
    assert [b.beam_number for b in merged.beams] == [100, 101, 200, 201]
    assert merged.num_beams_this_datagram == 4
    assert merged.num_datagram == 1 and merged.datagram_num == 1
    # Ping-level metadata is preserved from the first fragment.
    assert merged.counter == 100 and merged.num_beams_ping == 4


def test_reassemble_groups_by_counter_and_drops_incomplete():
    stream = [
        make_wcd_fragment(1, 1, 1, [0, 1, 2], 3),        # complete single-fragment ping
        make_wcd_fragment(2, 2, 1, [0, 1], 4),           # ping 2 fragment 1/2
        make_wcd_fragment(2, 2, 2, [2, 3], 4),           # ping 2 fragment 2/2 -> completes
        make_wcd_fragment(3, 2, 1, [0], 2),              # ping 3 fragment 1/2 -> never completes
    ]
    pings = list(wcp.reassemble_wcd_pings(stream))
    assert [p.counter for p in pings] == [1, 2]
    assert [len(p.beams) for p in pings] == [3, 4]
    # Every emitted ping reconstructs its declared full beam count.
    assert all(len(p.beams) == p.num_beams_ping for p in pings)

    # allow_incomplete emits the truncated tail ping too.
    pings2 = list(wcp.reassemble_wcd_pings(stream, allow_incomplete=True))
    assert [p.counter for p in pings2] == [1, 2, 3]
    assert len(pings2[-1].beams) == 1  # only the one fragment that arrived


# ---------------------------------------------------------------------------
# 2. Gridding.
# ---------------------------------------------------------------------------


def test_grid_nadir_sample_maps_to_depth_across_zero():
    # One nadir beam, a single sample at range-sample 2 (=2 m one-way) -> depth 2.
    frame = make_frame(amp_db=[[np.nan, np.nan, -30.0]], angles_deg=[0.0], detected_samples=[0])
    g = wcp.grid_frame(frame)  # default 1 m cells
    # depth axis spans 0..2 in 1 m cells; the sample lands in the deepest row.
    assert g.amplitude_db.shape == (2, 1)
    assert g.amplitude_db[1, 0] == pytest.approx(-30.0)
    assert np.isnan(g.amplitude_db[0, 0])
    # Nadir across position (0 m) sits in the single across column.
    assert g.amplitude_db.shape[1] == 1
    assert g.across_edges[0] <= 0.0 <= g.across_edges[-1]
    assert g.counts[1, 0] == 1


def test_grid_reduce_mean_is_linear_intensity_domain():
    # Two nadir samples in one big cell: mean must be in linear-intensity domain.
    frame = make_frame(amp_db=[[np.nan, -20.0], [np.nan, -30.0]],
                       angles_deg=[0.0, 0.0], detected_samples=[0, 0])
    g = wcp.grid_frame(frame, across_res_m=10.0, depth_res_m=10.0)
    assert g.amplitude_db.shape == (1, 1) and g.counts[0, 0] == 2
    expected = 10.0 * np.log10((10 ** (-20 / 10) + 10 ** (-30 / 10)) / 2)
    assert g.amplitude_db[0, 0] == pytest.approx(expected)
    # A plain dB arithmetic mean would be -25; intensity mean is brighter.
    assert g.amplitude_db[0, 0] > -25.0

    gmax = wcp.grid_frame(frame, across_res_m=10.0, depth_res_m=10.0, reduce="max")
    assert gmax.amplitude_db[0, 0] == pytest.approx(-20.0)


def test_grid_empty_frame_returns_degenerate_grid():
    frame = make_frame(amp_db=[[np.nan, np.nan]], angles_deg=[0.0], detected_samples=[0])
    g = wcp.grid_frame(frame)
    assert g.amplitude_db.shape == (1, 1)
    assert np.isnan(g.amplitude_db[0, 0]) and g.counts.sum() == 0


def test_grid_reduce_validation():
    frame = make_frame(amp_db=[[-30.0]], angles_deg=[0.0], detected_samples=[0])
    with pytest.raises(ValueError):
        wcp.grid_frame(frame, reduce="bogus")


# ---------------------------------------------------------------------------
# 3. Water mask + anomaly detection.
# ---------------------------------------------------------------------------


def test_water_mask_excludes_surface_and_bottom_guard():
    # One beam, 10 samples, bottom at sample 6.
    frame = make_frame(amp_db=[[0.0] * 10], angles_deg=[0.0], detected_samples=[6])
    mask = wcp.water_column_water_mask(
        frame, surface_guard_samples=2, min_range_m=0.0, bottom_guard_samples=1
    )
    # water = idx in [2, 6-1=5): samples 2,3,4 only.
    assert list(mask[0]) == [False, False, True, True, True, False, False, False, False, False]


def test_water_mask_no_bottom_uses_full_column_above_surface():
    frame = make_frame(amp_db=[[0.0] * 6], angles_deg=[0.0], detected_samples=[0])
    mask = wcp.water_column_water_mask(frame, surface_guard_samples=2, bottom_guard_samples=1)
    assert list(mask[0]) == [False, False, True, True, True, True]


def test_detect_anomalies_flags_only_off_bottom_bright_return():
    nb, width = 12, 30
    amp = np.full((nb, width), -50.0)
    det = np.full(nb, 25)
    # Three bright spots on beam 5: above bottom (water), below bottom, near surface.
    amp[5, 15] = -10.0   # midwater -> should flag
    amp[5, 28] = -10.0   # below the bottom -> excluded
    amp[5, 1] = -10.0    # inside surface guard -> excluded
    angles = np.linspace(-30, 30, nb)
    frame = make_frame(amp, angles, det)

    anom = wcp.detect_anomalies(
        frame, threshold_db=20.0, surface_guard_samples=5, bottom_guard_samples=2
    )
    assert anom.n_flagged == 1
    assert anom.mask[5, 15]
    assert not anom.mask[5, 28] and not anom.mask[5, 1]
    # Background at the midwater column is the across-beam median (~-50 dB).
    assert anom.background_db[15] == pytest.approx(-50.0)
    assert anom.residual_db[5, 15] == pytest.approx(40.0)


def test_detect_anomalies_robust_threshold_scales_with_spread():
    # Flat water + one strong outlier: robust MAD threshold flags the outlier.
    nb, width = 40, 40
    rng = np.random.default_rng(0)
    amp = -60.0 + rng.normal(0, 1.0, size=(nb, width))
    det = np.full(nb, 35)
    amp[20, 18] = -5.0  # a very bright midwater return
    frame = make_frame(amp, np.linspace(-30, 30, nb), det)
    anom = wcp.detect_anomalies(frame, surface_guard_samples=3, bottom_guard_samples=2)
    assert anom.mask[20, 18]
    assert np.isfinite(anom.threshold_db)
    s = wcp.summarize_anomalies(anom)
    assert s["max_at_beam"] == 20
    assert s["n_flagged"] >= 1


def test_summarize_anomalies_empty():
    frame = make_frame(amp_db=[[-50.0] * 10], angles_deg=[0.0], detected_samples=[8])
    anom = wcp.detect_anomalies(frame, threshold_db=100.0)  # nothing exceeds
    s = wcp.summarize_anomalies(anom)
    assert s["n_flagged"] == 0 and "max_residual_db" not in s


def test_detect_anomalies_validation():
    frame = make_frame(amp_db=[[-30.0]], angles_deg=[0.0], detected_samples=[0])
    with pytest.raises(ValueError):
        wcp.detect_anomalies(frame, background_stat="bogus")


# ---------------------------------------------------------------------------
# Gated end-to-end over the committed real fixtures.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def test_reassembled_wcd_frames_over_real_fixture():
    """The committed 3-ping EM122 .wcd reassembles to 3 full-swath frames."""
    fx = FIXTURES / "sample_atlantis_em122.wcd"
    if not fx.exists():
        pytest.skip("wcd fixture not present")
    frames = list(wcp.reassembled_wcd_frames(fx))
    assert len(frames) == 3
    assert all(f.amp_db.shape[0] == 288 for f in frames)  # 288-beam EM122 swath


def test_grid_and_detect_over_real_mwc_fixture():
    """Grid + anomaly on the real EM124 #MWC ping produce sane geometry."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    frame = next(wcp.mwc_frames(fx))
    g = wcp.grid_frame(frame, max_depth_m=7000.0)
    assert g.amplitude_db.ndim == 2 and g.counts.sum() > 0
    # Symmetric ±30° EM124 swath -> across extent spans both signs.
    assert g.across_edges.min() < -1000 and g.across_edges.max() > 1000
    anom = wcp.detect_anomalies(frame)
    assert anom.water_mask.sum() > 0
    assert anom.mask.shape == frame.amp_db.shape


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_generate_panels(tmp_path):
    mwc = [p for p in [FIXTURES / "sample_tn447_em124.kmwcd",
                       FIXTURES / "sample_em2040_wc_phase1.kmall"] if p.exists()]
    wcd_files = [p for p in [FIXTURES / "sample_atlantis_em122.wcd"] if p.exists()]
    if not mwc and not wcd_files:
        pytest.skip("water-column fixtures not present")
    made = wcp.generate(tmp_path, mwc_files=mwc, wcd_files=wcd_files)
    assert made and all(p.exists() and p.stat().st_size > 0 for p in made)
    # Each input file yields a grid panel and an anomaly panel.
    n = len(mwc) + len(wcd_files)
    assert sum("wc_grid_" in p.name for p in made) == n
    assert sum("wc_anomaly_" in p.name for p in made) == n


# ---------------------------------------------------------------------------
# Minimum-slant-range (clean water column) filter.
# ---------------------------------------------------------------------------


def test_minimum_slant_range_sample_is_shortest_detection():
    # Bottoms at samples 60, 40, 80 -> R_min = 40. A no-bottom beam (0) is ignored.
    det = np.array([60, 40, 0, 80])
    assert wcp.minimum_slant_range_sample(det) == 40
    # No detection anywhere -> None (no reference).
    assert wcp.minimum_slant_range_sample(np.zeros(4, dtype=int)) is None


def test_apply_min_slant_range_cuts_at_shortest_bottom():
    # 3 beams, 100 samples, all finite; bottoms at 70, 50, 90 -> R_min sample = 50.
    amp = np.zeros((3, 100))
    frame = make_frame(amp, angles_deg=[-20.0, 0.0, 20.0], detected_samples=[70, 50, 90])
    out = wcp.apply_min_slant_range(frame)
    # Samples < 50 kept (finite) on every beam; samples >= 50 are NaN on every beam.
    assert np.isfinite(out.amp_db[:, :50]).all()
    assert np.isnan(out.amp_db[:, 50:]).all()
    # Original frame is untouched (returns a copy).
    assert np.isfinite(frame.amp_db).all()


def test_apply_min_slant_range_guard_pulls_cutoff_inward():
    amp = np.zeros((2, 100))
    # c=1500/fs=750 -> 1 m per sample; guard 10 m -> 10 samples inward from R_min=50.
    frame = make_frame(amp, angles_deg=[0.0, 10.0], detected_samples=[50, 60])
    out = wcp.apply_min_slant_range(frame, guard_m=10.0)
    assert np.isfinite(out.amp_db[:, :40]).all()
    assert np.isnan(out.amp_db[:, 40:]).all()


def test_apply_min_slant_range_no_bottom_is_passthrough():
    amp = np.zeros((2, 30))
    frame = make_frame(amp, angles_deg=[0.0, 5.0], detected_samples=[0, 0])
    out = wcp.apply_min_slant_range(frame)
    assert out is frame  # no R_min reference -> unchanged


def test_minimum_slant_range_percentile_is_more_forgiving():
    # One shallow outlier (30) would shrink the cone under the strict min;
    # a percentile ignores it, giving a larger (more forgiving) R_min.
    det = np.array([30, 80, 82, 84, 86])
    assert wcp.minimum_slant_range_sample(det) == 30                 # strict min
    assert wcp.minimum_slant_range_sample(det, percentile=25) > 30   # relaxed off the outlier


def test_apply_min_slant_range_percentile_keeps_more_samples():
    amp = np.zeros((5, 120))
    frame = make_frame(amp, angles_deg=[-40, -20, 0, 20, 40],
                       detected_samples=[30, 80, 82, 84, 86])
    strict = wcp.apply_min_slant_range(frame)                        # cutoff 30
    relaxed = wcp.apply_min_slant_range(frame, percentile=25)        # cutoff > 30
    assert np.isfinite(strict.amp_db).sum() < np.isfinite(relaxed.amp_db).sum()
