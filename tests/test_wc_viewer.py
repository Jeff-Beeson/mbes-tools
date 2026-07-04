"""Tests for the interactive water-column viewer (mbes_tools.wc_viewer).

The per-ping geometry and the fan->column reduction are unit-tested
synthetically (always run, no matplotlib, no data). The whole-file model and the
static renderer are exercised end-to-end over the committed EM124 ``.kmwcd`` and
EM122 ``.wcd`` fixtures (gated on their presence / matplotlib).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import wc_viewer as wv
from mbes_tools import water_column_geo as wg

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


# ---------------------------------------------------------------------------
# PingView geometry + fan->column reduction (synthetic, pure).
# ---------------------------------------------------------------------------


def make_ping(angles_deg, amp, *, heave=0.0, tdepth=0.0, detected=None):
    """A PingView with c=1500, fs=750 -> range = 1 m per absolute sample."""
    angles_deg = np.asarray(angles_deg, float)
    amp = np.asarray(amp, float)
    ns = amp.shape[1]
    if detected is None:
        detected = np.zeros(angles_deg.size, int)
    return wv.PingView(
        index=0, time=0.0, label="synthetic", lat=0.0, lon=0.0,
        heading_deg=0.0, roll_deg=0.0, pitch_deg=0.0, heave_m=heave,
        transducer_depth_m=tdepth, sound_speed_m_s=1500.0, sample_freq_hz=750.0,
        angles_deg=angles_deg, detected_samples=np.asarray(detected, int),
        sample_idx=np.arange(ns), amp=amp.astype(np.float16),
    )


def test_fan_points_place_nadir_below_and_port_positive():
    # One nadir beam, one +30 deg (port) beam; range = sample index (m).
    p = make_ping([0.0, 30.0], [[np.nan, 10.0, 20.0], [np.nan, 5.0, 8.0]])
    across, depth, amp = p.fan_points()
    # nadir sample at k=2 -> across ~0, depth ~2.
    nadir = np.isclose(across, 0.0)
    assert nadir.any() and np.isclose(depth[nadir].max(), 2.0)
    # +30 deg beam sample at k=2 -> across = 2*sin30 = +1 (port), depth = 2*cos30.
    port = np.isclose(across, 1.0, atol=1e-6)
    assert port.any() and np.allclose(depth[port], 2.0 * np.cos(np.deg2rad(30)))


def test_fan_points_add_heave_and_transducer_depth_to_depth():
    p = make_ping([0.0], [[np.nan, 10.0]], heave=1.5, tdepth=6.0)
    across, depth, amp = p.fan_points()
    # nadir sample k=1 -> depth = 1 (range) + 1.5 (heave) + 6 (transducer) = 8.5.
    assert np.isclose(depth.max(), 8.5)


def test_depth_column_swath_max_is_peak_hold_across_beams():
    # Two nadir beams; depth == sample index. Peak-hold per 1 m depth bin.
    a = [np.nan, 5.0, 3.0, 8.0]
    b = [np.nan, 1.0, 9.0, 1.0]
    p = make_ping([0.0, 0.0], [a, b])
    edges = np.linspace(0.0, 4.0, 5)  # bins centred on depths 0,1,2,3
    col = p.depth_column(edges, "swath-max", 3.0)
    # depth 2 m bin: max(3, 9) dB = 9.
    assert np.isclose(col[2], 9.0)
    assert np.isclose(col[3], 8.0)


def test_depth_column_swath_mean_averages_in_linear_domain():
    p = make_ping([0.0, 0.0], [[np.nan, np.nan, 3.0], [np.nan, np.nan, 9.0]])
    edges = np.linspace(0.0, 4.0, 5)
    col = p.depth_column(edges, "swath-mean", 3.0)
    lin_mean = (10 ** 0.3 + 10 ** 0.9) / 2.0
    assert np.isclose(col[2], 10.0 * np.log10(lin_mean), atol=1e-4)  # ~6.96 dB, not 6.0


def test_depth_column_nadir_uses_only_near_vertical_beams():
    # A bright off-nadir beam must be ignored by the nadir section.
    p = make_ping([0.0, 40.0], [[np.nan, np.nan, 2.0], [np.nan, np.nan, 50.0]])
    edges = np.linspace(0.0, 4.0, 5)
    col = p.depth_column(edges, "nadir", 3.0)
    # Only the nadir beam contributes (depth 2 m -> 2 dB); the 40 deg beam maps to
    # depth 2*cos40 ~1.5 m and is excluded, so no huge value appears.
    assert np.isclose(np.nanmax(col), 2.0)


def test_max_bottom_depth_from_detection():
    p = make_ping([0.0, 30.0], [[np.nan, 1.0], [np.nan, 1.0]], detected=[100, 200])
    # nadir detect at sample 100 -> 100 m; 30 deg at 200 -> 200*cos30 ~173 m.
    # The deepest detected-bottom depth is the larger of the two.
    assert np.isclose(p.max_bottom_depth(), 200.0 * np.cos(np.deg2rad(30)))


def test_across_to_lonlat_nadir_is_vessel_and_port_rotates_by_heading():
    p = make_ping([0.0], [[np.nan, 1.0]])
    p.lat, p.lon = 45.0, -125.0
    # across 0 -> the vessel position exactly.
    assert p.across_to_lonlat(0.0) == (p.lon, p.lat)
    # Heading east (90): +port (left) faces north -> lat increases, lon ~unchanged.
    p.heading_deg = 90.0
    lon, lat = p.across_to_lonlat(1000.0)
    assert lat > 45.0 and abs(lon - (-125.0)) < 1e-6
    assert np.isclose(lat - 45.0, np.degrees(1000.0 / wv._WGS84_A))
    # Heading north (0): +port faces west -> lon decreases, lat ~unchanged.
    p.heading_deg = 0.0
    lon, lat = p.across_to_lonlat(1000.0)
    assert lon < -125.0 and abs(lat - 45.0) < 1e-6


def test_depth_column_across_window_excludes_off_band_samples():
    # Nadir beam (across 0, 2 dB) and a bright 40 deg beam (across ~1.29, 50 dB).
    p = make_ping([0.0, 40.0], [[np.nan, np.nan, 2.0], [np.nan, np.nan, 50.0]])
    edges = np.linspace(0.0, 4.0, 5)
    col = p.depth_column(edges, "swath-max", 3.0, across_window=(-0.5, 0.5))
    # Only the nadir sample is inside the band, so the bright 50 dB is excluded.
    assert np.isclose(np.nanmax(col), 2.0)


# ---------------------------------------------------------------------------
# Whole-file model validation.
# ---------------------------------------------------------------------------


def test_from_file_rejects_bad_on_uncovered(tmp_path):
    f = tmp_path / "x.kmwcd"
    f.write_bytes(b"")
    with pytest.raises(ValueError, match="on_uncovered"):
        wv.WaterColumnFileView.from_file(f, on_uncovered="bogus")


# ---------------------------------------------------------------------------
# End-to-end over committed real fixtures (gated).
# ---------------------------------------------------------------------------


def test_from_file_over_real_kmwcd_with_companion_attitude():
    """The committed .kmwcd is a lone clip whose #MWC ping sits outside the
    same-stem .kmall nav span, so default skip yields nothing; clamp grids it and
    the #SKM companion supplies roll/pitch/heave."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    # Default skip -> the single ping is uncovered -> actionable error.
    with pytest.raises(ValueError, match="covered"):
        wv.WaterColumnFileView.from_file(fx)
    # Clamp -> one ping, companion #SKM nav + attitude, finite stack.
    view = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp")
    assert view.n_pings == 1 and view.n_total == 1 and view.n_uncovered == 1
    assert view.nav_position_source == "#SKM" and view.nav_attitude_source == "#SKM"
    assert view.stack.shape == (600, 1) and np.isfinite(view.stack).any()
    p = view.pings[0]
    assert p.transducer_depth_m > 0  # install lever arm resolved
    across, depth, amp = p.fan_points()
    assert amp.size > 0 and across.min() < 0 < across.max()  # symmetric swath


def test_from_file_multi_ping_stack_via_wcd_and_synthetic_nav():
    """The 3-ping EM122 .wcd fixture (no nav of its own) + covering synthetic nav
    exercises the multi-ping stack assembly and along-track distance."""
    wcd = FIXTURES / "sample_atlantis_em122.wcd"
    if not wcd.exists():
        pytest.skip("wcd fixture not present")
    from mbes_tools.wcd import iter_water_column_datagrams
    from mbes_tools.water_column import reassemble_wcd_pings

    pings = list(reassemble_wcd_pings(iter_water_column_datagrams(wcd)))
    times = [wg._all_header_time(p.header) for p in pings]
    t0, t1 = min(times) - 1.0, max(times) + 1.0
    # Two positions 1 arc-minute apart -> a non-zero along-track baseline.
    nav = wg.NavTrack.from_lists(
        [t0, t1], [30.0, 30.02], [-45.0, -45.0], [t0, t1], [0.0, 0.0],
        "synthetic", "synthetic",
    )
    view = wv.WaterColumnFileView.from_file(wcd, nav=nav)
    assert view.n_pings == 3 and view.n_uncovered == 0
    assert view.stack.shape[1] == 3 and np.isfinite(view.stack).any()
    assert view.along_track_m[-1] > 0  # positions advance along track


def test_stack_modes_differ_on_real_data():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    mx = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp", stack_mode="swath-max")
    nd = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp", stack_mode="nadir")
    both = np.isfinite(mx.stack) & np.isfinite(nd.stack)
    # Peak-hold over the whole swath is >= the near-nadir section everywhere.
    assert both.any() and np.all(mx.stack[both] >= nd.stack[both] - 1e-3)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_render_static_writes_png(tmp_path):
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    view = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp")
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    out = viewer.render_static(tmp_path / "viewer.png", 0)
    assert out.exists() and out.stat().st_size > 0
    plt.close(viewer.fig)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_viewer_select_moves_cursor_and_updates(tmp_path):
    """select() clamps, moves the cursor, and swaps the fan without error."""
    wcd = FIXTURES / "sample_atlantis_em122.wcd"
    if not wcd.exists():
        pytest.skip("wcd fixture not present")
    from mbes_tools.wcd import iter_water_column_datagrams
    from mbes_tools.water_column import reassemble_wcd_pings

    pings = list(reassemble_wcd_pings(iter_water_column_datagrams(wcd)))
    times = [wg._all_header_time(p.header) for p in pings]
    t0, t1 = min(times) - 1.0, max(times) + 1.0
    nav = wg.NavTrack.from_lists([t0, t1], [30.0, 30.0], [-45.0, -45.0],
                                 [t0, t1], [0.0, 0.0], "synthetic", "synthetic")
    view = wv.WaterColumnFileView.from_file(wcd, nav=nav)
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    viewer.select(2)
    assert viewer.current == 2 and viewer.cursor.get_xdata()[0] == 2
    viewer.select(999)  # clamped to last ping
    assert viewer.current == view.n_pings - 1
    plt.close(viewer.fig)


def _real_kmwcd_view():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    return wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp")


def test_amp_sample_populated_and_rebuild_stack_narrows_with_window():
    view = _real_kmwcd_view()
    assert view.amp_sample.size > 0 and np.isfinite(view.amp_sample).all()
    full = np.count_nonzero(np.isfinite(view.stack))
    out = view.rebuild_stack(across_window=(-200.0, 200.0))
    assert out.shape == (600, view.n_pings) and out is view.stack
    narrow = np.count_nonzero(np.isfinite(view.stack))
    # A near-nadir band fills no more depth bins than the whole swath.
    assert 0 < narrow <= full


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_apply_clim_sets_both_panels_and_hist_guides():
    view = _real_kmwcd_view()
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    viewer._apply_clim(-20.0, 10.0)
    assert viewer.im.get_clim() == (-20.0, 10.0)
    assert viewer.sc.get_clim() == (-20.0, 10.0)
    assert viewer._hist_lo.get_xdata()[0] == -20.0 and viewer._hist_hi.get_xdata()[0] == 10.0
    viewer._apply_clim(5.0, 5.0)  # invalid (vmax<=vmin) -> ignored
    assert viewer.im.get_clim() == (-20.0, 10.0)
    plt.close(viewer.fig)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_set_clip_toggles_transparency():
    view = _real_kmwcd_view()
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt, clip_mode="clamp")
    assert viewer._cmap.get_under()[3] == 1.0  # clamp -> opaque end colour
    viewer._set_clip("cut (transparent)")
    assert viewer._clip_mode == "cut"
    assert viewer._cmap.get_under()[3] == 0.0 and viewer._cmap.get_over()[3] == 0.0
    assert viewer.im.cmap.get_under()[3] == 0.0  # propagated to the artist
    viewer._set_clip("clamp (end colours)")
    assert viewer._clip_mode == "clamp" and viewer._cmap.get_under()[3] == 1.0
    plt.close(viewer.fig)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_on_swath_rebuilds_stack_and_reset_restores():
    view = _real_kmwcd_view()
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    before = view.stack.copy()
    viewer._on_swath(-500.0, 500.0)
    assert viewer._across_window == (-500.0, 500.0)
    both = np.isfinite(view.stack) & np.isfinite(before)
    # Peak-hold over a sub-swath is <= the full swath, and something changed.
    assert both.any() and np.all(view.stack[both] <= before[both] + 1e-3)
    assert not np.allclose(view.stack[both], before[both])
    assert "swath [" in viewer.ax_stack.get_title()  # band annotation, not the mode name
    viewer._on_swath(10.0, 10.2)  # sub-1m drag treated as a click -> no change
    assert viewer._across_window == (-500.0, 500.0)
    viewer._reset_swath()
    assert viewer._across_window is None and "swath [" not in viewer.ax_stack.get_title()
    plt.close(viewer.fig)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_build_controls_constructs_widgets_and_slider_drives_clim():
    from matplotlib.widgets import RadioButtons, RangeSlider, SpanSelector

    view = _real_kmwcd_view()
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    viewer._build_controls()  # the interactive-widget path, headless
    assert isinstance(viewer._clim_slider, RangeSlider)
    assert isinstance(viewer._clip_radio, RadioButtons)
    assert isinstance(viewer._span, SpanSelector)
    viewer._clim_slider.set_val((-30.0, 5.0))  # -> on_changed -> _apply_clim
    assert viewer.im.get_clim() == (-30.0, 5.0)
    plt.close(viewer.fig)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_status_readouts_over_fan_and_stack():
    view = _real_kmwcd_view()
    plt = wv._plt(interactive=False)
    viewer = wv.WaterColumnViewer(view, plt)
    p = view.pings[0]
    fan = viewer._status_over_fan(0.0, 1500.0)  # across 0 -> vessel lon/lat
    assert f"{p.lat:+.5f}" in fan and f"{p.lon:+.5f}" in fan and "depth" in fan
    stk = viewer._status_over_stack(0, 1200.0)
    assert "ping" in stk and f"{p.lat:+.5f}" in stk
    plt.close(viewer.fig)


def test_clean_water_reduces_fan_samples():
    """--clean-water drops the below-nadir-range samples, so the ping's fan has
    strictly fewer finite points than the unfiltered view."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    full = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp")
    clean = wv.WaterColumnFileView.from_file(fx, on_uncovered="clamp", clean_water=True)
    n_full = full.pings[0].fan_points()[2].size
    n_clean = clean.pings[0].fan_points()[2].size
    assert 0 < n_clean < n_full
