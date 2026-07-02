"""Tests for mbes_tools.water_column_geo (geo-referenced WC products, Slice 2).

NavTrack interpolation, per-ping georeferencing geometry, and the streaming
mosaic accumulator are unit-tested synthetically (always run, no matplotlib, no
data). Gated tests exercise the end-to-end mosaic + panel over the committed
real EM124 ``.kmwcd`` fixture.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import water_column_geo as wg
from mbes_tools.install_params import InstallationParameters
from mbes_tools.wc_diagnostics import WCFrame


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def make_frame(amp_db, angles_deg, c=1500.0, fs=750.0):
    """A synthetic WCFrame; c=1500, fs=750 -> 1 m one-way range per sample."""
    amp = np.asarray(amp_db, dtype=float)
    return WCFrame(
        amp_db=amp,
        phase_deg=None,
        angles_deg=np.asarray(angles_deg, dtype=float),
        detected_samples=np.zeros(amp.shape[0], dtype=int),
        sound_speed_m_s=c,
        sample_freq_hz=fs,
        phase_flag=0,
        sector_freqs_hz=[],
        label="synthetic",
    )


def one_sample_frame(angle_deg, sample_k, amp=-20.0, width=None):
    """Single-beam frame with exactly one finite sample at range-sample ``k``.

    With c=1500/fs=750 the sample sits at one-way range ``k`` metres, so
    across = sin(angle)*k and depth = cos(angle)*k.
    """
    width = width if width is not None else sample_k + 1
    grid = np.full((1, width), np.nan)
    grid[0, sample_k] = amp
    return make_frame(grid, [angle_deg])


def const_nav(lat, lon, heading_deg):
    """A NavTrack with a single position + constant heading (two samples)."""
    return wg.NavTrack.from_lists(
        [0.0, 1.0], [lat, lat], [lon, lon], [0.0, 1.0],
        [heading_deg, heading_deg], "test", "test",
    )


def geo_samples(easting, northing, depth, amp):
    """A GeoSamples bundle built directly from coordinate arrays."""
    return wg.GeoSamples(
        easting_m=np.asarray(easting, float),
        northing_m=np.asarray(northing, float),
        depth_m=np.asarray(depth, float),
        amplitude_db=np.asarray(amp, float),
        crs_label="test-crs",
        projector="local",
        vessel_lon=0.0,
        vessel_lat=0.0,
        heading_deg=0.0,
        label="synthetic",
    )


# ---------------------------------------------------------------------------
# 1. NavTrack.
# ---------------------------------------------------------------------------


def test_navtrack_sorts_and_interpolates_position():
    # Fed out of order; position_at interpolates on the sorted track.
    nav = wg.NavTrack.from_lists(
        [2.0, 0.0], [20.0, 0.0], [200.0, 0.0], [0.0], [10.0], "src", "hdg"
    )
    assert list(nav.t_pos) == [0.0, 2.0]
    lat, lon = nav.position_at(1.0)
    assert lat == pytest.approx(10.0) and lon == pytest.approx(100.0)
    # Clamped past the ends (no extrapolation).
    lat_end, _ = nav.position_at(99.0)
    assert lat_end == pytest.approx(20.0)


def test_navtrack_heading_is_circular():
    # Interpolating between 350 and 30 degrees must cross 0 (mid ~10), not 190.
    nav = wg.NavTrack.from_lists(
        [0.0], [0.0], [0.0], [0.0, 10.0], [350.0, 30.0], "p", "h"
    )
    assert nav.heading_at(5.0) == pytest.approx(10.0, abs=0.1)  # short way across 0
    # Endpoints recovered exactly.
    assert nav.heading_at(0.0) == pytest.approx(350.0)
    assert nav.heading_at(10.0) == pytest.approx(30.0)


def test_navtrack_covers_and_span():
    nav = const_nav(1.0, 2.0, 90.0)
    assert nav.time_span == (0.0, 1.0)
    assert nav.covers(0.5) and not nav.covers(2.0)


def test_navtrack_requires_samples():
    with pytest.raises(ValueError):
        wg.NavTrack.from_lists([], [], [], [], [], "p", "h")


# ---------------------------------------------------------------------------
# 2. georeference_frame geometry.
# ---------------------------------------------------------------------------


def test_georeference_places_nadir_sample_below_vessel():
    # A nadir sample at range-sample 10 -> depth 10 at the vessel position.
    frame = one_sample_frame(angle_deg=0.0, sample_k=10)
    nav = const_nav(0.0, 0.0, 0.0)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    assert len(gs) == 1
    assert gs.easting_m[0] == pytest.approx(0.0, abs=1e-6)
    assert gs.northing_m[0] == pytest.approx(0.0, abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "heading, exp_e, exp_n",
    [
        (0.0, -5.0, 0.0),    # north-heading: a port return lands to the west
        (90.0, 0.0, 5.0),    # east-heading:  a port return lands to the north
        (180.0, 5.0, 0.0),   # south-heading: a port return lands to the east
    ],
)
def test_georeference_rotates_port_return_by_heading(heading, exp_e, exp_n):
    # angle +30 (port), sample 10 -> across 5 m; depth cos30*10.
    frame = one_sample_frame(angle_deg=30.0, sample_k=10)
    nav = const_nav(0.0, 0.0, heading)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    assert gs.easting_m[0] == pytest.approx(exp_e, abs=1e-6)
    assert gs.northing_m[0] == pytest.approx(exp_n, abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(np.cos(np.deg2rad(30.0)) * 10.0)
    assert gs.heading_deg == pytest.approx(heading)


def test_georeference_applies_lever_arm():
    # Install lever arm (forward=1, starboard=2, down=3) shifts every sample.
    install = InstallationParameters(raw="", params={"S1X": "1.0", "S1Y": "2.0", "S1Z": "3.0"})
    assert wg._resolve_group(install) == "S1"
    frame = one_sample_frame(angle_deg=30.0, sample_k=10)  # across +5 (port)
    nav = const_nav(0.0, 0.0, 0.0)  # heading north
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local", install=install)
    # x_v = 1 (fwd), y_v = 2 - 5 = -3 (starboard); heading 0 -> east=y_v, north=x_v.
    assert gs.easting_m[0] == pytest.approx(-3.0, abs=1e-6)
    assert gs.northing_m[0] == pytest.approx(1.0, abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(3.0 + np.cos(np.deg2rad(30.0)) * 10.0)


def test_georeference_max_depth_filters_samples():
    grid = np.full((1, 60), np.nan)
    grid[0, 5] = -20.0    # depth 5
    grid[0, 50] = -25.0   # depth 50 -> dropped
    frame = make_frame(grid, [0.0])
    nav = const_nav(0.0, 0.0, 0.0)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local", max_depth_m=10.0)
    assert len(gs) == 1
    assert gs.depth_m[0] == pytest.approx(5.0)


def test_georeference_local_label_carries_utm_zone():
    frame = one_sample_frame(angle_deg=0.0, sample_k=5)
    nav = const_nav(6.90354, 126.97404, 0.0)  # near the EM124 fixture -> UTM 52N
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    assert gs.projector == "local"
    assert "EPSG:32652" in gs.crs_label and "local-ENU" in gs.crs_label


@pytest.mark.skipif(wg._pyproj_available(), reason="pyproj is installed")
def test_georeference_utm_without_pyproj_raises():
    frame = one_sample_frame(angle_deg=0.0, sample_k=5)
    nav = const_nav(0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError):
        wg.georeference_frame(frame, 0.0, nav, projector="utm")


# ---------------------------------------------------------------------------
# 3. GeoMosaic.
# ---------------------------------------------------------------------------


def test_mosaic_max_is_peak_hold_across_pings():
    m = wg.GeoMosaic(cell_m=10.0, reduce="max")
    # Two returns share cell (0,0); a third lands in cell (1,0).
    m.add(geo_samples([5.0, 15.0], [5.0, 5.0], [100.0, 100.0], [-30.0, -40.0]))
    m.add(geo_samples([6.0], [6.0], [100.0], [-10.0]))  # overlaps (0,0), brighter
    res = m.finalize()
    assert res.n_pings == 2
    assert res.amplitude_db.shape == (1, 2)  # 1 north row, 2 east cols
    assert res.amplitude_db[0, 0] == pytest.approx(-10.0)   # peak-hold winner
    assert res.amplitude_db[0, 1] == pytest.approx(-40.0)
    assert list(res.east_edges) == [0.0, 10.0, 20.0]
    assert list(res.north_edges) == [0.0, 10.0]
    assert res.counts[0, 0] == 1  # max mode records occupancy, not sample count


def test_mosaic_mean_is_linear_intensity_domain():
    m = wg.GeoMosaic(cell_m=100.0, reduce="mean")
    m.add(geo_samples([1.0, 2.0], [1.0, 2.0], [50.0, 50.0], [-20.0, -30.0]))
    res = m.finalize()
    expected = 10.0 * np.log10((10 ** (-20 / 10) + 10 ** (-30 / 10)) / 2)
    assert res.amplitude_db[0, 0] == pytest.approx(expected)
    assert res.amplitude_db[0, 0] > -25.0  # brighter than a naive dB mean
    assert res.counts[0, 0] == 2


def test_mosaic_depth_band_filters():
    m = wg.GeoMosaic(cell_m=10.0, reduce="max", depth_band=(100.0, 200.0))
    # In-band, below-band, above-band returns to the same cell.
    m.add(geo_samples([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [150.0, 50.0, 500.0],
                      [-30.0, -5.0, -1.0]))
    res = m.finalize()
    # Only the in-band (-30) return survives; the brighter out-of-band ones drop.
    assert res.amplitude_db[0, 0] == pytest.approx(-30.0)
    assert res.depth_band == (100.0, 200.0)


def test_mosaic_empty_finalize_is_degenerate():
    res = wg.GeoMosaic(cell_m=25.0).finalize()
    assert res.amplitude_db.shape == (1, 1)
    assert np.isnan(res.amplitude_db[0, 0])
    assert res.n_pings == 0 and res.crs_label == "unknown"


def test_mosaic_out_of_band_ping_touches_no_cells():
    m = wg.GeoMosaic(cell_m=10.0, depth_band=(100.0, 200.0))
    touched = m.add(geo_samples([1.0], [1.0], [500.0], [-10.0]))
    assert touched == 0 and m.n_pings == 1
    assert np.isnan(m.finalize().amplitude_db[0, 0])


def test_mosaic_validation():
    with pytest.raises(ValueError):
        wg.GeoMosaic(cell_m=10.0, reduce="bogus")
    with pytest.raises(ValueError):
        wg.GeoMosaic(cell_m=0.0)


def test_mosaic_centers_are_edge_midpoints():
    res = wg.GeoMosaic(cell_m=10.0, reduce="max")
    res.add(geo_samples([5.0], [25.0], [10.0], [-20.0]))
    out = res.finalize()
    assert out.east_centers[0] == pytest.approx(5.0)
    assert out.north_centers[0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Gated end-to-end over the committed real EM124 .kmwcd fixture.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def test_nav_track_and_install_from_real_kmwcd():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    nav = wg.nav_track_from_kmall([fx])
    # This water-column-only file has no #SKM, so nav falls back to #SPO COG.
    assert nav.position_source == "#SPO"
    assert nav.t_pos.size > 0 and nav.time_span[1] >= nav.time_span[0]
    install = wg.load_installation([fx])
    assert install is not None and install.em_model == "EM124"
    assert wg._resolve_group(install) in ("TRAI_RX1", "TRAI_TX1")


def test_build_mosaic_over_real_kmwcd():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    res = wg.build_mosaic_from_kmall(fx, projector="local", cell_m=25.0, reduce="max")
    assert res.n_pings == 1
    assert res.amplitude_db.ndim == 2
    occupied = np.isfinite(res.amplitude_db)
    assert occupied.sum() > 0
    # Deep abyssal EM124 ping -> the mapped footprint spans hundreds of metres.
    assert res.east_edges[-1] - res.east_edges[0] > 100.0
    assert "EPSG:326" in res.crs_label  # a northern-hemisphere UTM zone


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_generate_mosaic_panel(tmp_path):
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    made = wg.generate(tmp_path, mwc_files=[fx], projector="local", cell_m=25.0)
    assert made and all(p.exists() and p.stat().st_size > 0 for p in made)
    assert all("wc_mosaic_" in p.name for p in made)
