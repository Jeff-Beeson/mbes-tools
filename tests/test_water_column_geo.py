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


def make_frame(amp_db, angles_deg, c=1500.0, fs=750.0, detected_samples=None):
    """A synthetic WCFrame; c=1500, fs=750 -> 1 m one-way range per sample.

    ``detected_samples`` (per beam) sets the bottom-detect sample index; the
    default (all zeros) means "no bottom detected" on every beam.
    """
    amp = np.asarray(amp_db, dtype=float)
    det = (np.zeros(amp.shape[0], dtype=int) if detected_samples is None
           else np.asarray(detected_samples, dtype=int))
    return WCFrame(
        amp_db=amp,
        phase_deg=None,
        angles_deg=np.asarray(angles_deg, dtype=float),
        detected_samples=det,
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


def geo_samples(easting, northing, depth, amp, epsg=None):
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
        epsg=epsg,
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
# 2b. Attitude (Slice-3): roll/pitch/heave.
# ---------------------------------------------------------------------------


def nav_with_attitude(heading, roll, pitch, heave, lat=0.0, lon=0.0):
    return wg.NavTrack.from_lists(
        [0.0, 1.0], [lat, lat], [lon, lon], [0.0, 1.0], [heading, heading], "t", "t",
        t_att=[0.0, 1.0], roll_deg=[roll, roll], pitch_deg=[pitch, pitch],
        heave_m=[heave, heave], attitude_source="test",
    )


def test_attitude_at_zero_when_track_has_none():
    nav = const_nav(0.0, 0.0, 90.0)
    assert nav.has_attitude is False
    assert nav.attitude_at(0.5) == (0.0, 0.0, 0.0)


def test_attitude_at_interpolates_linearly():
    nav = wg.NavTrack.from_lists(
        [0.0, 1.0], [0, 0], [0, 0], [0.0, 1.0], [0, 0], "p", "h",
        t_att=[0.0, 10.0], roll_deg=[0.0, 10.0], pitch_deg=[0.0, -2.0],
        heave_m=[0.0, 4.0], attitude_source="x")
    assert nav.has_attitude
    r, p, h = nav.attitude_at(5.0)
    assert r == pytest.approx(5.0) and p == pytest.approx(-1.0) and h == pytest.approx(2.0)


def test_dcm_reduces_to_yaw_when_level():
    d = wg._dcm(30.0, 0.0, 0.0)
    h = np.radians(30.0)
    assert np.allclose(d, [[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])


def test_georeference_stabilized_ignores_roll_but_adds_heave():
    # Kongsberg beams are vertical-stabilized -> roll must NOT move them; heave -> depth.
    frame = one_sample_frame(angle_deg=30.0, sample_k=10)  # across +5 (port)
    nav = nav_with_attitude(heading=0.0, roll=20.0, pitch=0.0, heave=3.0)
    gs = wg.georeference_frame(frame, 0.5, nav, projector="local")  # stabilized default
    assert gs.easting_m[0] == pytest.approx(-5.0, abs=1e-6)  # roll did not move the beam
    assert gs.northing_m[0] == pytest.approx(0.0, abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(np.cos(np.radians(30.0)) * 10.0 + 3.0)  # + heave
    assert gs.roll_deg == pytest.approx(20.0) and gs.heave_m == pytest.approx(3.0)


def test_georeference_unstabilized_rotates_beam_by_roll():
    frame = one_sample_frame(angle_deg=30.0, sample_k=10)
    nav = nav_with_attitude(heading=0.0, roll=20.0, pitch=0.0, heave=0.0)
    gs = wg.georeference_frame(frame, 0.5, nav, projector="local", stabilized_beams=False)
    r = np.radians(20.0)
    stbd, down = -5.0, np.cos(np.radians(30.0)) * 10.0
    assert gs.easting_m[0] == pytest.approx(stbd * np.cos(r) - down * np.sin(r), abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(stbd * np.sin(r) + down * np.cos(r), abs=1e-6)


def test_georeference_apply_attitude_false_is_heading_only():
    frame = one_sample_frame(angle_deg=30.0, sample_k=10)
    nav = nav_with_attitude(heading=0.0, roll=20.0, pitch=5.0, heave=3.0)
    gs = wg.georeference_frame(frame, 0.5, nav, projector="local", apply_attitude=False)
    assert gs.easting_m[0] == pytest.approx(-5.0, abs=1e-6)
    assert gs.depth_m[0] == pytest.approx(np.cos(np.radians(30.0)) * 10.0)  # no heave
    assert gs.roll_deg == 0.0 and gs.heave_m == 0.0


def test_georeference_rotates_lever_arm_by_full_attitude():
    # Nadir sample + starboard lever (0, 2, 0); roll 90 tips starboard -> down.
    install = InstallationParameters(raw="", params={"S1X": "0.0", "S1Y": "2.0", "S1Z": "0.0"})
    frame = one_sample_frame(angle_deg=0.0, sample_k=10)  # nadir, across 0, depth 10
    nav = nav_with_attitude(heading=0.0, roll=90.0, pitch=0.0, heave=0.0)
    gs = wg.georeference_frame(frame, 0.5, nav, projector="local", install=install)
    assert gs.easting_m[0] == pytest.approx(0.0, abs=1e-6)   # starboard lever rolled out of horizontal
    assert gs.depth_m[0] == pytest.approx(12.0, abs=1e-6)    # 10 (beam) + 2 (lever now points down)


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


def test_build_mosaic_on_uncovered_validation():
    # Bad policy is rejected before any file I/O (nav supplied, path never read).
    nav = const_nav(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="on_uncovered"):
        wg.build_mosaic_from_kmall("no_such_file.kmwcd", nav=nav, on_uncovered="bogus")


def test_build_mosaic_rejects_unknown_extension():
    nav = const_nav(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="cannot build"):
        wg.build_mosaic("survey.foo", nav=nav)


def test_composite_on_uncovered_validation():
    with pytest.raises(ValueError, match="on_uncovered"):
        wg.build_composite_mosaic([], on_uncovered="bogus")


def test_composite_skips_navless_file_without_crashing(tmp_path):
    orphan = tmp_path / "0001_x.kmwcd"
    orphan.write_bytes(b"not a datagram stream")
    res = wg.build_composite_mosaic([orphan], projector="local")  # no nav -> file skipped
    assert res.n_pings == 0 and not np.isfinite(res.amplitude_db).any()


def test_file_ping_source_rejects_unknown_extension():
    with pytest.raises(ValueError):
        wg._file_ping_source(Path("survey.foo"))


# ---------------------------------------------------------------------------
# .all-family clock: k water-column time and P nav time must share it.
# ---------------------------------------------------------------------------


class _Hdr:
    def __init__(self, date, ms):
        self.record_date = date
        self.record_time_ms = ms


def test_all_header_time_is_absolute_and_monotonic():
    import datetime
    ordi = datetime.date(2013, 7, 3).toordinal()
    assert wg._all_header_time(_Hdr(20130703, 57406690)) == pytest.approx(ordi * 86400 + 57406.690)
    # Monotonic across midnight (so interpolation doesn't jump at the wrap).
    late = wg._all_header_time(_Hdr(20130703, (23 * 3600 + 59 * 60 + 59) * 1000))
    nextday = wg._all_header_time(_Hdr(20130704, 1000))
    assert nextday > late
    # Unusable date -> falls back to seconds-since-midnight (both sources agree).
    assert wg._all_header_time(_Hdr(0, 5000)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 4. Nav-source resolution (WC files are not assumed self-contained).
# ---------------------------------------------------------------------------


def test_companion_nav_paths_maps_by_stem(tmp_path):
    kmwcd = tmp_path / "0180_survey.kmwcd"
    kmall = tmp_path / "0180_survey.kmall"
    kmwcd.write_bytes(b"")
    # No sibling yet -> nothing discovered.
    assert wg._companion_nav_paths(kmwcd) == []
    kmall.write_bytes(b"")
    assert wg._companion_nav_paths(kmwcd) == [kmall]
    # .wcd -> .all; a full-datagram file has no companion of its own.
    wcd = tmp_path / "line.wcd"
    allf = tmp_path / "line.all"
    wcd.write_bytes(b"")
    allf.write_bytes(b"")
    assert wg._companion_nav_paths(wcd) == [allf]
    assert wg._companion_nav_paths(kmall) == []


def test_resolve_nav_track_errors_without_any_source(tmp_path):
    # A lone .kmwcd with no parseable nav and no sibling -> actionable error.
    orphan = tmp_path / "0001_orphan.kmwcd"
    orphan.write_bytes(b"not a real datagram stream")
    with pytest.raises(ValueError, match="no navigation"):
        wg.resolve_nav_track(orphan)


# ---------------------------------------------------------------------------
# Gated end-to-end over the committed real EM124 .kmwcd fixture.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


def test_resolve_nav_prefers_companion_skm_over_wc_spo():
    """The committed .kmwcd has only #SPO; its same-stem .kmall companion has
    #SKM true heading, so companion discovery must upgrade the nav source."""
    kmwcd = FIXTURES / "sample_tn447_em124.kmwcd"
    kmall = FIXTURES / "sample_tn447_em124.kmall"
    if not (kmwcd.exists() and kmall.exists()):
        pytest.skip("EM124 fixture pair not present")
    # Auto companion -> nav comes from the .kmall's #SKM (true heading).
    nav_auto = wg.resolve_nav_track(kmwcd, auto_companion=True)
    assert nav_auto.position_source == "#SKM"
    # Opt out -> falls back to the .kmwcd's own #SPO course-over-ground.
    nav_own = wg.resolve_nav_track(kmwcd, auto_companion=False)
    assert nav_own.position_source == "#SPO"
    # Explicit nav_paths wins regardless.
    nav_explicit = wg.resolve_nav_track(kmwcd, nav_paths=[kmall])
    assert nav_explicit.position_source == "#SKM"


def test_skm_nav_carries_attitude():
    """#SKM (unlike #SPO) supplies roll/pitch/heave, so the track has attitude."""
    kmall = FIXTURES / "sample_tn447_em124.kmall"
    kmwcd = FIXTURES / "sample_tn447_em124.kmwcd"
    if not (kmall.exists() and kmwcd.exists()):
        pytest.skip("EM124 fixture pair not present")
    nav_skm = wg.nav_track_from_kmall([kmall])
    assert nav_skm.position_source == "#SKM" and nav_skm.has_attitude
    assert nav_skm.attitude_source == "#SKM"
    r, p, h = nav_skm.attitude_at(nav_skm.time_span[0])
    assert np.isfinite([r, p, h]).all() and abs(r) < 45 and abs(p) < 45
    # The #SPO fallback (WC file, no #SKM) has no attitude.
    nav_spo = wg.resolve_nav_track(kmwcd, auto_companion=False)
    assert not nav_spo.has_attitude


def test_resolve_nav_track_errors_on_navless_wcd():
    """A bare .wcd carries no P position datagram, so nav must come elsewhere."""
    wcd = FIXTURES / "sample_atlantis_em122.wcd"
    if not wcd.exists():
        pytest.skip("wcd fixture not present")
    # No same-stem .all sibling committed -> resolution fails with guidance.
    with pytest.raises(ValueError, match="no navigation"):
        wg.resolve_nav_track(wcd)


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


def test_build_mosaic_skips_ping_when_nav_does_not_cover_it():
    """The committed .kmwcd/.kmall are independent clips (~2.8 h apart), so the
    #MWC ping is outside every committed nav span. The coverage guard must skip
    it (and warn) rather than silently clamp it tens of km away."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    with pytest.warns(UserWarning, match="did not cover"):
        res = wg.build_mosaic_from_kmall(fx, projector="local", cell_m=25.0)  # default skip
    assert res.n_uncovered == 1 and res.n_pings == 0
    assert not np.isfinite(res.amplitude_db).any()


def test_build_mosaic_clamp_grids_real_ping():
    """With on_uncovered='clamp' the (clip-edge) ping is still gridded, so this
    exercises the full georeference->grid geometry path on real data."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    res = wg.build_mosaic_from_kmall(
        fx, projector="local", cell_m=25.0, reduce="max",
        auto_companion=False, on_uncovered="clamp",  # its own #SPO, clamped over the 18.8 s clip gap
    )
    assert res.n_pings == 1 and res.n_uncovered == 1
    assert res.amplitude_db.ndim == 2 and np.isfinite(res.amplitude_db).any()
    # Deep abyssal EM124 ping -> the mapped footprint spans hundreds of metres.
    assert res.east_edges[-1] - res.east_edges[0] > 100.0
    assert "EPSG:326" in res.crs_label  # a northern-hemisphere UTM zone


def test_build_mosaic_from_wcd_over_real_fixture():
    """The 3-ping EM122 .wcd fixture has no position of its own, so nav is
    supplied; this exercises k-datagram reassembly -> georef -> mosaic."""
    wcd = FIXTURES / "sample_atlantis_em122.wcd"
    if not wcd.exists():
        pytest.skip("wcd fixture not present")
    from mbes_tools.wcd import iter_water_column_datagrams
    from mbes_tools.water_column import reassemble_wcd_pings

    pings = list(reassemble_wcd_pings(iter_water_column_datagrams(wcd)))
    times = [wg._all_header_time(p.header) for p in pings]
    t0, t1 = min(times) - 1.0, max(times) + 1.0
    nav = wg.NavTrack.from_lists([t0, t1], [30.0, 30.0], [-45.0, -45.0],
                                 [t0, t1], [45.0, 45.0], "synthetic", "synthetic")
    res = wg.build_mosaic_from_wcd(wcd, nav=nav, projector="local", cell_m=25.0, reduce="max")
    assert res.n_pings == 3 and res.n_uncovered == 0
    assert np.isfinite(res.amplitude_db).any()
    assert "EPSG:326" in res.crs_label  # UTM 23N for the 30N/-45E synthetic anchor
    # The extension dispatcher routes .wcd to the same builder.
    via_dispatch = wg.build_mosaic(wcd, nav=nav, projector="local", cell_m=25.0)
    assert via_dispatch.n_pings == 3


def test_composite_accumulates_multiple_files_into_one_grid():
    """Two files -> one mosaic with a shared anchor. Feeding the same real ping
    twice must land in the same cells (n_pings=2, geometry unchanged)."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    single = wg.build_mosaic_from_kmall(
        fx, projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    comp = wg.build_composite_mosaic(
        [fx, fx], projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    assert comp.n_pings == 2  # both files contributed a ping
    # Shared anchor -> identical footprint; peak-hold value unchanged by duplication.
    assert comp.amplitude_db.shape == single.amplitude_db.shape
    assert np.array_equal(np.isfinite(comp.amplitude_db), np.isfinite(single.amplitude_db))


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_generate_mosaic_panel(tmp_path):
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    made = wg.generate(tmp_path, mwc_files=[fx], projector="local", cell_m=25.0,
                       auto_companion=False, on_uncovered="clamp")
    assert made and all(p.exists() and p.stat().st_size > 0 for p in made)
    assert all("wc_mosaic_" in p.name for p in made)


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_generate_combine_makes_single_composite_panel(tmp_path):
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    made = wg.generate(tmp_path, mwc_files=[fx, fx], combine=True, projector="local",
                       cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    assert len(made) == 1 and "composite" in made[0].name
    assert made[0].exists() and made[0].stat().st_size > 0


# ---------------------------------------------------------------------------
# 5. Raster export: EPSG threading, ESRI ASCII (numpy-only), GeoTIFF (rasterio).
# ---------------------------------------------------------------------------

_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None
_HAS_PYPROJ = wg._pyproj_available()


def _gap_mosaic(epsg=None):
    """2x2 mosaic with cells (0,0)=-30 (south-west) and (1,1)=-10 (north-east);
    the other two cells stay NaN. cell_m=10 -> edges [0,10,20]."""
    m = wg.GeoMosaic(cell_m=10.0, reduce="max")
    m.add(geo_samples([5.0, 15.0], [5.0, 15.0], [100.0, 100.0], [-30.0, -10.0], epsg=epsg))
    return m.finalize()


def test_epsg_threads_into_result_from_geosamples():
    # GeoMosaic captures epsg from the first ping; finalize propagates it.
    assert _gap_mosaic(epsg=32610).epsg == 32610
    assert _gap_mosaic(epsg=None).epsg is None


@pytest.mark.skipif(not _HAS_PYPROJ, reason="needs pyproj")
def test_georeference_utm_sets_epsg_but_local_does_not():
    frame = one_sample_frame(angle_deg=0.0, sample_k=5)
    nav = const_nav(6.90354, 126.97404, 0.0)  # UTM 52N
    assert wg.georeference_frame(frame, 0.0, nav, projector="utm").epsg == 32652
    assert wg.georeference_frame(frame, 0.0, nav, projector="local").epsg is None


def test_export_ascii_grid_orientation_and_header(tmp_path):
    res = _gap_mosaic()
    path = wg.export_ascii_grid(res, tmp_path / "wc_mosaic_x")
    assert path.suffix == ".asc" and path.exists()

    lines = path.read_text().splitlines()
    hdr = {p.split()[0]: p.split()[1] for p in lines[:6]}
    assert hdr["ncols"] == "2" and hdr["nrows"] == "2"
    assert float(hdr["xllcorner"]) == pytest.approx(0.0)
    assert float(hdr["yllcorner"]) == pytest.approx(0.0)
    assert float(hdr["cellsize"]) == pytest.approx(10.0)
    nodata = float(hdr["NODATA_value"])

    body = np.loadtxt(tmp_path / "wc_mosaic_x.asc", skiprows=6)
    # North-up: row 0 is the northernmost. NE cell (-10) is top-right; SW cell
    # (-30) is bottom-left; the two off-diagonal cells are NODATA.
    assert body[0, 1] == pytest.approx(-10.0)
    assert body[1, 0] == pytest.approx(-30.0)
    assert body[0, 0] == pytest.approx(nodata)
    assert body[1, 1] == pytest.approx(nodata)


def test_export_ascii_grid_prj_only_when_projected(tmp_path):
    # Local frame (epsg None) -> no .prj sidecar.
    wg.export_ascii_grid(_gap_mosaic(epsg=None), tmp_path / "local")
    assert not (tmp_path / "local.prj").exists()
    if _HAS_PYPROJ:
        wg.export_ascii_grid(_gap_mosaic(epsg=32610), tmp_path / "utm")
        assert (tmp_path / "utm.prj").exists()
        assert "32610" in (tmp_path / "utm.prj").read_text() or "UTM zone 10N" in (tmp_path / "utm.prj").read_text()


def test_export_geotiff_requires_projected_crs(tmp_path):
    # epsg=None (local frame) -> refuses before touching rasterio.
    with pytest.raises(RuntimeError, match="projected CRS"):
        wg.export_geotiff(_gap_mosaic(epsg=None), tmp_path / "local")


@pytest.mark.skipif(not _HAS_RASTERIO, reason="needs rasterio")
def test_export_geotiff_roundtrip(tmp_path):
    import rasterio
    res = _gap_mosaic(epsg=32610)
    path = wg.export_geotiff(res, tmp_path / "wc_mosaic_x")
    assert path.suffix == ".tif" and path.exists()
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == 32610
        assert ds.width == 2 and ds.height == 2 and ds.count == 1
        # Bounds match the mosaic edges.
        assert ds.bounds.left == pytest.approx(0.0)
        assert ds.bounds.bottom == pytest.approx(0.0)
        assert ds.bounds.right == pytest.approx(20.0)
        assert ds.bounds.top == pytest.approx(20.0)
        band = ds.read(1)
        # North-up (row 0 = north): NE cell (-10) top-right, SW cell (-30) bottom-left.
        assert band[0, 1] == pytest.approx(-10.0)
        assert band[1, 0] == pytest.approx(-30.0)
        assert np.isnan(band[0, 0]) and np.isnan(band[1, 1])


def test_generate_writes_asc_alongside(tmp_path):
    # The .asc writer is numpy-only, so this runs even without matplotlib
    # (the PNG panel fails-soft; the raster is still written).
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    made = wg.generate(tmp_path, mwc_files=[fx], projector="local", cell_m=25.0,
                       auto_companion=False, on_uncovered="clamp", write_asc=True)
    ascs = [p for p in made if p.suffix == ".asc"]
    assert ascs and ascs[0].exists() and ascs[0].stat().st_size > 0


# ---------------------------------------------------------------------------
# 6. Height-above-seafloor band (Slice 2).
# ---------------------------------------------------------------------------


def bottom_frame(angle_deg, samples_amp, det_sample):
    """One-beam frame with a detected bottom at ``det_sample`` and finite samples.

    ``samples_amp`` is a dict {sample_k: amp_db}. With c=1500/fs=750 a sample at
    range-sample k sits at one-way range k m; the seafloor depth for angle a is
    cos(a)*det_sample, so a sample's height above bottom is cos(a)*(det - k).
    """
    width = max(max(samples_amp), det_sample) + 1
    grid = np.full((1, width), np.nan)
    for k, v in samples_amp.items():
        grid[0, k] = v
    return make_frame(grid, [angle_deg], detected_samples=[det_sample])


def test_georeference_computes_height_above_seafloor():
    # Nadir beam, bottom at sample 50 -> Zb = 50 m. Samples at 30 and 45 ->
    # HAB 20 and 5 m. (heading/attitude irrelevant to the scalar HAB.)
    frame = bottom_frame(angle_deg=0.0, samples_amp={30: -20.0, 45: -25.0}, det_sample=50)
    nav = const_nav(0.0, 0.0, 0.0)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    hab = gs.height_above_seafloor_m
    assert hab is not None and hab.shape == gs.depth_m.shape
    # Samples come out in ascending sample order (30 then 45).
    order = np.argsort(gs.depth_m)
    assert hab[order][0] == pytest.approx(20.0)
    assert hab[order][1] == pytest.approx(5.0)


def test_georeference_hab_is_nan_when_no_bottom():
    # det_sample 0 -> "no bottom": HAB must be NaN so altitude bands drop it.
    frame = one_sample_frame(angle_deg=0.0, sample_k=10)  # detected_samples defaults to 0
    nav = const_nav(0.0, 0.0, 0.0)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    assert gs.height_above_seafloor_m is not None
    assert np.isnan(gs.height_above_seafloor_m).all()


def test_georeference_hab_uses_beam_angle_for_seafloor_depth():
    # A 60-deg beam with bottom at sample 50: seafloor depth = cos(60)*50 = 25 m.
    # A sample at range-sample 20 sits at depth cos(60)*20 = 10 m -> HAB 15 m.
    frame = bottom_frame(angle_deg=60.0, samples_amp={20: -20.0}, det_sample=50)
    nav = const_nav(0.0, 0.0, 0.0)
    gs = wg.georeference_frame(frame, 0.0, nav, projector="local")
    assert gs.depth_m[0] == pytest.approx(np.cos(np.radians(60.0)) * 20.0)
    assert gs.height_above_seafloor_m[0] == pytest.approx(np.cos(np.radians(60.0)) * 30.0)


def _hab_samples(depth, amp, hab, epsg=None):
    """A GeoSamples carrying explicit height_above_seafloor_m (single cell (0,0))."""
    gs = geo_samples([1.0] * len(depth), [1.0] * len(depth), depth, amp, epsg=epsg)
    gs.height_above_seafloor_m = np.asarray(hab, float)
    return gs


def test_mosaic_altitude_band_filters_by_height_above_bottom():
    m = wg.GeoMosaic(cell_m=10.0, reduce="max", altitude_band=(20.0, 200.0))
    # Three returns to the same cell: in-band (HAB 100), below-band (HAB 5, brighter),
    # and no-bottom (HAB NaN, brightest). Only the in-band one survives.
    m.add(_hab_samples([300.0, 480.0, 480.0], [-30.0, -5.0, -1.0], [100.0, 5.0, np.nan]))
    res = m.finalize()
    assert res.amplitude_db[0, 0] == pytest.approx(-30.0)
    assert res.altitude_band == (20.0, 200.0)


def test_mosaic_altitude_and_depth_bands_compose():
    # depth in [0,400] AND height-above-bottom in [50,150]. Only sample 0 passes both.
    m = wg.GeoMosaic(cell_m=10.0, reduce="max",
                     depth_band=(0.0, 400.0), altitude_band=(50.0, 150.0))
    m.add(_hab_samples(
        depth=[300.0, 500.0, 350.0],   # sample 1 fails depth band
        amp=[-30.0, -1.0, -2.0],
        hab=[100.0, 100.0, 10.0],      # sample 2 fails altitude band
    ))
    res = m.finalize()
    assert res.amplitude_db[0, 0] == pytest.approx(-30.0)


def test_mosaic_altitude_band_requires_hab_data():
    # A GeoSamples with no HAB array but an altitude band set -> clear error.
    m = wg.GeoMosaic(cell_m=10.0, altitude_band=(0.0, 100.0))
    gs = geo_samples([1.0], [1.0], [100.0], [-20.0])  # height_above_seafloor_m is None
    with pytest.raises(ValueError, match="height_above_seafloor_m"):
        m.add(gs)


def test_build_mosaic_altitude_band_clamp_grids_real_ping():
    """The altitude band flows end-to-end through the builder on the real EM124
    ping. A wide band keeps near-bottom returns; the mosaic still grids."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    res = wg.build_mosaic_from_kmall(
        fx, projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp",
        altitude_band=(0.0, 500.0),
    )
    assert res.altitude_band == (0.0, 500.0)
    assert res.n_pings == 1 and res.amplitude_db.ndim == 2


# ---------------------------------------------------------------------------
# 7. Performance: vectorized map-reduce accumulator + parallel composite (Slice 3).
# ---------------------------------------------------------------------------


def _reference_mosaic(pings, cell_m, reduce):
    """A dict-based reference accumulator (the pre-Slice-3 algorithm) for a list
    of GeoSamples, used to assert the buffered accumulator is bit-identical."""
    cells = {}
    for gs in pings:
        e, n, a = gs.easting_m, gs.northing_m, gs.amplitude_db
        ie = np.floor(e / cell_m).astype(np.int64)
        jn = np.floor(n / cell_m).astype(np.int64)
        uniq, inv = np.unique(np.column_stack([ie, jn]), axis=0, return_inverse=True)
        inv = inv.ravel()
        if reduce == "max":
            best = np.full(uniq.shape[0], -np.inf)
            np.maximum.at(best, inv, a)
            for (ke, kn), v in zip(map(tuple, uniq.tolist()), best.tolist()):
                cur = cells.get((ke, kn))
                if cur is None or v > cur:
                    cells[(ke, kn)] = v
        else:
            lin = np.power(10.0, a / 10.0)
            sums = np.zeros(uniq.shape[0]); np.add.at(sums, inv, lin)
            cnts = np.bincount(inv, minlength=uniq.shape[0])
            for (ke, kn), s, c in zip(map(tuple, uniq.tolist()), sums.tolist(), cnts.tolist()):
                cur = cells.get((ke, kn))
                if cur is None:
                    cells[(ke, kn)] = [s, c]
                else:
                    cur[0] += s; cur[1] += c
    return cells


def _random_pings(n_pings, seed=0):
    """Synthetic GeoSamples spread over a few overlapping cells (dense overlap so
    max/mean reductions actually combine many samples per cell)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_pings):
        k = int(rng.integers(20, 60))
        e = rng.uniform(0, 120, k)
        n = rng.uniform(0, 120, k)
        a = rng.uniform(-60, -5, k)
        out.append(geo_samples(e, n, np.full(k, 100.0), a))
    return out


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_accumulator_is_bit_identical_to_reference(reduce):
    pings = _random_pings(40, seed=3)
    m = wg.GeoMosaic(cell_m=10.0, reduce=reduce)
    for gs in pings:
        m.add(gs)
    res = m.finalize()

    ref = _reference_mosaic(pings, 10.0, reduce)
    # Compare cell-by-cell against the dict reference, bit-for-bit.
    ie_min = min(k[0] for k in ref)
    jn_min = min(k[1] for k in ref)
    for (ke, kn), v in ref.items():
        row, col = kn - jn_min, ke - ie_min
        expected = v if reduce == "max" else 10.0 * np.log10(v[0] / v[1])
        assert res.amplitude_db[row, col] == expected  # exact, not approx


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_compaction_is_bit_identical(reduce):
    # A tiny compaction threshold forces mid-stream compaction; the result must
    # match a single-shot accumulation exactly (compaction preserves order).
    pings = _random_pings(30, seed=7)
    big = wg.GeoMosaic(cell_m=10.0, reduce=reduce, compact_rows=10**9)
    small = wg.GeoMosaic(cell_m=10.0, reduce=reduce, compact_rows=50)
    for gs in pings:
        big.add(gs); small.add(gs)
    rb, rs = big.finalize(), small.finalize()
    assert rb.amplitude_db.shape == rs.amplitude_db.shape
    np.testing.assert_array_equal(  # exact
        np.nan_to_num(rb.amplitude_db, nan=-999), np.nan_to_num(rs.amplitude_db, nan=-999)
    )


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_add_partial_matches_add(reduce):
    # Feeding rows via add_partial (the parallel merge primitive) must match add().
    pings = _random_pings(12, seed=11)
    direct = wg.GeoMosaic(cell_m=10.0, reduce=reduce)
    for gs in pings:
        direct.add(gs)
    viap = wg.GeoMosaic(cell_m=10.0, reduce=reduce)
    for gs in pings:
        ie, jn, val, cnt = wg._ping_cell_partials(gs, 10.0, reduce, None, None)
        viap.add_partial(ie, jn, val, cnt, n_pings=1)
    rd, rp = direct.finalize(), viap.finalize()
    np.testing.assert_array_equal(
        np.nan_to_num(rd.amplitude_db, nan=-999), np.nan_to_num(rp.amplitude_db, nan=-999)
    )


@pytest.mark.parametrize("reduce", ["max", "mean"])
def test_parallel_composite_matches_serial(reduce):
    """workers=2 must produce a bit-identical mosaic to workers=1 on real data."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False,
              on_uncovered="clamp", reduce=reduce)
    serial = wg.build_composite_mosaic([fx, fx], workers=1, **kw)
    parallel = wg.build_composite_mosaic([fx, fx], workers=2, **kw)
    assert parallel.n_pings == serial.n_pings == 2
    assert parallel.amplitude_db.shape == serial.amplitude_db.shape
    np.testing.assert_array_equal(  # bit-for-bit
        np.nan_to_num(serial.amplitude_db, nan=-999),
        np.nan_to_num(parallel.amplitude_db, nan=-999),
    )
    np.testing.assert_array_equal(serial.counts, parallel.counts)


# ---------------------------------------------------------------------------
# 8. Empirical normalization wired through the mosaic (Slice 4a).
# ---------------------------------------------------------------------------


def test_normalize_none_is_default_and_identity():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    default = wg.build_mosaic_from_kmall(fx, **kw)
    explicit_none = wg.build_mosaic_from_kmall(fx, normalize="none", **kw)
    np.testing.assert_array_equal(
        np.nan_to_num(default.amplitude_db, nan=-999),
        np.nan_to_num(explicit_none.amplitude_db, nan=-999),
    )


def test_normalize_empirical_keeps_footprint_changes_values():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    raw = wg.build_mosaic_from_kmall(fx, **kw)
    norm = wg.build_mosaic_from_kmall(fx, normalize="empirical", **kw)
    # Normalization changes amplitudes but not the geometry / occupied cells.
    assert np.array_equal(np.isfinite(raw.amplitude_db), np.isfinite(norm.amplitude_db))
    assert not np.allclose(
        np.nan_to_num(raw.amplitude_db), np.nan_to_num(norm.amplitude_db)
    )


def test_normalize_parallel_matches_serial():
    """--normalize must survive the process-pool path bit-for-bit vs serial."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False,
              on_uncovered="clamp", normalize="empirical")
    serial = wg.build_composite_mosaic([fx, fx], workers=1, **kw)
    parallel = wg.build_composite_mosaic([fx, fx], workers=2, **kw)
    np.testing.assert_array_equal(
        np.nan_to_num(serial.amplitude_db, nan=-999),
        np.nan_to_num(parallel.amplitude_db, nan=-999),
    )


# ---------------------------------------------------------------------------
# 9. Minimum-slant-range (--clean-water) filter wired through the mosaic.
# ---------------------------------------------------------------------------


def test_clean_water_drops_beyond_nadir_samples():
    """--clean-water must remove the below-nadir-range returns, so the mosaic keeps
    strictly fewer cells than the unfiltered build (the seafloor + sidelobe zone
    is what gets cut)."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    full = wg.build_mosaic_from_kmall(fx, **kw)
    clean = wg.build_mosaic_from_kmall(fx, clean_water=True, **kw)
    n_full = int(np.isfinite(full.amplitude_db).sum())
    n_clean = int(np.isfinite(clean.amplitude_db).sum())
    assert 0 < n_clean < n_full  # some clean water remains, but the deep zone is gone
    # A guard pulls the cutoff further in -> no more cells than the unguarded clean cut.
    guarded = wg.build_mosaic_from_kmall(fx, clean_water=True, msr_guard_m=200.0, **kw)
    assert int(np.isfinite(guarded.amplitude_db).sum()) <= n_clean


def test_clean_water_parallel_matches_serial():
    """--clean-water must survive the process-pool path bit-for-bit vs serial."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False,
              on_uncovered="clamp", clean_water=True)
    serial = wg.build_composite_mosaic([fx, fx], workers=1, **kw)
    parallel = wg.build_composite_mosaic([fx, fx], workers=2, **kw)
    np.testing.assert_array_equal(
        np.nan_to_num(serial.amplitude_db, nan=-999),
        np.nan_to_num(parallel.amplitude_db, nan=-999),
    )


# ---------------------------------------------------------------------------
# 10. Relative-Sv normalization (--normalize sv) wired through the mosaic.
# ---------------------------------------------------------------------------


def test_frame_from_mwc_carries_applied_tvg():
    """The real #MWC frame must carry TVGfunctionApplied/TVGoffset so Sv works."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    from mbes_tools.kmwcd import iter_mwc_datagrams
    from mbes_tools.wc_diagnostics import frame_from_mwc
    frame = frame_from_mwc(next(iter_mwc_datagrams(fx)))
    assert frame.tvg_function_x is not None and frame.tvg_offset_db is not None
    assert frame.tvg_function_x > 0  # a real X log R constant (e.g. 30)


def test_normalize_sv_keeps_footprint_changes_values():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp")
    raw = wg.build_mosaic_from_kmall(fx, **kw)
    sv = wg.build_mosaic_from_kmall(fx, normalize="sv", **kw)
    assert np.array_equal(np.isfinite(raw.amplitude_db), np.isfinite(sv.amplitude_db))
    assert not np.allclose(np.nan_to_num(raw.amplitude_db), np.nan_to_num(sv.amplitude_db))


def test_normalize_sv_absorption_boosts_deep_returns():
    """+2·alpha·R grows with range, so alpha>0 shifts the (deep) mosaic upward vs alpha=0."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp",
              normalize="sv")
    a0 = wg.build_mosaic_from_kmall(fx, absorption_db_km=0.0, **kw)
    a8 = wg.build_mosaic_from_kmall(fx, absorption_db_km=8.0, **kw)
    assert np.nanmedian(a8.amplitude_db) > np.nanmedian(a0.amplitude_db)


def test_normalize_sv_parallel_matches_serial():
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    kw = dict(projector="local", cell_m=25.0, auto_companion=False, on_uncovered="clamp",
              normalize="sv", absorption_db_km=8.0)
    serial = wg.build_composite_mosaic([fx, fx], workers=1, **kw)
    parallel = wg.build_composite_mosaic([fx, fx], workers=2, **kw)
    np.testing.assert_array_equal(
        np.nan_to_num(serial.amplitude_db, nan=-999),
        np.nan_to_num(parallel.amplitude_db, nan=-999),
    )


# ---------------------------------------------------------------------------
# 11. Gap fill / nodata interpolation (--fill-nodata).
# ---------------------------------------------------------------------------


def _grid_result(amp):
    amp = np.asarray(amp, float)
    h, w = amp.shape
    return wg.GeoMosaicResult(
        amplitude_db=amp, counts=np.where(np.isfinite(amp), 1, 0).astype(int),
        east_edges=np.arange(w + 1.0), north_edges=np.arange(h + 1.0),
        cell_m=1.0, reduce="mean", crs_label="test", n_pings=1, depth_band=None,
    )


def test_fill_nodata_fills_small_hole_keeps_count_zero():
    amp = np.full((5, 5), -20.0)
    amp[2, 2] = np.nan
    out = wg.fill_mosaic_nodata(_grid_result(amp), 1)
    assert out.amplitude_db[2, 2] == pytest.approx(-20.0)   # linear mean of equal neighbours
    assert out.counts[2, 2] == 0                            # interpolated, not observed


def test_fill_nodata_is_distance_bounded():
    amp = np.full((7, 7), np.nan)
    amp[0, :] = -30.0                                       # only the top row has data
    r = _grid_result(amp)
    near = wg.fill_mosaic_nodata(r, 1)
    far = wg.fill_mosaic_nodata(r, 3)
    assert np.isfinite(near.amplitude_db[1, 3])            # 1 cell away -> filled at dist 1
    assert not np.isfinite(near.amplitude_db[3, 3])        # 3 cells away -> still nodata
    assert np.isfinite(far.amplitude_db[3, 3])            # reached at dist 3


def test_fill_nodata_zero_is_identity():
    amp = np.array([[-20.0, np.nan], [np.nan, -22.0]])
    out = wg.fill_mosaic_nodata(_grid_result(amp), 0)
    np.testing.assert_array_equal(np.nan_to_num(out.amplitude_db, nan=-999),
                                  np.nan_to_num(amp, nan=-999))


def test_fill_nodata_averages_in_linear_domain():
    # Two neighbours -10 and -20 dB: linear mean -> 10log10((0.1+0.01)/2) ≈ -12.6 dB,
    # which is > the dB mean (-15), confirming linear-domain averaging.
    amp = np.array([[-10.0, np.nan, -20.0]])
    out = wg.fill_mosaic_nodata(_grid_result(amp), 1)
    expected = 10 * np.log10((10 ** (-1.0) + 10 ** (-2.0)) / 2)
    assert out.amplitude_db[0, 1] == pytest.approx(expected)
    assert out.amplitude_db[0, 1] > -15.0


def test_generate_fill_nodata_reduces_holes(tmp_path):
    """End-to-end: --fill-nodata leaves >= as many finite cells as the raw mosaic,
    written into the .asc raster."""
    fx = FIXTURES / "sample_tn447_em124.kmwcd"
    if not fx.exists():
        pytest.skip("kmwcd fixture not present")
    raw = wg.build_mosaic_from_kmall(fx, projector="local", cell_m=25.0,
                                     auto_companion=False, on_uncovered="clamp")
    n_raw = int(np.isfinite(raw.amplitude_db).sum())
    filled = wg.fill_mosaic_nodata(raw, 2)
    n_filled = int(np.isfinite(filled.amplitude_db).sum())
    assert n_filled >= n_raw
    # Observed cells are unchanged; only NaNs may become finite.
    obs = np.isfinite(raw.amplitude_db)
    np.testing.assert_allclose(filled.amplitude_db[obs], raw.amplitude_db[obs])
