"""Tests for mbes_tools.wc_diagnostics.

The numeric helpers are unit-tested synthetically (always run, no matplotlib,
no data). A gated end-to-end test renders panels from the committed real
water-column fixtures when matplotlib is importable.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import wc_diagnostics as wcd


def test_phase_scale_deg():
    assert wcd.phase_scale_deg(0) == 0.0
    assert wcd.phase_scale_deg(1) == pytest.approx(180.0 / 128.0)
    assert wcd.phase_scale_deg(2) == pytest.approx(0.01)
    # Stored integer extremes map onto ~+/-180 deg for both phase modes.
    assert 128 * wcd.phase_scale_deg(1) == pytest.approx(180.0)
    assert 18000 * wcd.phase_scale_deg(2) == pytest.approx(180.0)
    with pytest.raises(ValueError):
        wcd.phase_scale_deg(3)


def test_padded_grid_places_samples_and_pads_with_nan():
    # Two beams: beam0 starts at 0 with 3 samples, beam1 starts at 2 with 2.
    grid = wcd.padded_grid(
        starts=[0, 2], nsamps=[3, 2], value_lists=[[-100, -110, -120], [10, 20]], scale=0.5
    )
    assert grid.shape == (2, 4)  # width = max(start + n) = max(3, 4) = 4
    assert np.allclose(grid[0, :3], [-50.0, -55.0, -60.0])
    assert np.isnan(grid[0, 3])           # beam0 has no sample at position 3
    assert np.isnan(grid[1, 0]) and np.isnan(grid[1, 1])  # beam1 starts at 2
    assert np.allclose(grid[1, 2:], [5.0, 10.0])


def test_range_axis_scaling():
    # range = c * k / (2 * fs); k=2, c=1500, fs=750 -> 2.0 m at sample 2.
    r = wcd.range_axis(3, sound_speed_m_s=1500.0, sample_freq_hz=750.0)
    assert np.allclose(r, [0.0, 1.0, 2.0])


def test_wedge_coords_geometry():
    # angle 0 -> straight down (X=0, Z=range); angle +90 -> horizontal port.
    angles = np.array([0.0, 90.0])
    X, Z = wcd.wedge_coords(angles, width=3, sound_speed_m_s=1500.0, sample_freq_hz=750.0)
    r = wcd.range_axis(3, 1500.0, 750.0)
    assert np.allclose(X[0], 0.0) and np.allclose(Z[0], r)        # nadir beam
    assert np.allclose(X[1], r) and np.allclose(Z[1], 0.0, atol=1e-9)  # +90 deg port


def test_peak_sample_indices_is_nan_safe():
    grid = np.array([
        [np.nan, -50.0, -10.0, -40.0],   # peak at index 2
        [np.nan, np.nan, np.nan, np.nan],  # no finite -> 0
    ])
    assert list(wcd.peak_sample_indices(grid)) == [2, 0]


# ---------------------------------------------------------------------------
# Gated end-to-end rendering against the committed real fixtures.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_MPL = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not _HAS_MPL, reason="needs matplotlib")
def test_generate_water_column_panels(tmp_path):
    mwc = [FIXTURES / "sample_tn447_em124.kmwcd", FIXTURES / "sample_em2040_wc_phase1.kmall"]
    wcd_files = [FIXTURES / "sample_atlantis_em122.wcd"]
    mwc = [p for p in mwc if p.exists()]
    wcd_files = [p for p in wcd_files if p.exists()]
    if not mwc and not wcd_files:
        pytest.skip("water-column fixtures not present")

    made = wcd.generate(tmp_path, mwc_files=mwc, wcd_files=wcd_files)
    assert made, "no panels were produced"
    assert all(p.exists() and p.stat().st_size > 0 for p in made)
    # Each input file yields an echogram panel.
    assert sum("wc_echogram_" in p.name for p in made) == len(mwc) + len(wcd_files)
    # The phase-1 fixture (phaseFlag 1) triggers a phase panel.
    if (FIXTURES / "sample_em2040_wc_phase1.kmall").exists():
        assert any("wc_phase_" in p.name for p in made)
    # Cross-file sector-frequency sanity panel is produced.
    assert any(p.name == "wc_sector_frequencies.png" for p in made)
