"""Tests for mbes_tools.diagnostics.

The numeric helpers are unit-tested on synthetic ARC rows (always run, no
matplotlib / no data). A gated end-to-end test renders panels from real files
when MBES_TEST_DATA_ROOT is set and matplotlib is importable.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mbes_tools import diagnostics as diag


def _row(angle, intensity, sector=0, depth_mode=2):
    return {"beam_angle_deg": angle, "avgIntensity_dB": intensity, "sector": sector, "depthMode": depth_mode}


def test_arc_xy_sorts_by_angle():
    rows = [_row(10, -20), _row(-30, -25), _row(0, -18)]
    a, v, s = diag.arc_xy(rows)
    assert list(a) == [-30, 0, 10]
    assert list(v) == [-25, -18, -20]


def test_port_starboard_symmetry_zero_when_symmetric():
    # Mirror-image port/starboard -> perfectly symmetric.
    rows = []
    for ang in (10, 20, 30):
        rows.append(_row(ang, -20 - ang * 0.1))    # port
        rows.append(_row(-ang, -20 - ang * 0.1))   # starboard (same |angle| value)
    assert diag.port_starboard_symmetry(rows) == pytest.approx(0.0)


def test_port_starboard_symmetry_detects_imbalance():
    rows = []
    for ang in (10, 20, 30):
        rows.append(_row(ang, -20))        # port flat at -20
        rows.append(_row(-ang, -23))       # starboard flat at -23 -> 3 dB imbalance
    assert diag.port_starboard_symmetry(rows) == pytest.approx(3.0)


def test_port_starboard_symmetry_nan_without_overlap():
    rows = [_row(10, -20), _row(20, -21)]  # port only
    assert np.isnan(diag.port_starboard_symmetry(rows))


def test_mode_mean_curve_averages_sectors_per_angle():
    rows = [
        _row(10, -20, sector=0, depth_mode=2), _row(10, -22, sector=1, depth_mode=2),
        _row(20, -25, sector=0, depth_mode=2), _row(10, -30, sector=0, depth_mode=4),
    ]
    ang, val = diag.mode_mean_curve(rows, depth_mode=2)
    assert list(ang) == [10, 20]
    assert val[0] == pytest.approx(-21.0)  # mean of -20 and -22
    assert val[1] == pytest.approx(-25.0)


def test_fold_port_starboard_splits_by_sign():
    rows = [_row(15, -20), _row(-15, -24)]
    g = diag.fold_port_starboard(rows)
    assert g["port"][0].tolist() == [15] and g["port"][1].tolist() == [-20]
    assert g["stbd"][0].tolist() == [15] and g["stbd"][1].tolist() == [-24]


# ---------------------------------------------------------------------------
# Gated end-to-end rendering against real data.
# ---------------------------------------------------------------------------

MBES_TEST_DATA_ROOT = os.environ.get("MBES_TEST_DATA_ROOT")
_HAS_MPL = __import__("importlib").util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not (MBES_TEST_DATA_ROOT and _HAS_MPL),
                    reason="needs MBES_TEST_DATA_ROOT and matplotlib")
def test_generate_writes_panels(tmp_path):
    root = Path(MBES_TEST_DATA_ROOT)
    all_files = sorted(root.rglob("*.all"))
    kmall_files = sorted(root.rglob("*.kmall"))
    made = diag.generate(
        tmp_path,
        all_file=str(all_files[0]) if all_files else None,
        kmall_file=str(kmall_files[0]) if kmall_files else None,
    )
    assert made, "no panels were produced from the discovered files"
    assert all(p.exists() and p.stat().st_size > 0 for p in made)
