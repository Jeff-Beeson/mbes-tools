"""Tests for mbes_tools.beam_stat (pure numpy; no data dependency)."""
import numpy as np
import pytest

from mbes_tools import beam_stat as bs


def test_window_bounds_full_and_windowed():
    assert bs.window_bounds(100, 40, None) == (0, 100)      # full beam
    assert bs.window_bounds(100, 40, 5) == (35, 46)         # +/-5 around centre
    assert bs.window_bounds(100, 2, 5) == (0, 8)            # clamped low
    assert bs.window_bounds(100, 98, 5) == (93, 100)        # clamped high
    assert bs.window_bounds(100, None, 5) == (45, 56)       # missing centre -> midpoint
    assert bs.window_bounds(0, 0, 5) == (0, 0)              # empty beam


def test_clean_db_drops_sentinels_and_scales():
    out = bs.clean_db([-250, -260, -32767, -32768, -300])
    # -32767 and -32768 dropped; remaining scaled by 0.1
    assert np.allclose(out, [-25.0, -26.0, -30.0])


def test_reducers_on_known_samples():
    # deci-dB samples -> dB: -10, -20, -30, -40
    samples = [-100, -200, -300, -400]
    assert bs.reduce_beam(samples, None, None, "mean") == pytest.approx(-25.0)
    assert bs.reduce_beam(samples, None, None, "median") == pytest.approx(-25.0)
    assert bs.reduce_beam(samples, None, None, "min") == pytest.approx(-40.0)
    assert bs.reduce_beam(samples, None, None, "max") == pytest.approx(-10.0)
    assert bs.reduce_beam(samples, None, None, "range") == pytest.approx(30.0)
    assert bs.reduce_beam(samples, None, None, "count") == pytest.approx(4.0)
    assert bs.reduce_beam(samples, None, None, "std") == pytest.approx(
        float(np.std([-10.0, -20.0, -30.0, -40.0]))
    )


def test_percentile_and_mode_and_trimmed():
    samples = [-100, -100, -200, -300, -400, -500]  # -10,-10,-20,-30,-40,-50 dB
    assert bs.reduce_beam(samples, None, None, "mode") == pytest.approx(-10.0)
    assert bs.reduce_beam(samples, None, None, "p50") == pytest.approx(
        float(np.percentile([-10, -10, -20, -30, -40, -50], 50))
    )
    # trimmed mean (10%) of 6 samples trims 0 each side -> same as mean here
    assert bs.reduce_beam(samples, None, None, "trimmed_mean") == pytest.approx(
        float(np.mean([-10, -10, -20, -30, -40, -50]))
    )


def test_window_selects_subset_around_centre():
    # 9 samples; centre index 4; window +/-1 -> indices 3,4,5
    samples = [-10, -20, -30, -40, -50, -60, -70, -80, -90]
    # values at idx 3,4,5 = -4,-5,-6 dB -> mean -5.0
    assert bs.reduce_beam(samples, 4, 1, "mean") == pytest.approx(-5.0)


def test_reduce_multi_matches_individual():
    samples = [-100, -200, -300, -400, -500]
    multi = bs.reduce_beam_multi(samples, None, None, ["mean", "std", "p90"])
    assert multi["mean"] == pytest.approx(bs.reduce_beam(samples, None, None, "mean"))
    assert multi["std"] == pytest.approx(bs.reduce_beam(samples, None, None, "std"))
    assert multi["p90"] == pytest.approx(bs.reduce_beam(samples, None, None, "p90"))


def test_empty_window_returns_nan():
    assert np.isnan(bs.reduce_beam([], None, None, "mean"))
    assert np.isnan(bs.reduce_beam([-32767, -32768], None, None, "mean"))  # all sentinels
    multi = bs.reduce_beam_multi([], None, None, ["mean", "median"])
    assert all(np.isnan(v) for v in multi.values())


def test_unknown_stat_raises():
    with pytest.raises(ValueError):
        bs.get_reducer("bogus")
    with pytest.raises(ValueError):
        bs.get_reducer("p150")  # out of range
