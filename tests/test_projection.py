"""Tests for mbes_tools.projection (pure; no pyproj dependency)."""
import pytest

from mbes_tools import projection as proj


def test_utm_epsg_hemispheres_and_zones():
    # Monterey Bay -> UTM 10N = 32610; southern flips to 327xx.
    assert proj.utm_epsg_from_lonlat(-122.6, 36.4) == 32610
    assert proj.utm_epsg_from_lonlat(-122.6, -36.4) == 32710
    # Samoa critical-minerals area -> UTM 2S = 32702 (plan says 2S/3S).
    assert proj.utm_epsg_from_lonlat(-171.8, -13.8) == 32702


def test_utm_epsg_antimeridian_safe():
    assert proj.utm_epsg_from_lonlat(179.9, -14.0) == 32760   # zone 60S
    assert proj.utm_epsg_from_lonlat(181.0, 10.0) == 32601    # wraps to zone 1N
    assert proj.utm_epsg_from_lonlat(-180.0, 10.0) == 32601


def test_polar_falls_back_to_ups():
    assert proj.utm_epsg_from_lonlat(10.0, 88.0) == proj.EPSG_UPS_NORTH
    assert proj.utm_epsg_from_lonlat(10.0, -85.0) == proj.EPSG_UPS_SOUTH


def test_resolve_target_crs_modes():
    assert proj.resolve_target_crs("auto", -171.8, -13.8) == "EPSG:32702"
    assert proj.resolve_target_crs(None, -122.6, 36.4) == "EPSG:32610"
    assert proj.resolve_target_crs(32756) == "EPSG:32756"
    assert proj.resolve_target_crs("32602") == "EPSG:32602"
    assert proj.resolve_target_crs("EPSG:32602") == "EPSG:32602"
    assert proj.resolve_target_crs("+proj=utm +zone=2 +south") == "+proj=utm +zone=2 +south"


def test_resolve_auto_requires_position():
    with pytest.raises(ValueError):
        proj.resolve_target_crs("auto")
    with pytest.raises(ValueError):
        proj.utm_epsg_from_lonlat(float("nan"), 10.0)
