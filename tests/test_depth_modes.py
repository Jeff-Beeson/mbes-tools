"""Tests for mbes_tools.depth_modes (synthetic; no data dependency)."""
from mbes_tools import depth_modes as dm


def test_em2040_mode_byte_is_frequency():
    # EM2040 / EM2045 encode frequency, not depth band, in the low bits.
    assert dm.all_runtime_mode_info(2040, 0) == (0, "200kHz")
    assert dm.all_runtime_mode_info(2040, 1) == (1, "300kHz")
    assert dm.all_runtime_mode_info(2040, 2) == (2, "400kHz")
    assert dm.all_runtime_mode_info(2045, 1) == (1, "300kHz")
    # High bits (pulse form / dual swath) must not change the frequency mode.
    assert dm.all_runtime_mode_info(2040, 0b0100_0001) == (1, "300kHz")


def test_general_model_mode_byte_is_depth_band():
    expected = {
        0: "Very Shallow",
        1: "Shallow",
        2: "Medium",
        3: "Deep",
        4: "Very Deep",
        5: "Extra Deep",
    }
    for raw, label in expected.items():
        assert dm.all_runtime_mode_info(302, raw) == (raw, label)
    # Bit combinations 6/7 collapse onto Very Deep / Extra Deep.
    assert dm.all_runtime_mode_info(302, 6) == (4, "Very Deep")
    assert dm.all_runtime_mode_info(302, 7) == (5, "Extra Deep")
    # High bits (pulse form etc.) are masked off.
    assert dm.all_runtime_mode_info(122, 0b0011_0010) == (2, "Medium")


def test_mode_id_to_calib_matches_kmall_convention():
    # Shallow=2, Medium=3, ... matching mbes_tools.backscatter.apply / normalize labels.
    assert dm.mode_id_to_calib(1) == 2
    assert dm.mode_id_to_calib(2) == 3
    assert dm.all_mode_label(2040, 1) == "300kHz"


def test_kmall_raw_to_calib_manual_and_auto():
    assert dm.kmall_raw_to_calib(101) == 2   # manual Shallow
    assert dm.kmall_raw_to_calib(106) == 7
    assert dm.kmall_raw_to_calib(1) == 2      # auto Shallow
    assert dm.kmall_raw_to_calib(5) == 6
    assert dm.kmall_depth_mode_label(1) == "Shallow"
    assert dm.kmall_depth_mode_label(5) == "Very Deep"


def test_apply_delegates_to_depth_modes():
    """apply.depth_mode_raw_to_calib stays in sync with the consolidated map."""
    from mbes_tools.backscatter import apply
    for raw in [0, 1, 5, 100, 101, 106, 108]:
        assert apply.depth_mode_raw_to_calib(raw) == dm.kmall_raw_to_calib(raw)
