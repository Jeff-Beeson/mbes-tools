"""Tests for mbes_tools.install_params (Kongsberg install/runtime text parsing).

Synthetic strings model both schemes: .all flat ``KEY=VALUE`` and .kmall #IIP
nested ``SECTION:sub=val;sub=val``.
"""
from mbes_tools.install_params import (
    InstallationParameters,
    parse_install_sections,
    parse_install_text,
)

ALL_TEXT = (
    "WLZ=-2.070,SMH=110,S1X=3.495,S1Y=-0.139,S1Z=2.730,S1R=0.686,S1P=-0.048,"
    "S1H=0.088,S2X=1.514,S2Y=0.032,S2Z=2.729,S2R=0.714,S2P=0.285,S2H=359.892"
)
KMALL_TEXT = (
    "OSCV:Empty,EMXV:EM124,SN=10055,SYSTEM:EM 124,"
    "TRAI_TX1:N=0;X=4.221;Y=0.914;Z=6.225;R=0.060;P=-0.070;H=0.120;S=1.0,"
    "TRAI_RX1:N=0;V=;W=;X=8.558;Y=1.517;Z=6.225;R=0.000;P=-0.180;H=0.060,"
    "EMXI:SWLZ=0.740"
)


def test_parse_install_text_flat():
    p = parse_install_text(ALL_TEXT)
    assert p["WLZ"] == "-2.070"
    assert p["S1X"] == "3.495"
    assert p["S2H"] == "359.892"


def test_parse_install_sections_nested():
    s = parse_install_sections(KMALL_TEXT)
    assert s["TRAI_TX1"]["X"] == "4.221"
    assert s["TRAI_TX1"]["R"] == "0.060"
    assert s["TRAI_RX1"]["X"] == "8.558"
    assert s["EMXI"]["SWLZ"] == "0.740"
    # Flat text yields no sections.
    assert parse_install_sections(ALL_TEXT) == {}


def test_installation_parameters_all_flat():
    ip = InstallationParameters.from_text(ALL_TEXT)
    assert ip.waterline_m == -2.070
    assert ip.sections == {}
    assert ip.transducer_offsets("S1") == (3.495, -0.139, 2.730)
    assert ip.mount_angles("S1") == (0.686, -0.048, 0.088)
    assert ip.transducer_offsets("S2") == (1.514, 0.032, 2.729)
    assert ip.transducer_offsets("S9") is None  # absent group


def test_installation_parameters_kmall_nested():
    ip = InstallationParameters.from_text(KMALL_TEXT)
    assert ip.em_model == "EM124"
    assert ip.serial_number == "10055"
    assert ip.waterline_m == 0.740  # from EMXI:SWLZ
    assert ip.transducer_offsets("TRAI_TX1") == (4.221, 0.914, 6.225)
    assert ip.mount_angles("TRAI_TX1") == (0.060, -0.070, 0.120)
    assert ip.transducer_offsets("TRAI_RX1") == (8.558, 1.517, 6.225)
    assert ip.mount_angles("TRAI_RX1") == (0.000, -0.180, 0.060)


def test_raw_text_preserved():
    assert InstallationParameters.from_text(ALL_TEXT).raw == ALL_TEXT
