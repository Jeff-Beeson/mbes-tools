"""Depth-mode normalization maps across Kongsberg models and file formats.

Backscatter angular response is grouped by *depth mode* (a.k.a. ping mode),
because the sonar's transmit characteristics change between modes. The trouble
is that the depth-mode encoding differs by **file format** and by **EM model**:

- **.kmall** stores a ``depthMode`` integer directly in the #MRZ pingInfo.
  Manual selection adds a +100 offset (handled in :mod:`mbes_tools.kmall`).
- **.all** stores a ``mode`` *byte* in the ``R`` runtime datagram whose meaning
  is bit-encoded and **model-specific**. For most EM models the low 3 bits give
  a depth mode; for **EM2040 / EM2045** the low bits select the *frequency*
  (200 / 300 / 400 kHz) rather than a depth band.

This module is the single documented place that maps each (model, format)
encoding onto a canonical mode id + human label, and onto the calibration-file
mode numbering used by :mod:`mbes_tools.backscatter.apply` (calib = id + 1).

Canonical ids follow the KMALL auto-mode ladder::

    0 Very Shallow   2 Medium   4 Deeper      6 Extra Deep
    1 Shallow        3 Deep     5 Very Deep   7 Extreme Deep

Note the **.all general ladder has no "Deeper" step**: its modes are
Very Shallow / Shallow / Medium / Deep / Very Deep / Extra Deep. So a .all
"Very Deep" (canonical id 4 here) and a .kmall "Very Deep" (id 5) are *not* the
same integer. Within a single survey/sonar this is self-consistent; only when
comparing .all and .kmall tables across the gap above "Deep" does it matter, and
it is documented here rather than silently coerced.
"""

from __future__ import annotations

from typing import Dict, Tuple

# KMALL canonical depth-mode ladder (normalized: manual +100 offset removed).
KMALL_DEPTH_MODE_LABELS: Dict[int, str] = {
    0: "Very Shallow",
    1: "Shallow",
    2: "Medium",
    3: "Deep",
    4: "Deeper",
    5: "Very Deep",
    6: "Extra Deep",
    7: "Extreme Deep",
}

# .all general-model depth-mode ladder (no "Deeper" step).
ALL_GENERAL_DEPTH_MODE_LABELS: Dict[int, str] = {
    0: "Very Shallow",
    1: "Shallow",
    2: "Medium",
    3: "Deep",
    4: "Very Deep",
    5: "Extra Deep",
}

# EM2040 / EM2045: the runtime mode byte's low bits select frequency.
EM2040_FREQUENCY_LABELS: Dict[int, str] = {
    0: "200kHz",
    1: "300kHz",
    2: "400kHz",
}

# EM models whose .all runtime mode byte encodes frequency, not depth band.
FREQUENCY_MODE_MODELS = frozenset({2040, 2045})

# Map the low 3 bits of the .all general mode byte onto a canonical id. Mirrors
# the bit logic in pyall's R_RUNTIME: bit0=Shallow, bit1=Medium, bit0&bit1=Deep,
# bit2=Very Deep, bit0&bit2=Extra Deep. Bit combinations 6/7 collapse onto the
# Very Deep / Extra Deep canonical ids.
_ALL_GENERAL_BITS_TO_ID: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 5}


def all_runtime_mode_info(em_model: int, mode_byte: int) -> Tuple[int, str]:
    """Decode an .all ``R`` runtime ``mode`` byte to ``(mode_id, label)``.

    Model-aware: EM2040/EM2045 return a frequency mode (0=200kHz, 1=300kHz,
    2=400kHz); all other models return a depth-mode id from the low 3 bits.
    """
    if em_model in FREQUENCY_MODE_MODELS:
        if mode_byte & 0x02:
            idx = 2
        elif mode_byte & 0x01:
            idx = 1
        else:
            idx = 0
        return idx, EM2040_FREQUENCY_LABELS[idx]

    mode_id = _ALL_GENERAL_BITS_TO_ID[mode_byte & 0x07]
    return mode_id, ALL_GENERAL_DEPTH_MODE_LABELS[mode_id]


def all_mode_label(em_model: int, mode_byte: int) -> str:
    """Human-readable label for an .all runtime mode byte (see :func:`all_runtime_mode_info`)."""
    return all_runtime_mode_info(em_model, mode_byte)[1]


def kmall_depth_mode_label(mode_id: int) -> str:
    """Label for a normalized .kmall depthMode id (manual +100 offset removed)."""
    return KMALL_DEPTH_MODE_LABELS.get(int(mode_id), f"mode{int(mode_id)}")


def kmall_raw_to_calib(raw_depth_mode: int) -> int:
    """Map a raw .kmall ``depthMode`` to the calibration-file mode number.

    Manual modes are encoded as raw 100-108 (e.g. 101 = manual Shallow); auto
    modes are the compact 0-8 ids. Both map onto the calib numbering used by
    :mod:`mbes_tools.backscatter.apply` (Shallow = 2, Medium = 3, ...)::

        101 -> 2 Shallow   104 -> 5 Deeper      107 -> 8 Extra Deep
        102 -> 3 Medium    105 -> 6 Very Deep
        103 -> 4 Deep      106 -> 7 Extra Deep
    """
    raw = int(raw_depth_mode)
    if 100 <= raw <= 108:
        return raw - 99
    return raw + 1


def mode_id_to_calib(mode_id: int) -> int:
    """Map a canonical/auto depth-mode id to the calibration-file mode number.

    The calibration files used by :mod:`mbes_tools.backscatter.apply` number
    modes from 1 (Shallow=2, Medium=3, ...). For both .kmall (auto) and .all the
    auto convention is ``calib = id + 1``; .kmall manual modes (raw >= 100) are
    normalized before reaching here.
    """
    return int(mode_id) + 1
