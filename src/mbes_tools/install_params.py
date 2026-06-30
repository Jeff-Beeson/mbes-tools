"""Parse Kongsberg installation / runtime parameter text into structured values.

Both the ``.all`` ``I`` installation datagram (73) and the ``.kmall`` ``#IIP``
installation / ``#IOP`` runtime datagrams carry their payload as a delimited
ASCII parameter string — but in two different schemes:

* **.all (flat):** comma-separated ``KEY=VALUE`` tokens, e.g.
  ``WLZ=-2.070,SMH=110,S1X=3.495,S1Y=-0.139,S1Z=2.730,S1R=0.686,S1P=-0.048,S1H=0.088,...``
  Transducer geometry is keyed by system number: ``S{n}X/Y/Z`` (lever arm) and
  ``S{n}R/P/H`` (mount angles); waterline is ``WLZ``.
* **.kmall #IIP (nested):** comma-separated *sections* ``NAME:sub=val;sub=val``,
  e.g. ``TRAI_TX1:N=0;X=4.221;Y=0.914;Z=6.225;R=0.060;P=-0.070;H=0.120;S=1.0`` for
  the TX transducer (``TRAI_RX1`` for RX), ``ATTI_1:...`` for attitude sensors,
  ``EMXI:SWLZ=0.740`` for the waterline, plus flat ``EMXV=EM124``, ``SN=10055``.

This module exposes both views (flat ``params`` and nested ``sections``) and the
survey-critical pieces — transducer **lever arms** (X/Y/Z), **mount angles**
(R/P/H), the **waterline**, and the sensor **serial / EM model** — so downstream
code can re-georeference and refine ARC without re-scraping raw text. Pure stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# A flat token is KEY, then '=' or ':', then everything up to the next
# ','/';'/newline. Tolerates free-form text by ignoring non-matching tokens.
_KV_RE = re.compile(r"([A-Za-z0-9_]+)\s*[:=]\s*([^,;\n\r]*)")


def parse_install_text(text: str) -> Dict[str, str]:
    """Extract flat ``KEY=VALUE`` / ``KEY:VALUE`` pairs from a parameter string.

    Best-effort: tokens without a separator are ignored, values are stripped,
    later duplicates win. Good for ``.all`` install text and ``.kmall``
    top-level scalars (``EMXV``, ``SN``); for nested ``.kmall`` transducer
    geometry use :func:`parse_install_sections`.
    """
    out: Dict[str, str] = {}
    for key, val in _KV_RE.findall(text):
        out[key] = val.strip()
    return out


def parse_install_sections(text: str) -> Dict[str, Dict[str, str]]:
    """Parse ``.kmall`` ``#IIP`` nested sections into ``{name: {sub: value}}``.

    Each comma-separated entry of the form ``NAME:sub=val;sub=val`` becomes one
    section. Flat ``.all`` text (no ``NAME:...;...`` structure) yields ``{}``.
    """
    sections: Dict[str, Dict[str, str]] = {}
    for entry in text.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        name, _, rest = entry.partition(":")
        sub: Dict[str, str] = {}
        for kv in rest.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                sub[k.strip()] = v.strip()
        if sub:
            sections[name.strip()] = sub
    return sections


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


@dataclass
class InstallationParameters:
    """Structured view of an installation / runtime parameter string.

    ``raw`` is the original text. ``params`` is the flat ``KEY=VALUE`` view;
    ``sections`` is the nested ``.kmall`` view (``{}`` for ``.all``). The
    convenience fields and accessors cover the survey-critical install geometry,
    working across both the ``.all`` (``S{n}`` group) and ``.kmall``
    (``TRAI_TX1``/``TRAI_RX1`` section) conventions; absent values are ``None``.
    """

    raw: str
    params: Dict[str, str] = field(default_factory=dict)
    sections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    em_model: Optional[str] = None
    serial_number: Optional[str] = None
    waterline_m: Optional[float] = None

    @classmethod
    def from_text(cls, text: str) -> "InstallationParameters":
        params = parse_install_text(text)
        sections = parse_install_sections(text)
        # Waterline: .all WLZ (flat) or .kmall EMXI:SWLZ (section).
        waterline = _to_float(params.get("WLZ"))
        if waterline is None:
            waterline = _to_float(sections.get("EMXI", {}).get("SWLZ"))
        return cls(
            raw=text,
            params=params,
            sections=sections,
            em_model=params.get("EMXV"),
            serial_number=params.get("SN"),
            waterline_m=waterline,
        )

    def _group_values(self, group: str, axes: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
        """Read three axis values for a transducer/sensor group.

        ``group`` is a ``.kmall`` section name (e.g. ``"TRAI_TX1"``) or a ``.all``
        flat prefix (e.g. ``"S1"``). Returns ``None`` if any axis is missing.
        """
        if group in self.sections:
            src = self.sections[group]
            vals = [_to_float(src.get(ax)) for ax in axes]
        else:
            vals = [_to_float(self.params.get(f"{group}{ax}")) for ax in axes]
        if any(v is None for v in vals):
            return None
        return (vals[0], vals[1], vals[2])  # type: ignore[return-value]

    def transducer_offsets(self, group: str = "S1") -> Optional[Tuple[float, float, float]]:
        """Lever arm ``(x_forward_m, y_starboard_m, z_down_m)`` for a transducer.

        ``group`` = ``.all`` system prefix (``"S1"``/``"S2"``) or ``.kmall``
        section (``"TRAI_TX1"``/``"TRAI_RX1"``).
        """
        return self._group_values(group, ("X", "Y", "Z"))

    def mount_angles(self, group: str = "S1") -> Optional[Tuple[float, float, float]]:
        """Mount angles ``(roll_deg, pitch_deg, heading_deg)`` for a transducer.

        ``group`` = ``.all`` system prefix (``"S1"``/``"S2"``) or ``.kmall``
        section (``"TRAI_TX1"``/``"TRAI_RX1"``).
        """
        return self._group_values(group, ("R", "P", "H"))
