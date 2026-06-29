"""Geography-agnostic target-CRS resolution for projecting soundings to metres.

No survey-specific UTM zone is baked in anywhere in the library. A caller either
names an explicit target CRS (``EPSG:32602``, an EPSG integer, or a PROJ/WKT
string) or asks for ``auto`` and supplies a representative longitude/latitude;
``auto`` then picks the appropriate UTM zone (or polar UPS) for that position.

This is pure arithmetic — no pyproj dependency — so it is importable and testable
in the base environment. The actual coordinate transform (which does need pyproj)
lives in :class:`mbes_tools.backscatter.qc.SpatialProjector`, which calls
:func:`resolve_target_crs` to turn ``auto`` into a concrete EPSG.

Safe across hemispheres, the antimeridian (longitude is wrapped to [-180, 180)),
and high latitudes (polar regions fall back to UPS rather than an invalid zone).
"""

from __future__ import annotations

import math
from typing import Optional, Union

# Latitude limits of UTM validity; beyond these, UTM zones are not defined and
# the polar stereographic (UPS) systems are used instead.
_UTM_NORTH_LIMIT = 84.0
_UTM_SOUTH_LIMIT = -80.0
EPSG_UPS_NORTH = 5041
EPSG_UPS_SOUTH = 5042


def utm_zone_number(lon_deg: float) -> int:
    """UTM zone number (1-60) for a longitude, antimeridian-safe."""
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    zone = int((lon + 180.0) / 6.0) + 1
    return min(max(zone, 1), 60)


def utm_epsg_from_lonlat(lon_deg: float, lat_deg: float) -> int:
    """Return the EPSG code of the UTM (or polar UPS) zone for a position.

    Northern zones are 326xx, southern 327xx. Latitudes beyond the UTM band use
    UPS North (5041) / UPS South (5042).
    """
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        raise ValueError(f"non-finite position: lon={lon_deg}, lat={lat_deg}")
    if lat_deg > _UTM_NORTH_LIMIT:
        return EPSG_UPS_NORTH
    if lat_deg < _UTM_SOUTH_LIMIT:
        return EPSG_UPS_SOUTH
    zone = utm_zone_number(lon_deg)
    return (32600 if lat_deg >= 0 else 32700) + zone


def resolve_target_crs(
    spec: Optional[Union[str, int]],
    lon_deg: Optional[float] = None,
    lat_deg: Optional[float] = None,
) -> str:
    """Resolve a CRS spec to an ``EPSG:NNNNN`` / PROJ string.

    ``spec`` may be:
    - ``None`` or ``"auto"`` — derive the UTM/UPS zone from ``lon_deg``/``lat_deg``
      (both required in this case);
    - an int or digit string — treated as an EPSG code;
    - anything else (``"EPSG:32756"``, a PROJ4/WKT string) — returned unchanged.
    """
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "auto"):
        if lon_deg is None or lat_deg is None:
            raise ValueError(
                "target CRS 'auto' requires a representative lon/lat to pick a UTM zone"
            )
        return f"EPSG:{utm_epsg_from_lonlat(lon_deg, lat_deg)}"
    if isinstance(spec, int):
        return f"EPSG:{spec}"
    s = spec.strip()
    if s.isdigit():
        return f"EPSG:{s}"
    return s
