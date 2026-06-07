"""Kongsberg M3 .imb / .002 packet reader.

Binary layout (IMB.cs field definitions, byte offsets validated against
a working prototype):

  Offset from packet start    Size   Content
  --------------------------  -----  -------
   0–7                             8  Sync word (0x00 0x80 × 4)
   8–131                         124  Fixed header (version, ID, timing,
                                       geometry, beam count)
   132                           var  BeamList — n_beams × float32 angles
   132 + n_beams*4              4340  Extended header — nav/attitude fields
                                       (UNVALIDATED: layout from IMB.cs only)
   132 + n_beams*4 + 4340       var  fPacketBody — n_beams × n_samples × float32
                                       (footer 44 bytes at end of each packet)

The extended header between BeamList and fPacketBody has NOT been confirmed
against real .imb data.  ``iter_imb_packets()`` skips it by default
(``extended=False``).  Pass ``extended=True`` once you have files to validate
the layout; if the offsets prove wrong the intensity array will be garbage.

Typical usage::

    from pathlib import Path
    from mbes_tools.imb import iter_imb_packets

    for pkt in iter_imb_packets(Path("survey.002")):
        print(pkt.timestamp, pkt.intensity.shape)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC_WORD = b"\x00\x80\x00\x80\x00\x80\x00\x80"
FOOTER_SIZE = 44
MIN_PACKET_BYTES = 5000  # heuristic: skip spurious sync matches / status records

# ---------------------------------------------------------------------------
# Fixed header (bytes 8–131 from packet start, i.e. right after sync word)
#
# Fields (in order):
#   version, sonarID,
#   sonarInfo × 20,   ← IMB.cs declares UInt32[8] but empirical offsets
#                       require 20 entries (80 bytes) to land nNumSamples at
#                       byte 108 and nNumBeams at byte 128.
#   timeSec, timeMsec, velocitySound,
#   nNumSamples, nearRange, farRange, SWST, SWL,
#   nNumBeams, processingType
# ---------------------------------------------------------------------------

_FIXED_FMT = "<2I20I2IfIffffHH"
_FIXED_SIZE = struct.calcsize(_FIXED_FMT)  # 124 bytes → BeamList at byte 132
assert _FIXED_SIZE == 124, f"Fixed header size is {_FIXED_SIZE}, expected 124"

BEAM_LIST_OFFSET = len(SYNC_WORD) + _FIXED_SIZE  # 132 — confirmed by prototype

# ---------------------------------------------------------------------------
# Extended header (4340 bytes, follows BeamList, precedes fPacketBody)
# All fields from IMB.cs in declaration order.  UNVALIDATED against real data.
# ---------------------------------------------------------------------------

_EXT_FMT = (
    "<"
    "fHHIiiHHHH"    # imageSampleInterval, imgDest, reserved2, modeID,
                     # numHybridPRI, hybridIndex, phaseSeqLen, phaseSeqIdx,
                     # numImages, subImageIdx                             → v[0..9]
    "IIIfff"         # sonarFreq, pulseLength, pingNumber,
                     # rxFilterBW, rxNomRes, pulseRepFreq                → v[10..15]
    "128s64s"        # strAppName, strTXPulseName64                      → v[16..17]
    "HHff"           # TVG: spreadingCoeff, absorptionCoeff, curve, max  → v[18..21]
    "ffffffffffff"   # heading, magVar, pitch, roll, depth, temp,
                     # xOff, yOff, zOff, xRotOff, yRotOff, zRotOff      → v[22..33]
    "I"              # mounting                                          → v[34]
    "ddf"            # latitude, longitude, txWST                        → v[35..37]
    "BBBB"           # headSensorVer, headHWStatus, reserved1, reserved2 → v[38..41]
    "fff"            # internalHeading, internalPitch, internalRoll      → v[42..44]
    "12f"            # sAxesRotatorOffsets[3] × {a, b, r, angle}        → v[45..56]
    "HH"             # startElement, endElement                          → v[57..58]
    "32s32s"         # customText1, customText2                          → v[59..60]
    "ffffff"         # localTimeOffset, vesselSOG, heave,
                     # priMinRange, priMaxRange, soundSpeedRT            → v[61..66]
    "3856s"          # Reserved                                          → v[67]
)
_EXT_SIZE = struct.calcsize(_EXT_FMT)  # 4340 bytes
assert _EXT_SIZE == 4340, f"Extended header size is {_EXT_SIZE}, expected 4340"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ImbPacket:
    """One decoded M3 IMB ping.

    ``beam_angles_deg`` and ``intensity`` are always populated.  The
    extended-header fields (``ping_number`` through ``sound_speed_rt_ms``)
    are ``None`` unless ``iter_imb_packets`` was called with
    ``extended=True`` and the layout proved correct for the file.
    """

    # Timing
    time_sec: int       # seconds since Unix epoch
    time_msec: int      # millisecond part

    # Sonar identity
    version: int
    sonar_id: int
    processing_type: int  # 0=standard 1=EIQ 2=profiling 3=hybrid 4=interferometry …

    # Geometry
    num_beams: int
    num_samples: int
    near_range_m: float
    far_range_m: float
    swst_s: float            # sampling window start time (s)
    swl_s: float             # sampling window length (s)
    velocity_sound_ms: float

    # Data arrays
    beam_angles_deg: np.ndarray  # shape (num_beams,), float32
    intensity: np.ndarray        # shape (num_beams, num_samples), float32

    # Extended header — None unless extended=True was passed and layout confirmed
    ping_number: Optional[int] = None
    sonar_freq_hz: Optional[int] = None
    pulse_length_us: Optional[int] = None
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    depth_m: Optional[float] = None
    temperature_c: Optional[float] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    heave_m: Optional[float] = None
    vessel_sog_kn: Optional[float] = None
    sound_speed_rt_ms: Optional[float] = None

    @property
    def timestamp(self) -> datetime:
        """UTC datetime for this ping."""
        return datetime.fromtimestamp(
            self.time_sec + self.time_msec / 1000.0, tz=timezone.utc
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_sync_offsets(data: bytes) -> List[int]:
    offsets: List[int] = []
    pos = 0
    while (pos := data.find(SYNC_WORD, pos)) != -1:
        offsets.append(pos)
        pos += len(SYNC_WORD)
    return offsets


def _unpack_extended(buf: bytes) -> dict:
    """Parse nav/attitude fields from the extended-header buffer.

    Returns an empty dict if *buf* is too short.
    """
    if len(buf) < _EXT_SIZE:
        return {}
    v = struct.unpack(_EXT_FMT, buf[:_EXT_SIZE])
    return {
        "ping_number":       v[12],   # dwPingNumber
        "sonar_freq_hz":     v[10],   # dwSonarFreq
        "pulse_length_us":   v[11],   # dwPulseLength
        "heading_deg":       v[22],   # fCompassHeading
        "pitch_deg":         v[24],   # fPitch
        "roll_deg":          v[25],   # fRoll
        "depth_m":           v[26],   # fDepth
        "temperature_c":     v[27],   # Temperature
        "latitude_deg":      v[35],   # dbLatitude
        "longitude_deg":     v[36],   # dbLongitude
        "heave_m":           v[63],   # fHeave
        "vessel_sog_kn":     v[62],   # fVesselSOG
        "sound_speed_rt_ms": v[66],   # fSoundSpeed_RT
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iter_imb_packets(
    path: Path,
    *,
    min_packet_bytes: int = MIN_PACKET_BYTES,
    extended: bool = False,
) -> Iterator[ImbPacket]:
    """Iterate over IMB packets in an M3 .imb / .002 file.

    Args:
        path: Path to the .imb or .002 file.
        min_packet_bytes: Skip packets shorter than this threshold.  The
            default (5000) rejects spurious sync-word matches and short
            status-only records observed in prototype testing.
        extended: Parse the extended header between BeamList and fPacketBody
            to populate nav/attitude fields (lat, lon, heading, pitch, roll,
            depth, etc.).  Defaults to ``False`` because the layout has not
            been validated against real .imb data.  Enable once confirmed.

    Yields:
        :class:`ImbPacket` for each successfully decoded ping.
    """
    data = Path(path).read_bytes()
    offsets = _find_sync_offsets(data)
    if not offsets:
        return

    for i, offset in enumerate(offsets):
        # Measure packet length from this sync to the next.
        # The last sync word has no successor, so skip it (unknown length).
        packet_length = offsets[i + 1] - offset if i + 1 < len(offsets) else None
        if packet_length is not None and packet_length < min_packet_bytes:
            continue

        # --- Fixed header ---
        hdr_start = offset + len(SYNC_WORD)
        if hdr_start + _FIXED_SIZE > len(data):
            continue
        hv = struct.unpack_from(_FIXED_FMT, data, hdr_start)

        version     = hv[0]
        sonar_id    = hv[1]
        # hv[2..21]: dwSonarInfo × 20 (not exposed)
        time_sec    = hv[22]
        time_msec   = hv[23]
        vel_sound   = hv[24]
        num_samples = hv[25]
        near_range  = hv[26]
        far_range   = hv[27]
        swst        = hv[28]
        swl         = hv[29]
        num_beams   = hv[30]
        proc_type   = hv[31]

        if num_beams == 0 or num_samples == 0:
            continue

        # --- BeamList ---
        beam_start = offset + BEAM_LIST_OFFSET
        beam_bytes = num_beams * 4
        if beam_start + beam_bytes > len(data):
            continue
        beam_angles = np.frombuffer(data, dtype="<f4", count=num_beams, offset=beam_start)

        body_start = beam_start + beam_bytes

        # --- Extended header (nav/attitude) ---
        ext_fields: dict = {}
        if extended:
            ext_fields = _unpack_extended(data[body_start: body_start + _EXT_SIZE])
            body_start += _EXT_SIZE

        # --- fPacketBody ---
        body_count = num_beams * num_samples
        if body_start + body_count * 4 > len(data):
            continue
        body = np.frombuffer(data, dtype="<f4", count=body_count, offset=body_start)
        intensity = body.reshape((num_beams, num_samples))

        yield ImbPacket(
            time_sec=time_sec,
            time_msec=time_msec,
            version=version,
            sonar_id=sonar_id,
            processing_type=proc_type,
            num_beams=num_beams,
            num_samples=num_samples,
            near_range_m=near_range,
            far_range_m=far_range,
            swst_s=swst,
            swl_s=swl,
            velocity_sound_ms=vel_sound,
            beam_angles_deg=beam_angles.copy(),
            intensity=intensity.copy(),
            **ext_fields,
        )
