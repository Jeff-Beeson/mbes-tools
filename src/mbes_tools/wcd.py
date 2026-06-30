"""Kongsberg .wcd water column data reader.

The .wcd format is the water-column companion to .all. It uses the SAME
datagram envelope as .all (older Kongsberg binary format), so this module
composes on top of :mod:`mbes_tools.all` for envelope walking and only
adds the ``k`` Water Column datagram parser.

Source: written from the public Kongsberg EM Series Data Format spec
(October 2013). Binary structures cross-referenced against Jeff's
existing WCD-aware fork of pyall in
``G:\\My Drive\\[02] Work\\[10] Python\\Water_Column_Decode\\pyAllConditioner-master\\pyall.py``.

Status: validated against real data (Capability D1) on the Atlantis EM122
file ``0197_20130703_155651_Atlantis.wcd`` — all 1407 ``k`` datagrams
reconcile exactly (predicted body == actual, modulo the single Kongsberg
spare byte before the footer, i.e. no drift). EM122 deep-water: 8 sectors
near 12 kHz, 288 beams, monotonic beam angles; most pings are fragmented
across several ``k`` datagrams (``num_datagram`` > 1), which the caller
reassembles by ``counter``. Committed clip:
``tests/fixtures/sample_atlantis_em122.wcd`` (3 complete pings).

Typical usage::

    from pathlib import Path
    from mbes_tools.wcd import iter_water_column_datagrams

    for dgm in iter_water_column_datagrams(Path("survey.wcd")):
        for beam in dgm.beams:
            ...  # beam.samples is a tuple of int8 amplitude samples (0.5 dB units)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

from mbes_tools.all import DatagramHeader, DatagramRecord, iter_datagrams


# ---------------------------------------------------------------------------
# Datagram type code.
# ---------------------------------------------------------------------------

DGM_TYPE_WATER_COLUMN = b"k"


# ---------------------------------------------------------------------------
# Binary structure definitions for the ``k`` Water Column datagram body.
# (All sizes are body-relative; the .all envelope is handled upstream.)
#
# Layout, in order:
#   - K body (28 bytes): counter, serialNumber, numDatagram, datagramNum,
#     numTxSectors, numBeams (total across this ping), numBeamsThisDatagram,
#     soundSpeedAtTransducer*10 (uint16), sampleFreq (uint32, 0.01 Hz),
#     txHeave*100 (int16), tvgFunction (uint8), tvgOffset (int8),
#     spare (uint32, unused).
#   - K_TX_SECTOR (6 bytes), repeated numTxSectors times.
#   - For each of numBeamsThisDatagram beams:
#       - K_BEAM header (10 bytes)
#       - numSamples int8 amplitude samples (0.5 dB resolution per spec)
#   - Trailing footer (handled by upstream envelope iterator).
# ---------------------------------------------------------------------------

K_BODY_FMT = "<8HLhBbL"
K_BODY_SIZE = struct.calcsize(K_BODY_FMT)

K_TX_SECTOR_FMT = "<hHBB"
K_TX_SECTOR_SIZE = struct.calcsize(K_TX_SECTOR_FMT)

K_BEAM_FMT = "<hHHHBB"
K_BEAM_SIZE = struct.calcsize(K_BEAM_FMT)


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass
class WCTxSector:
    """One transmit sector record in a ``k`` Water Column datagram."""

    tilt_angle_deg: float
    centre_frequency_hz: float
    tx_sector_number: int
    scanning_info: int


@dataclass
class WCBeam:
    """One receive beam in a ``k`` Water Column datagram.

    ``samples`` is the per-beam int8 amplitude array, units 0.5 dB per spec.
    Multiply by 0.5 to convert to dB if downstream code wants real-valued
    amplitudes.
    """

    pointing_angle_deg: float
    start_range_sample: int
    number_of_samples: int
    detected_range_samples: int  # detection range in samples (the "rangeDR" field)
    tx_sector_number: int
    beam_number: int
    samples: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class WaterColumnDatagram:
    """Parsed ``k`` Water Column datagram.

    Large pings may be split across multiple K datagrams. ``num_datagram``
    is the total fragment count for the parent ping; ``datagram_num`` is the
    1-based index of this fragment. Reassembly is the caller's job — collect
    all datagrams sharing the same ``counter`` and concatenate beams.
    """

    header: DatagramHeader
    counter: int
    serial_number: int
    num_datagram: int        # total fragments for parent ping
    datagram_num: int        # 1-based index of this fragment
    num_tx_sectors: int
    num_beams_ping: int      # total receive beams across all fragments
    num_beams_this_datagram: int
    sound_speed_m_s: float
    sample_frequency_hz: float
    tx_heave_m: float
    tvg_function: int
    tvg_offset_db: int
    tx_sectors: List[WCTxSector] = field(default_factory=list)
    beams: List[WCBeam] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------


def parse_water_column(record: DatagramRecord) -> WaterColumnDatagram:
    """Decode the body of a ``k`` Water Column datagram.

    Expects ``record`` to come from :func:`mbes_tools.all.iter_datagrams`
    or equivalent, i.e. ``record.body`` is the bytes between the .all
    envelope and the trailing footer.
    """
    if record.header.type_of_datagram != "k":
        raise ValueError(
            f"parse_water_column expects type 'k', got {record.header.type_of_datagram!r}"
        )

    s = struct.unpack_from(K_BODY_FMT, record.body, 0)
    (
        counter,
        serial,
        num_datagram,
        datagram_num,
        num_tx,
        num_beams_total,
        num_beams_this,
        sound_speed_raw,
        sample_freq_raw,
        tx_heave_raw,
        tvg_fn,
        tvg_off,
        _spare,
    ) = s

    cursor = K_BODY_SIZE

    # TX sectors.
    tx_sectors: List[WCTxSector] = []
    for _ in range(num_tx):
        t = struct.unpack_from(K_TX_SECTOR_FMT, record.body, cursor)
        tilt_raw, cf_raw, sec_num, scan = t
        tx_sectors.append(
            WCTxSector(
                tilt_angle_deg=tilt_raw / 100.0,
                centre_frequency_hz=cf_raw * 10.0,
                tx_sector_number=int(sec_num),
                scanning_info=int(scan),
            )
        )
        cursor += K_TX_SECTOR_SIZE

    # RX beams (each followed by N int8 amplitude samples).
    beams: List[WCBeam] = []
    for _ in range(num_beams_this):
        b = struct.unpack_from(K_BEAM_FMT, record.body, cursor)
        angle_raw, start_samp, n_samps, dr, sector, beam_num = b
        cursor += K_BEAM_SIZE

        if n_samps > 0:
            samples = struct.unpack_from(f"<{n_samps}b", record.body, cursor)
            cursor += n_samps
        else:
            samples = ()

        beams.append(
            WCBeam(
                pointing_angle_deg=angle_raw / 100.0,
                start_range_sample=int(start_samp),
                number_of_samples=int(n_samps),
                detected_range_samples=int(dr),
                tx_sector_number=int(sector),
                beam_number=int(beam_num),
                samples=samples,
            )
        )

    return WaterColumnDatagram(
        header=record.header,
        counter=counter,
        serial_number=serial,
        num_datagram=num_datagram,
        datagram_num=datagram_num,
        num_tx_sectors=num_tx,
        num_beams_ping=num_beams_total,
        num_beams_this_datagram=num_beams_this,
        sound_speed_m_s=sound_speed_raw / 10.0,
        sample_frequency_hz=sample_freq_raw / 100.0,
        tx_heave_m=tx_heave_raw / 100.0,
        tvg_function=int(tvg_fn),
        tvg_offset_db=int(tvg_off),
        tx_sectors=tx_sectors,
        beams=beams,
    )


# ---------------------------------------------------------------------------
# File discovery + convenience iterator.
# ---------------------------------------------------------------------------


def iter_wcd_files(path: Path, recursive: bool = False) -> List[Path]:
    """Return ``.wcd`` files from a file or directory."""
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = path.rglob("*.wcd") if recursive else path.glob("*.wcd")
    return sorted(set(files))


def iter_water_column_datagrams(path: Path) -> Iterator[WaterColumnDatagram]:
    """Walk a .wcd file and yield each parsed ``k`` Water Column datagram.

    Non-'k' datagrams (P position, A attitude, etc.) are walked past
    silently. Use :func:`mbes_tools.all.iter_datagrams` if you need the
    full datagram stream from a .wcd file.
    """
    for rec in iter_datagrams(path):
        if rec.header.type_of_datagram == "k":
            yield parse_water_column(rec)
