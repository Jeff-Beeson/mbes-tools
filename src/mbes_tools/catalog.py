"""Catalog Kongsberg multibeam files into a verification manifest.

Given one or more root directories, this walks for Kongsberg raw files
(``.all``, ``.wcd``, ``.kmall``, ``.kmwcd``) and records, per file, the
metadata needed to plan and verify a processing pipeline against real data:

    path, format, em_model, vessel, latitude, longitude, utm_zone,
    depth_regime, median_depth_m, datagram_types, has_seabed_image,
    n_datagrams_scanned, scan_truncated, error

It is deliberately *path-agnostic*: callers pass the roots to scan, so no
survey-specific location is baked in. Use it to build a small manifest that
documents which models / vessels / geographies a capability was verified on.

Design notes
------------
- Scanning is **envelope-only** for the datagram-type census: the file is
  walked by each datagram's declared length without reading bodies, so even
  large files are cheap. A few specific datagrams (first position, first
  depth/sounding, first seabed-image, installation) are decoded for model,
  geography, and depth regime.
- ``--per-dir-limit`` samples N files per directory so a survey of hundreds
  of near-identical lines yields a *small* representative manifest rather than
  hundreds of rows. Set 0 for no limit.
- ``--max-scan-bytes`` caps the type census on very large files; the row notes
  ``scan_truncated`` when the cap is hit.
- One bad file never aborts the run: per-file errors are captured in the
  ``error`` column.

CLI::

    python -m mbes_tools.catalog ~/RR1413 ~/MGL1701 /mnt/d -o manifest.csv
    python -m mbes_tools.catalog ROOT --per-dir-limit 2 --json manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Recognized Kongsberg extensions and their format family / MB-System format id.
ALL_FAMILY = {"all", "wcd"}
KMALL_FAMILY = {"kmall", "kmwcd"}

# MB-System MBIO format ids for the recognized extensions.
MBSYSTEM_FORMAT_ID = {
    "all": 56,     # MBF_EM710RAW  (Kongsberg current-generation .all)
    "wcd": 56,
    "kmall": 261,  # MBF_KEMKMALL  (Kongsberg .kmall)
    "kmwcd": 261,
}

# .all envelope: numberOfBytes(u32), STX(u8), type(u8), emModel(u16), date(u32), time(u32).
_ALL_HEADER_FMT = "<LBBHLL"
_ALL_HEADER_SIZE = struct.calcsize(_ALL_HEADER_FMT)

# .kmall envelope: numBytesDgm(u32), dgmType(4s).
_KMALL_HEADER_FMT = "<I4s"
_KMALL_HEADER_SIZE = struct.calcsize(_KMALL_HEADER_FMT)

# Filename trailing-token -> canonical vessel name. Heuristic; extend as needed.
_VESSEL_ALIASES = {
    "langseth": "R/V Marcus G. Langseth",
    "revelle": "R/V Roger Revelle",
    "nautilus": "E/V Nautilus",
    "davidpackard": "R/V David Packard",
    "thompson": "R/V Thomas G. Thompson",
    "atlantis": "R/V Atlantis",
    "fugroequinox": "Fugro Equinox",
    "equinox": "Fugro Equinox",
    "fugrobrasilis": "Fugro Brasilis",
    "searcher": "Searcher",
}

_EM_MODEL_RE = re.compile(rb"EM\s?(\d{3,4})")


@dataclass
class CatalogRow:
    path: str
    format: str
    mbsystem_format_id: int
    em_model: str = ""
    vessel: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    utm_zone: str = ""
    depth_regime: str = ""
    median_depth_m: Optional[float] = None
    datagram_types: str = ""
    has_seabed_image: Optional[bool] = None
    n_datagrams_scanned: int = 0
    scan_truncated: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers shared across formats.
# ---------------------------------------------------------------------------


def utm_zone_from_lonlat(lon_deg: float, lat_deg: float) -> str:
    """Return a UTM zone label like ``32N`` / ``2S`` from a lon/lat in degrees.

    Hemisphere-safe and antimeridian-safe (longitude is wrapped to [-180,180)).
    """
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        return ""
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    hemi = "N" if lat_deg >= 0 else "S"
    return f"{zone}{hemi}"


def classify_depth_regime(depth_m: Optional[float]) -> str:
    """Coarse depth-regime label used to plan coverage diversity."""
    if depth_m is None or not math.isfinite(depth_m):
        return ""
    d = abs(depth_m)
    if d < 100:
        return "shallow"
    if d < 1000:
        return "shelf/slope"
    if d < 3000:
        return "deep"
    return "abyssal"


def vessel_from_filename(path: Path) -> str:
    """Heuristic vessel name from the filename's trailing token."""
    stem = path.stem
    # Kongsberg convention: leading numeric/date fields, then a vessel token,
    # e.g. 0118_20210605_010253_Langseth -> "Langseth".
    token = stem.split("_")[-1]
    key = re.sub(r"[^a-z0-9]", "", token.lower())
    if key in _VESSEL_ALIASES:
        return _VESSEL_ALIASES[key]
    # Some names embed the model, e.g. FKt230303_EM124; fall back to whole stem scan.
    for alias_key, name in _VESSEL_ALIASES.items():
        if alias_key in stem.lower():
            return name
    # Don't mistake a trailing model code (e.g. "EM124") or a numeric field for a vessel.
    if not token or token.isdigit() or _EM_MODEL_RE.fullmatch(token.encode("ascii", "ignore")):
        return ""
    return token


def _format_of(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def _compression_note(path: Path) -> str:
    """Return a note if the file is a compressed container, else ''.

    Some archived surveys keep a ``.all``/``.kmall`` extension on a file that is
    actually gzip- or zip-compressed; give an actionable message instead of a
    misleading "bad datagram length".
    """
    try:
        with path.open("rb") as fid:
            magic = fid.read(4)
    except OSError as exc:
        return f"unreadable: {exc}"
    if magic[:2] == b"\x1f\x8b":
        return "gzip-compressed (decompress before reading: gunzip)"
    if magic[:2] == b"PK":
        return "zip-compressed (extract before reading)"
    return ""


# ---------------------------------------------------------------------------
# .all-family scanning.
# ---------------------------------------------------------------------------


def scan_all_file(path: Path, max_scan_bytes: int) -> CatalogRow:
    fmt = _format_of(path)
    row = CatalogRow(
        path=str(path),
        format=fmt,
        mbsystem_format_id=MBSYSTEM_FORMAT_ID.get(fmt, 0),
        vessel=vessel_from_filename(path),
    )

    note = _compression_note(path)
    if note:
        row.error = note
        return row

    from mbes_tools.all import (
        parse_depth,
        parse_position,
        DatagramHeader,
        DatagramRecord,
    )

    type_counts: Dict[str, int] = {}
    em_models: set = set()
    file_size = path.stat().st_size
    depths: List[float] = []

    try:
        with path.open("rb") as fid:
            offset = 0
            while offset < file_size:
                if max_scan_bytes and offset >= max_scan_bytes:
                    row.scan_truncated = True
                    break
                fid.seek(offset)
                head = fid.read(_ALL_HEADER_SIZE)
                if len(head) < _ALL_HEADER_SIZE:
                    break
                nbytes, _stx, type_code, em_model, _date, _time = struct.unpack(
                    _ALL_HEADER_FMT, head
                )
                total_size = nbytes + 4
                if nbytes < _ALL_HEADER_SIZE - 4 or offset + total_size > file_size:
                    row.error = f"bad datagram length {nbytes} at offset {offset}"
                    break
                type_char = chr(type_code)
                type_counts[type_char] = type_counts.get(type_char, 0) + 1
                em_models.add(em_model)
                row.n_datagrams_scanned += 1

                # Decode the first position and first few depth datagrams for
                # geography and depth regime.
                if type_char in ("P", "X") and (
                    (type_char == "P" and row.latitude is None)
                    or (type_char == "X" and len(depths) < 5)
                ):
                    body_size = total_size - _ALL_HEADER_SIZE - 3  # footer = ETX+chksum
                    fid.seek(offset)
                    fid.read(_ALL_HEADER_SIZE)
                    body = fid.read(max(body_size, 0))
                    hdr = DatagramHeader(
                        number_of_bytes=nbytes,
                        stx=_stx,
                        type_of_datagram=type_char,
                        em_model=em_model,
                        record_date=_date,
                        record_time_ms=_time,
                    )
                    rec = DatagramRecord(header=hdr, offset=offset, body=body)
                    try:
                        if type_char == "P" and row.latitude is None:
                            p = parse_position(rec)
                            if -90 <= p.latitude_deg <= 90 and -180 <= p.longitude_deg <= 180:
                                row.latitude = round(p.latitude_deg, 6)
                                row.longitude = round(p.longitude_deg, 6)
                        elif type_char == "X":
                            d = parse_depth(rec)
                            valid = [
                                b.depth_m
                                for b in d.beams
                                if math.isfinite(b.depth_m) and abs(b.depth_m) > 0.1
                            ]
                            if valid:
                                depths.append(sorted(valid)[len(valid) // 2])
                    except Exception:
                        pass

                offset += total_size
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run
        row.error = f"{type(exc).__name__}: {exc}"

    row.em_model = ";".join(_em_label(m) for m in sorted(em_models) if m)
    row.datagram_types = ";".join(f"{t}:{type_counts[t]}" for t in sorted(type_counts))
    row.has_seabed_image = type_counts.get("Y", 0) > 0
    if depths:
        med = sorted(depths)[len(depths) // 2]
        row.median_depth_m = round(med, 1)
        row.depth_regime = classify_depth_regime(med)
    if row.latitude is not None and row.longitude is not None:
        row.utm_zone = utm_zone_from_lonlat(row.longitude, row.latitude)
    return row


def _em_label(model: int) -> str:
    """Render an .all envelope emModel integer as e.g. ``EM2040``."""
    return f"EM{model}" if model else ""


# ---------------------------------------------------------------------------
# .kmall-family scanning.
# ---------------------------------------------------------------------------


def scan_kmall_file(path: Path, max_scan_bytes: int) -> CatalogRow:
    fmt = _format_of(path)
    row = CatalogRow(
        path=str(path),
        format=fmt,
        mbsystem_format_id=MBSYSTEM_FORMAT_ID.get(fmt, 0),
        vessel=vessel_from_filename(path),
    )

    note = _compression_note(path)
    if note:
        row.error = note
        return row

    type_counts: Dict[str, int] = {}
    file_size = path.stat().st_size
    em_model_from_iip = ""
    first_mrz_offset: Optional[int] = None
    first_mrz_size: Optional[int] = None
    first_iip_offset: Optional[int] = None
    first_iip_size: Optional[int] = None

    try:
        with path.open("rb") as fid:
            offset = 0
            while offset < file_size:
                if max_scan_bytes and offset >= max_scan_bytes:
                    row.scan_truncated = True
                    break
                fid.seek(offset)
                head = fid.read(_KMALL_HEADER_SIZE)
                if len(head) < _KMALL_HEADER_SIZE:
                    break
                size, dgm_type = struct.unpack(_KMALL_HEADER_FMT, head)
                if size < 8 or offset + size > file_size:
                    row.error = f"bad datagram length {size} at offset {offset}"
                    break
                try:
                    type_str = dgm_type.decode("ascii")
                except UnicodeDecodeError:
                    type_str = repr(dgm_type)
                type_counts[type_str] = type_counts.get(type_str, 0) + 1
                row.n_datagrams_scanned += 1

                if type_str == "#IIP" and first_iip_offset is None:
                    first_iip_offset = offset
                    first_iip_size = size

                if type_str == "#MRZ" and first_mrz_offset is None:
                    first_mrz_offset = offset
                    first_mrz_size = size

                offset += size
    except Exception as exc:  # noqa: BLE001
        row.error = f"{type(exc).__name__}: {exc}"

    # Decode the first #MRZ for geography, depth regime, and seabed-image flag.
    if first_mrz_offset is not None and first_mrz_size is not None:
        try:
            from mbes_tools.kmall import parse_mrz_datagram

            with path.open("rb") as fid:
                dgm = parse_mrz_datagram(
                    fid,
                    datagram_start=first_mrz_offset,
                    datagram_size=first_mrz_size,
                    parse_seabed_image=True,
                )
            if dgm is not None:
                if (
                    dgm.latitude_deg is not None
                    and dgm.longitude_deg is not None
                    and -90 <= dgm.latitude_deg <= 90
                    and -180 <= dgm.longitude_deg <= 180
                ):
                    row.latitude = round(dgm.latitude_deg, 6)
                    row.longitude = round(dgm.longitude_deg, 6)
                depths = [
                    s.z_m
                    for s in dgm.soundings
                    if s.is_valid and math.isfinite(s.z_m) and abs(s.z_m) > 0.1
                ]
                if depths:
                    med = sorted(depths)[len(depths) // 2]
                    row.median_depth_m = round(med, 1)
                    row.depth_regime = classify_depth_regime(med)
                row.has_seabed_image = bool(dgm.si_sample_desidb)
        except Exception as exc:  # noqa: BLE001
            if not row.error:
                row.error = f"mrz decode: {type(exc).__name__}: {exc}"

    # EM model from the first #IIP: the structured EMXV key via install_params
    # (supersedes the old raw-bytes regex scrape; also reads the full-length text
    # instead of capping at 4096 bytes), with a raw-text regex fallback so any
    # file the old scrape matched still resolves.
    if first_iip_offset is not None and first_iip_size is not None:
        try:
            from mbes_tools.kmall import parse_kmall_params_datagram

            with path.open("rb") as fid:
                params_dgm = parse_kmall_params_datagram(
                    fid,
                    datagram_start=first_iip_offset,
                    datagram_size=first_iip_size,
                )
            if params_dgm is not None:
                em_model_from_iip = params_dgm.parameters.em_model or ""
                if not em_model_from_iip:
                    m = _EM_MODEL_RE.search(
                        params_dgm.parameters.raw.encode("ascii", "ignore")
                    )
                    if m:
                        em_model_from_iip = f"EM{m.group(1).decode('ascii')}"
        except Exception as exc:  # noqa: BLE001
            if not row.error:
                row.error = f"iip decode: {type(exc).__name__}: {exc}"

    # Model: prefer #IIP text, fall back to a filename hint (e.g. *_EM124.kmall).
    row.em_model = em_model_from_iip
    if not row.em_model:
        m = _EM_MODEL_RE.search(path.name.encode("ascii", "ignore"))
        if m:
            row.em_model = f"EM{m.group(1).decode('ascii')}"

    row.datagram_types = ";".join(f"{t}:{type_counts[t]}" for t in sorted(type_counts))
    if row.has_seabed_image is None:
        row.has_seabed_image = False
    if row.latitude is not None and row.longitude is not None:
        row.utm_zone = utm_zone_from_lonlat(row.longitude, row.latitude)
    return row


# ---------------------------------------------------------------------------
# Discovery and top-level catalog.
# ---------------------------------------------------------------------------


def discover_files(roots: List[Path]) -> List[Path]:
    """Return sorted Kongsberg files under the given roots (recursive)."""
    exts = ALL_FAMILY | KMALL_FAMILY
    found: List[Path] = []
    for root in roots:
        root = root.expanduser()
        if root.is_file():
            if _format_of(root) in exts:
                found.append(root)
            continue
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and _format_of(p) in exts:
                found.append(p)
    return sorted(set(found))


def sample_per_directory(files: List[Path], per_dir_limit: int) -> List[Path]:
    """Keep at most ``per_dir_limit`` files per (parent dir, extension).

    Sampling is per directory *and* per extension so a folder that holds both
    .kmall and .kmwcd keeps an example of each. ``per_dir_limit <= 0`` keeps all.
    """
    if per_dir_limit <= 0:
        return files
    seen: Dict[tuple, int] = {}
    kept: List[Path] = []
    for p in files:
        key = (str(p.parent), _format_of(p))
        n = seen.get(key, 0)
        if n < per_dir_limit:
            kept.append(p)
            seen[key] = n + 1
    return kept


def scan_file(path: Path, max_scan_bytes: int) -> CatalogRow:
    fmt = _format_of(path)
    if fmt in ALL_FAMILY:
        return scan_all_file(path, max_scan_bytes)
    return scan_kmall_file(path, max_scan_bytes)


def build_catalog(
    roots: List[Path],
    per_dir_limit: int = 3,
    max_scan_bytes: int = 64 * 1024 * 1024,
    progress=None,
) -> List[CatalogRow]:
    """Discover, sample, and scan files under ``roots`` into catalog rows."""
    files = sample_per_directory(discover_files(roots), per_dir_limit)
    rows: List[CatalogRow] = []
    for i, path in enumerate(files, 1):
        if progress is not None:
            progress(i, len(files), path)
        rows.append(scan_file(path, max_scan_bytes))
    return rows


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "path",
    "format",
    "mbsystem_format_id",
    "em_model",
    "vessel",
    "latitude",
    "longitude",
    "utm_zone",
    "depth_regime",
    "median_depth_m",
    "datagram_types",
    "has_seabed_image",
    "n_datagrams_scanned",
    "scan_truncated",
    "error",
]


def write_csv(rows: List[CatalogRow], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def write_json(rows: List[CatalogRow], out_path: Path) -> None:
    out_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Catalog Kongsberg .all/.kmall files into a verification manifest."
    )
    parser.add_argument("roots", nargs="+", help="Root directories (or files) to scan.")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path.")
    parser.add_argument("--json", default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--per-dir-limit",
        type=int,
        default=3,
        help="Max files to sample per directory+extension (0 = all). Default 3.",
    )
    parser.add_argument(
        "--max-scan-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Cap the datagram-type census per file in bytes (0 = whole file). Default 64MiB.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file progress.")
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.roots]

    def progress(i, n, path):
        if not args.quiet:
            print(f"[{i}/{n}] {path}", flush=True)

    rows = build_catalog(
        roots,
        per_dir_limit=args.per_dir_limit,
        max_scan_bytes=args.max_scan_bytes,
        progress=progress,
    )

    if args.output:
        write_csv(rows, Path(args.output).expanduser())
        print(f"Wrote {len(rows)} rows -> {args.output}")
    if args.json:
        write_json(rows, Path(args.json).expanduser())
        print(f"Wrote {len(rows)} rows -> {args.json}")
    if not args.output and not args.json:
        # Default: print a compact summary to stdout.
        for r in rows:
            print(
                f"{r.format:5s} {r.em_model:18s} {r.vessel:24s} "
                f"{r.depth_regime:11s} SI={r.has_seabed_image} {r.path}"
            )


if __name__ == "__main__":
    main()
