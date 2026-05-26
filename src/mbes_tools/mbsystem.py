"""Python wrappers around MB-System CLI tools.

MB-System is a C-based external dependency commonly used for multibeam
processing. This module shells out to its binaries via ``subprocess`` and
parses their stdout into Python dataclasses. MB-System itself is not a
runtime dependency of mbes-tools — these wrappers raise a clear error
when a binary is missing from PATH, and integration tests skip cleanly.

v0 scope:
    - ``mbinfo(path, format=None)``: parse ``mbinfo -I <file>`` output.
    - ``make_datalist(files, output, format=...)``: write a .mb-1 datalist.
    - ``mbgrid(...)``: wrap mbgrid with first-class args for common params
      plus an ``extra_args`` escape hatch.
    - Format-code constants for the .kmall, .all, and GSF formats Jeff uses.

Source: written from MB-System's CLI documentation
(https://www.mbari.org/products/research-software/mb-system/).
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# MB-System format codes.
# ---------------------------------------------------------------------------

MB_FORMAT_ALL = 58       # Kongsberg .all (generic; default Jeff uses)
MB_FORMAT_GSF = 121      # Generic Sensor Format (.gsf)
MB_FORMAT_KMALL = 261    # Kongsberg .kmall


# ---------------------------------------------------------------------------
# Result dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class MBInfoResult:
    """Parsed output of ``mbinfo -I <file>``.

    Fields default to ``float("nan")`` / 0 / None when a value couldn't be
    located in the stdout (rare; mbinfo output is fairly stable). The raw
    stdout is preserved in :attr:`raw_stdout` for debugging.
    """

    file_path: Path
    format_code: int
    number_of_records: int
    longitude_min: float
    longitude_max: float
    latitude_min: float
    latitude_max: float
    depth_min: float
    depth_max: float
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    raw_stdout: str


@dataclass
class MBGridResult:
    """Result of an ``mbgrid`` run."""

    output_path: Path   # the .grd file mbgrid emits at <output_root>.grd
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.output_path.exists()


@dataclass
class Bounds:
    """W/E/S/N bounding box in decimal degrees."""

    west: float
    east: float
    south: float
    north: float


# ---------------------------------------------------------------------------
# Subprocess helpers.
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run an MB-System command, capture stdout/stderr. Does not raise.

    MB-System tools occasionally emit non-UTF-8 bytes in their progress
    output (filenames echoed verbatim from binary data, locale-encoded
    fragments). We decode with ``errors="replace"`` so a stray byte
    doesn't crash the wrapper — invalid bytes become Unicode replacement
    characters in the captured stdout/stderr.
    """
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def _require_mb_tool(name: str) -> str:
    """Locate an MB-System binary on PATH; raise with a clear message if missing."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"{name!r} not found on PATH. Install MB-System "
            "(https://www.mbari.org/products/research-software/mb-system/) "
            "or activate the conda env where it lives."
        )
    return path


# ---------------------------------------------------------------------------
# mbinfo.
# ---------------------------------------------------------------------------

# mbinfo's output is plain text. We extract the fields we care about with
# tolerant regexes — exact column widths vary across versions.
_RE_FORMAT_ID = re.compile(r"(?:MBIO Data Format ID|Format ID):\s+(\d+)")
_RE_NUM_RECORDS = re.compile(r"Number of Records:\s+(\d+)")
_RE_LON_MIN = re.compile(r"Minimum Longitude:\s+(-?\d+\.\d+)")
_RE_LON_MAX = re.compile(r"Maximum Longitude:\s+(-?\d+\.\d+)")
_RE_LAT_MIN = re.compile(r"Minimum Latitude:\s+(-?\d+\.\d+)")
_RE_LAT_MAX = re.compile(r"Maximum Latitude:\s+(-?\d+\.\d+)")
_RE_DEPTH_MIN = re.compile(r"Minimum Depth:\s+(-?\d+\.\d+)")
_RE_DEPTH_MAX = re.compile(r"Maximum Depth:\s+(-?\d+\.\d+)")
_RE_TIME_START = re.compile(r"Start of Data:\s*\n?\s*Time:\s+([^\n]+?)\s+\d")
_RE_TIME_END = re.compile(r"End of Data:\s*\n?\s*Time:\s+([^\n]+?)\s+\d")
# Alternative single-line formats some mbinfo versions emit.
_RE_TIME_START_ALT = re.compile(r"Start Time:\s+(\d+/\d+/\d+\s+\d+:\d+:\d+(?:\.\d+)?)")
_RE_TIME_END_ALT = re.compile(r"End Time:\s+(\d+/\d+/\d+\s+\d+:\d+:\d+(?:\.\d+)?)")


def _match_float(rx: re.Pattern, text: str) -> float:
    m = rx.search(text)
    return float(m.group(1)) if m else float("nan")


def _match_int(rx: re.Pattern, text: str) -> int:
    m = rx.search(text)
    return int(m.group(1)) if m else 0


def _parse_mbinfo_time(s: str) -> Optional[datetime]:
    """Try a couple of plausible mbinfo timestamp formats."""
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%m %d %Y %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _extract_time(text: str, *patterns: re.Pattern) -> Optional[datetime]:
    for rx in patterns:
        m = rx.search(text)
        if m:
            parsed = _parse_mbinfo_time(m.group(1))
            if parsed is not None:
                return parsed
    return None


def _parse_mbinfo_stdout(file_path: Path, stdout: str) -> MBInfoResult:
    """Build an :class:`MBInfoResult` from mbinfo's textual output."""
    return MBInfoResult(
        file_path=file_path,
        format_code=_match_int(_RE_FORMAT_ID, stdout),
        number_of_records=_match_int(_RE_NUM_RECORDS, stdout),
        longitude_min=_match_float(_RE_LON_MIN, stdout),
        longitude_max=_match_float(_RE_LON_MAX, stdout),
        latitude_min=_match_float(_RE_LAT_MIN, stdout),
        latitude_max=_match_float(_RE_LAT_MAX, stdout),
        depth_min=_match_float(_RE_DEPTH_MIN, stdout),
        depth_max=_match_float(_RE_DEPTH_MAX, stdout),
        time_start=_extract_time(stdout, _RE_TIME_START_ALT, _RE_TIME_START),
        time_end=_extract_time(stdout, _RE_TIME_END_ALT, _RE_TIME_END),
        raw_stdout=stdout,
    )


def mbinfo(input_path: Path, format: Optional[int] = None) -> MBInfoResult:
    """Run ``mbinfo -I <input_path>`` and return a parsed :class:`MBInfoResult`.

    Args:
        input_path: Source file or datalist.
        format: Optional MB-System format code (``-F``). When omitted,
            mbinfo will sniff the format from the file extension.

    Raises:
        RuntimeError: If the ``mbinfo`` binary is missing from PATH, or
            mbinfo returns a non-zero exit code.
    """
    bin_path = _require_mb_tool("mbinfo")
    cmd = [bin_path, "-I", str(input_path)]
    if format is not None:
        cmd += ["-F", str(format)]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"mbinfo failed for {input_path} (rc={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return _parse_mbinfo_stdout(Path(input_path), result.stdout)


# ---------------------------------------------------------------------------
# Datalist (.mb-1) generation.
# ---------------------------------------------------------------------------


def make_datalist(
    files: Sequence[Path],
    output_path: Path,
    format: int = MB_FORMAT_KMALL,
    overwrite: bool = False,
) -> Path:
    """Write a ``.mb-1`` datalist file.

    The file contains one line per input file with the format::

        /absolute/path/to/file.kmall 261

    Args:
        files: Input data files to list.
        output_path: Where to write the .mb-1 file.
        format: MB-System format code applied to every file. Default: KMALL.
        overwrite: If False (default), raise FileExistsError when the
            output already exists.

    Returns:
        The (resolved) path to the written datalist.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{Path(f).resolve()} {format}" for f in files]
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


# ---------------------------------------------------------------------------
# mbgrid.
# ---------------------------------------------------------------------------

_GRID_ALGORITHM = {
    "mean": 1,         # Gaussian-weighted mean
    "median": 2,       # median (default for bathymetry)
    "most_common": 3,
    "min": 4,
    "max": 5,
    "count": 6,
}


def mbgrid(
    input_path: Path,
    output_root: Path,
    cell_size_m: float,
    bounds: Optional[Bounds] = None,
    format: Optional[int] = None,
    grid_type: str = "median",
    extra_args: Optional[List[str]] = None,
) -> MBGridResult:
    """Run mbgrid to produce a bathymetry grid.

    mbgrid's ``-I`` flag requires a ``.mb-1`` datalist, never a single
    data file. To make calls ergonomic, this wrapper auto-wraps a single
    data file in a temporary datalist when ``input_path`` does not have
    the ``.mb-1`` extension. A single-file call requires ``format=`` so
    we know what to write in the temp datalist.

    Args:
        input_path: A ``.mb-1`` datalist, OR a single data file (which we
            auto-wrap in a temp datalist).
        output_root: Output prefix; mbgrid appends ``.grd`` (and produces
            companion ``.grd.cmd``, plot script, etc. in the same dir).
        cell_size_m: Grid cell size in meters (passed as ``-E m/m``).
        bounds: Optional W/E/S/N bounding box. If omitted and ``input_path``
            is a single data file, the wrapper auto-derives bounds by
            running ``mbinfo`` first. For datalist inputs without bounds,
            mbgrid must derive its own (which has been unreliable on some
            MB-System versions — supply explicit bounds if you hit a
            "Grid bounds not properly specified" error).
        format: MB-System format code (e.g. ``MB_FORMAT_KMALL``).
            Required when ``input_path`` is a single data file. Ignored
            when ``input_path`` is already a datalist (the datalist
            carries format per file).
        grid_type: Algorithm tag mapping to mbgrid's ``-A``:

            ====================  ====
            value                 ``-A``
            ====================  ====
            ``"mean"``            1 (Gaussian-weighted mean)
            ``"median"``          2 (default)
            ``"most_common"``     3
            ``"min"``             4
            ``"max"``             5
            ``"count"``           6
            ====================  ====

        extra_args: Pass-through args appended to the command, for the
            long tail of mbgrid flags this wrapper doesn't model.

    Returns:
        An :class:`MBGridResult` with the expected output ``.grd`` path,
        the process return code, and captured stdout / stderr. Inspect
        ``.succeeded`` for the quick yes/no.

    Raises:
        ValueError: If ``grid_type`` is unknown, or if a single data file
            is passed without a ``format=``.
    """
    bin_path = _require_mb_tool("mbgrid")
    if grid_type not in _GRID_ALGORITHM:
        raise ValueError(
            f"grid_type must be one of {sorted(_GRID_ALGORITHM)}; got {grid_type!r}"
        )

    input_path = Path(input_path)
    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Auto-wrap single data files in a temp datalist. mbgrid -I requires
    # a datalist; passing a binary data file there causes the underlying
    # mbgrid to try to parse it as text, which historically has manifested
    # as a glibc buffer overflow during mb_read_init.
    tmp_datalist_dir: Optional[Path] = None
    is_single_file = input_path.suffix.lower() != ".mb-1"
    if not is_single_file:
        datalist_path = input_path
    else:
        if format is None:
            raise ValueError(
                "format= is required when input_path is a single data file "
                "(needed to write the temp datalist). Pass MB_FORMAT_KMALL, "
                "MB_FORMAT_ALL, etc., or build a .mb-1 datalist with "
                "make_datalist() and pass that instead."
            )
        tmp_datalist_dir = Path(tempfile.mkdtemp(prefix="mbes_tools_mbgrid_"))
        datalist_path = tmp_datalist_dir / "auto.mb-1"
        make_datalist([input_path], datalist_path, format=format, overwrite=True)

    # Auto-derive bounds via mbinfo when caller didn't supply them, but
    # only for single-file inputs (datalist auto-bounds would need to
    # mbinfo each entry and union — a v0.1 feature).
    if bounds is None and is_single_file:
        info = mbinfo(input_path, format=format)
        if all(
            math.isfinite(v)
            for v in (info.longitude_min, info.longitude_max,
                      info.latitude_min, info.latitude_max)
        ):
            bounds = Bounds(
                west=info.longitude_min,
                east=info.longitude_max,
                south=info.latitude_min,
                north=info.latitude_max,
            )

    cmd = [
        bin_path,
        "-I", str(datalist_path),
        "-O", str(output_root),
        # mbgrid -E default unit is degrees; pass the "m" suffix so
        # cell_size_m is interpreted as meters.
        "-E", f"{cell_size_m}/{cell_size_m}/m",
        "-A", str(_GRID_ALGORITHM[grid_type]),
    ]
    if bounds is not None:
        cmd += ["-R", f"{bounds.west}/{bounds.east}/{bounds.south}/{bounds.north}"]
    if extra_args:
        cmd += list(extra_args)

    try:
        result = _run(cmd)
    finally:
        if tmp_datalist_dir is not None:
            shutil.rmtree(tmp_datalist_dir, ignore_errors=True)

    return MBGridResult(
        output_path=Path(f"{output_root}.grd"),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
