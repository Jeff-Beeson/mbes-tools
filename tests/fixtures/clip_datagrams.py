#!/usr/bin/env python3
"""Clip the first N target-type datagrams of a Kongsberg file.

Supports .all, .wcd, .kmall, and .kmwcd files. Used to produce small,
committable test fixtures from a full survey file without taking on a
large binary blob in the repo. Walks the source file by its internal
datagram-length fields and stops cleanly at a datagram boundary, so the
resulting clip is itself a valid (truncated) Kongsberg file.

Format is auto-detected from the file extension; override with
``--format``. The target datagram type defaults based on format:

    .all  -> X (depth)
    .wcd  -> k (water column)
    .kmall -> #MRZ
    .kmwcd -> #MWC

Override with ``--target-type``.

Examples
--------

    # .kmall: 2 #MRZ + everything in between
    python tests/fixtures/clip_datagrams.py \\
        /mnt/d/Cowork_OS/full.kmall tests/fixtures/sample.kmall -n 2

    # .all: 3 X-pings + everything in between
    python tests/fixtures/clip_datagrams.py \\
        /mnt/d/Cowork_OS/full.all tests/fixtures/sample.all -n 3

    # .all: stop after 5 Y (seabed image) datagrams instead of X
    python tests/fixtures/clip_datagrams.py \\
        /mnt/d/Cowork_OS/full.all tests/fixtures/y_only.all \\
        --target-type Y -n 5
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

ALL_FAMILY = {"all", "wcd"}
KMALL_FAMILY = {"kmall", "kmwcd"}

DEFAULT_TARGET = {
    "all": "X",
    "wcd": "k",
    "kmall": "#MRZ",
    "kmwcd": "#MWC",
}

DEFAULT_COUNT = {
    "all": 3,
    "wcd": 3,
    "kmall": 2,
    "kmwcd": 2,
}


def clip_all_family(src: Path, dst: Path, target_type: str, max_count: int) -> tuple[int, int]:
    """Walk a .all-format file (16-byte envelope, single-char type code).

    Stop after copying ``max_count`` datagrams whose type matches ``target_type``.
    """
    if len(target_type) != 1:
        raise SystemExit(
            f"--target-type for .all/.wcd must be a single character, got {target_type!r}"
        )
    target_code = ord(target_type)
    found = 0
    total = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            head = fin.read(16)
            if len(head) < 16:
                break
            nbytes, _stx, type_code, _model, _date, _time = struct.unpack("<LBBHLL", head)
            total_size = nbytes + 4
            fin.seek(-16, 1)
            dgm = fin.read(total_size)
            if len(dgm) < total_size:
                break
            fout.write(dgm)
            total += 1
            if type_code == target_code:
                found += 1
                if found >= max_count:
                    break
    return found, total


def clip_kmall_family(src: Path, dst: Path, target_type: str, max_count: int) -> tuple[int, int]:
    """Walk a .kmall-format file (8-byte size+type envelope, 4-char type).

    Stop after copying ``max_count`` datagrams whose type matches ``target_type``
    (e.g. ``#MRZ``, ``#MWC``).
    """
    target_bytes = target_type.encode("ascii")
    if len(target_bytes) != 4:
        raise SystemExit(
            f"--target-type for .kmall/.kmwcd must be 4 ASCII chars, got {target_type!r}"
        )
    found = 0
    total = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            head = fin.read(8)
            if len(head) < 8:
                break
            size, type_bytes = struct.unpack("<I4s", head)
            fin.seek(-8, 1)
            dgm = fin.read(size)
            if len(dgm) < size:
                break
            fout.write(dgm)
            total += 1
            if type_bytes == target_bytes:
                found += 1
                if found >= max_count:
                    break
    return found, total


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clip the first N target-type datagrams of a Kongsberg file."
    )
    p.add_argument("input", type=Path, help="Source .all/.wcd/.kmall/.kmwcd file")
    p.add_argument("output", type=Path, help="Output clip file")
    p.add_argument(
        "--format",
        choices=sorted(ALL_FAMILY | KMALL_FAMILY),
        default=None,
        help="File format. Defaults to the input file extension.",
    )
    p.add_argument(
        "--target-type",
        default=None,
        help="Datagram type code to stop at. Defaults: X (.all), k (.wcd), #MRZ (.kmall), #MWC (.kmwcd).",
    )
    p.add_argument(
        "-n",
        type=int,
        default=None,
        help="Stop after this many target datagrams. Defaults: 3 (.all/.wcd), 2 (.kmall/.kmwcd).",
    )
    args = p.parse_args()

    fmt = (args.format or args.input.suffix.lstrip(".").lower())
    if fmt not in ALL_FAMILY and fmt not in KMALL_FAMILY:
        raise SystemExit(f"Unknown format: {fmt!r}. Use --format to override.")

    target_type = args.target_type or DEFAULT_TARGET[fmt]
    count = args.n if args.n is not None else DEFAULT_COUNT[fmt]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ALL_FAMILY:
        found, total = clip_all_family(args.input, args.output, target_type, count)
    else:
        found, total = clip_kmall_family(args.input, args.output, target_type, count)

    print(
        f"Wrote {args.output} ({args.output.stat().st_size:,} bytes); "
        f"{total} datagrams total, {found} '{target_type}' datagrams."
    )


if __name__ == "__main__":
    main()
