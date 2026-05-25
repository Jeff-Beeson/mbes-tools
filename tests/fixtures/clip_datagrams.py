#!/usr/bin/env python3
"""Clip the first N datagrams of a Kongsberg .all or .kmall file.

Used to produce small, committable test fixtures from a full survey file
without taking on a large binary blob in the repo. Walks the source file
by its internal datagram-length fields and stops cleanly at a datagram
boundary, so the resulting clip is itself a valid (truncated) Kongsberg
file.

Examples
--------

    # Clip a .kmall to the first 2 #MRZ datagrams plus everything in between
    python tests/fixtures/clip_datagrams.py \\
        /mnt/d/Cowork_OS/0014_20251112_063307_DavidPackard.kmall \\
        tests/fixtures/sample_dpdk027.kmall \\
        --mrz 2

    # Clip a .all to the first 3 X (depth) datagrams plus everything in between
    python tests/fixtures/clip_datagrams.py \\
        /mnt/d/Cowork_OS/0000_20250507_021406_Nautilus.all \\
        tests/fixtures/sample_nautilus.all \\
        --pings 3

Format is auto-detected from the file extension (``.all`` or ``.kmall``);
override with ``--format``.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def clip_all(src: Path, dst: Path, max_pings: int) -> tuple[int, int]:
    """Copy datagrams from src to dst until we've seen ``max_pings`` X-type pings.

    .all envelope: leading uint32 ``numberOfBytes`` (excludes itself),
    so total bytes on disk for a datagram = numberOfBytes + 4.
    """
    pings = 0
    total_datagrams = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            head = fin.read(16)
            if len(head) < 16:
                break
            nbytes, _stx, type_code, _model, _date, _time = struct.unpack("<LBBHLL", head)
            total = nbytes + 4
            # Rewind to start of datagram and copy it whole.
            fin.seek(-16, 1)
            dgm = fin.read(total)
            if len(dgm) < total:
                break
            fout.write(dgm)
            total_datagrams += 1
            if chr(type_code) == "X":
                pings += 1
                if pings >= max_pings:
                    break
    return pings, total_datagrams


def clip_kmall(src: Path, dst: Path, max_mrz: int) -> tuple[int, int]:
    """Copy datagrams from src to dst until we've seen ``max_mrz`` #MRZ datagrams.

    .kmall envelope: leading uint32 ``datagram_size`` (includes itself),
    plus 4-byte ASCII type code like b'#MRZ'.
    """
    mrz = 0
    total_datagrams = 0
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
            total_datagrams += 1
            if type_bytes == b"#MRZ":
                mrz += 1
                if mrz >= max_mrz:
                    break
    return mrz, total_datagrams


def main() -> None:
    p = argparse.ArgumentParser(
        description="Clip the first N datagrams of a Kongsberg .all or .kmall file."
    )
    p.add_argument("input", type=Path, help="Source .all or .kmall file")
    p.add_argument("output", type=Path, help="Output clip file")
    p.add_argument(
        "--format",
        choices=["all", "kmall"],
        default=None,
        help="Format. Defaults to the input file extension.",
    )
    p.add_argument(
        "--pings",
        type=int,
        default=3,
        help="For .all: stop after this many X (depth) datagrams. Default: 3.",
    )
    p.add_argument(
        "--mrz",
        type=int,
        default=2,
        help="For .kmall: stop after this many #MRZ datagrams. Default: 2.",
    )
    args = p.parse_args()

    fmt = args.format or args.input.suffix.lstrip(".").lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "all":
        target, total = clip_all(args.input, args.output, max_pings=args.pings)
        print(
            f"Wrote {args.output} ({args.output.stat().st_size:,} bytes); "
            f"{total} datagrams total, {target} X (depth) pings."
        )
    elif fmt == "kmall":
        target, total = clip_kmall(args.input, args.output, max_mrz=args.mrz)
        print(
            f"Wrote {args.output} ({args.output.stat().st_size:,} bytes); "
            f"{total} datagrams total, {target} #MRZ datagrams."
        )
    else:
        raise SystemExit(f"Unknown format: {fmt!r}. Use --format all or --format kmall.")


if __name__ == "__main__":
    main()
