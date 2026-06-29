#!/usr/bin/env python3
"""Apply sector/mode/fan backscatter corrections to KMALL seabed-image samples.

Inputs
------
1. A correction CSV (e.g. exported from the balancer GUI) with schema::

       depthMode,rxFanIndex,txSectorNumb,correction_dB

   using calibration-file numbering:
   - depthMode: Shallow=2, Medium=3, Deep=4, Deeper=5, Very Deep=6, ...
   - txSectorNumb: TX sector 1, 2, 3, ... (not zero-based).

2. One KMALL file or a directory of KMALL files.

Outputs
-------
- New KMALL files with a suffix (default ``_BSCORR``).
- A QA CSV summarizing pings, beams, samples, corrections, and patch status.

Method
------
This patches only the seabed-image sample array ``SIsample_desidB`` inside #MRZ
datagrams. It does not recompute higher-level backscatter products.

Because KMALL is a structured binary format, patching is conservative:
- Each #MRZ datagram is parsed via :mod:`mbes_tools.kmall` with
  ``parse_seabed_image=True`` to recover the ``SIsample_desidB`` array and the
  per-beam ``SInumSamples``.
- Corrected sample values are built in memory.
- The original ``SIsample_desidB`` byte sequence is located inside the raw MRZ
  datagram and replaced with the corrected sequence, preserving datagram length.
- If the original byte sequence cannot be found uniquely, the datagram is left
  unchanged and the QA CSV records the reason.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mbes_tools.all import (
    iter_all_files,
    iter_datagrams,
    parse_raw_range_angle,
    parse_runtime,
    parse_seabed_image,
)
from mbes_tools.depth_modes import all_runtime_mode_info, mode_id_to_calib
from mbes_tools.kmall import MRZDatagram, iter_kmall_files, iter_mrz_datagrams


CorrectionKey = Tuple[int, int, int]  # depthMode_calib, rxFanIndex, txSectorNumb_calib
CorrectionLookup = Dict[CorrectionKey, float]


# ---------------------------------------------------------------------------
# Correction CSV loading.
# ---------------------------------------------------------------------------


def load_corrections(correction_csv: str) -> CorrectionLookup:
    """Load a compact correction CSV (depthMode,rxFanIndex,txSectorNumb,correction_dB)."""
    import pandas as pd

    df = pd.read_csv(correction_csv)
    required = {"depthMode", "rxFanIndex", "txSectorNumb", "correction_dB"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Correction CSV is missing columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}"
        )

    lookup: CorrectionLookup = {}
    for _, row in df.iterrows():
        key = (int(row["depthMode"]), int(row["rxFanIndex"]), int(row["txSectorNumb"]))
        lookup[key] = float(row["correction_dB"])
    return lookup


def depth_mode_raw_to_calib(depth_mode_raw: int) -> int:
    """Convert a KMALL raw depthMode to a calibration-file depthMode ID.

    Encoded KMALL values (manual modes)::

        101 -> 2 Shallow   104 -> 5 Deeper      107 -> 8 Extreme Deep
        102 -> 3 Medium    105 -> 6 Very Deep
        103 -> 4 Deep      106 -> 7 Extra Deep

    Compact values (auto modes) 1 -> 2 Shallow, 2 -> 3 Medium, etc.
    """
    depth_mode_raw = int(depth_mode_raw)
    if 100 <= depth_mode_raw <= 108:
        return depth_mode_raw - 99
    return depth_mode_raw + 1


# ---------------------------------------------------------------------------
# Raw datagram byte I/O and packing.
# ---------------------------------------------------------------------------


def read_raw_datagram_bytes(path: Path, byte_offset: int, n_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(byte_offset)
        return f.read(n_bytes)


def write_raw_datagram_bytes(path: Path, byte_offset: int, data: bytes) -> None:
    with open(path, "r+b") as f:
        f.seek(byte_offset)
        f.write(data)


def pack_int16_le(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<i2")
    return arr.tobytes(order="C")


def pack_float32_le(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<f4")
    return arr.tobytes(order="C")


def find_unique_payload(raw_datagram: bytes, payload: bytes) -> Optional[int]:
    """Return the unique payload position in the raw datagram, or None."""
    first = raw_datagram.find(payload)
    if first < 0:
        return None
    second = raw_datagram.find(payload, first + 1)
    if second >= 0:
        return None
    return first


# ---------------------------------------------------------------------------
# Correction math.
# ---------------------------------------------------------------------------


def build_beam_corrections(
    dgm: MRZDatagram,
    correction_lookup: CorrectionLookup,
    correction_units: str,
) -> Tuple[np.ndarray, List[dict]]:
    """Build one correction value per beam (and a QA row list) for one datagram.

    correction_units:
    - ``raw``: correction CSV is in the same units as SIsample_desidB.
    - ``dB_to_desidB``: correction CSV is in dB and SIsample_desidB is deci-dB,
      so multiply by 10.
    """
    depth_mode_raw = int(dgm.raw_depth_mode)
    depth_mode_calib = depth_mode_raw_to_calib(depth_mode_raw)
    fan = int(dgm.rx_fan_index)

    num_samples_per_beam = np.array([s.si_num_samples for s in dgm.soundings], dtype=int)
    beam_angles = np.array([s.beam_angle_re_rx_deg for s in dgm.soundings], dtype=float)
    sectors_raw = np.array([s.tx_sector_numb for s in dgm.soundings], dtype=int)
    sectors_calib = sectors_raw + 1

    beam_corrections = np.zeros(len(num_samples_per_beam), dtype=np.float32)
    qa_beams: List[dict] = []

    for beam_idx, n_samples in enumerate(num_samples_per_beam):
        sector_calib = int(sectors_calib[beam_idx])
        key = (int(depth_mode_calib), int(fan), int(sector_calib))
        correction = float(correction_lookup.get(key, 0.0))
        correction_to_apply = correction * 10.0 if correction_units == "dB_to_desidB" else correction
        beam_corrections[beam_idx] = correction_to_apply

        qa_beams.append({
            "beam_index": int(beam_idx),
            "depthMode_raw": int(depth_mode_raw),
            "depthMode_calib": int(depth_mode_calib),
            "rxFanIndex": int(fan),
            "txSectorNumb_raw": int(sectors_raw[beam_idx]),
            "txSectorNumb_calib": int(sector_calib),
            "beam_angle_deg": float(beam_angles[beam_idx]),
            "n_SI_samples": int(n_samples),
            "correction_lookup_dB": correction,
            "correction_applied_to_SI_units": float(correction_to_apply),
        })

    return beam_corrections, qa_beams


def apply_corrections_to_si_array(
    si_samples: np.ndarray,
    num_samples_per_beam: np.ndarray,
    beam_corrections: np.ndarray,
    nodata_threshold: float = -32767.0,
) -> np.ndarray:
    """Apply each beam's correction to its valid SIsample_desidB samples.

    Values <= ``nodata_threshold`` are treated as no-data and preserved, so
    Kongsberg sentinels such as -32767 are not shifted.
    """
    corrected = np.asarray(si_samples).copy()
    idx = 0

    for beam_idx, n_samples in enumerate(num_samples_per_beam.astype(int)):
        if n_samples > 0:
            sample_slice = slice(idx, idx + n_samples)
            vals = corrected[sample_slice]
            valid = vals > nodata_threshold
            vals[valid] = vals[valid] + beam_corrections[beam_idx]
            corrected[sample_slice] = vals
        idx += n_samples

    return corrected


def patch_si_payload_in_datagram(
    raw_datagram: bytes,
    original_si: np.ndarray,
    corrected_si: np.ndarray,
    patch_dtype: str,
) -> Tuple[bytes, str, Optional[int]]:
    """Patch the SIsample_desidB byte sequence inside a raw MRZ datagram.

    patch_dtype: ``int16``, ``float32``, or ``auto`` (try int16 then float32).
    """
    attempts = []

    if patch_dtype in ("auto", "int16"):
        orig_i16 = np.rint(original_si).astype(np.int16)
        corr_i16 = np.rint(corrected_si).astype(np.int16)
        attempts.append(("int16", pack_int16_le(orig_i16), pack_int16_le(corr_i16)))

    if patch_dtype in ("auto", "float32"):
        attempts.append(("float32", pack_float32_le(original_si), pack_float32_le(corrected_si)))

    for dtype_name, original_payload, corrected_payload in attempts:
        pos = find_unique_payload(raw_datagram, original_payload)
        if pos is not None:
            patched = raw_datagram[:pos] + corrected_payload + raw_datagram[pos + len(original_payload):]
            return patched, f"patched_{dtype_name}", pos

    return raw_datagram, "payload_not_found_unique", None


# ---------------------------------------------------------------------------
# Per-file processing.
# ---------------------------------------------------------------------------


def output_path_for(input_file: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{input_file.stem}{suffix}{input_file.suffix}"


def process_one_kmall(
    input_file: Path,
    output_file: Path,
    correction_lookup: CorrectionLookup,
    qa_rows: List[dict],
    patch_dtype: str,
    correction_units: str,
    dry_run: bool,
) -> None:
    """Patch one KMALL file using :mod:`mbes_tools.kmall` for parsing."""
    if input_file.resolve() == output_file.resolve():
        raise ValueError(f"Output path must differ from input path: {input_file}")

    if not dry_run:
        shutil.copy2(input_file, output_file)

    target_file = input_file if dry_run else output_file
    correction_keys = set(correction_lookup.keys())
    n_mrz = 0

    for dgm in iter_mrz_datagrams(input_file, parse_seabed_image=True):
        n_mrz += 1
        byte_offset = int(dgm.byte_offset)
        dg_size = int(dgm.num_bytes_dgm)
        raw = read_raw_datagram_bytes(input_file, byte_offset, dg_size)

        num_samples_per_beam = np.array([s.si_num_samples for s in dgm.soundings], dtype=int)
        si_samples = np.asarray(dgm.si_sample_desidb if dgm.si_sample_desidb is not None else [])

        depth_mode_raw = int(dgm.raw_depth_mode)
        depth_mode_calib = depth_mode_raw_to_calib(depth_mode_raw)
        fan = int(dgm.rx_fan_index)
        sectors_raw = np.array([s.tx_sector_numb for s in dgm.soundings], dtype=int)
        sectors_calib = sectors_raw + 1

        keys_calib = {(depth_mode_calib, fan, int(sec)) for sec in sectors_calib}
        matches_calib = keys_calib & correction_keys

        beam_corrections, _qa_beams = build_beam_corrections(
            dgm=dgm,
            correction_lookup=correction_lookup,
            correction_units=correction_units,
        )
        corrected_si = apply_corrections_to_si_array(
            si_samples=si_samples,
            num_samples_per_beam=num_samples_per_beam,
            beam_corrections=beam_corrections,
        )

        patched_raw, patch_status, payload_offset = patch_si_payload_in_datagram(
            raw_datagram=raw,
            original_si=si_samples,
            corrected_si=corrected_si,
            patch_dtype=patch_dtype,
        )

        if not dry_run and patch_status.startswith("patched"):
            write_raw_datagram_bytes(target_file, byte_offset, patched_raw)

        total_samples = int(np.sum(num_samples_per_beam))
        nonzero_corr_beams = int(np.count_nonzero(beam_corrections))
        unique_sector_raw = ";".join(str(int(x)) for x in sorted(np.unique(sectors_raw)))
        unique_sector_calib = ";".join(str(int(x)) for x in sorted(np.unique(sectors_calib)))

        qa_rows.append({
            "file": str(input_file),
            "output_file": str(output_file),
            "byte_offset": byte_offset,
            "datagram_size": dg_size,
            "total_SI_samples": total_samples,
            "n_beams": int(len(num_samples_per_beam)),
            "n_beams_with_nonzero_correction": nonzero_corr_beams,
            "mean_original_SI": float(np.nanmean(si_samples)) if si_samples.size else np.nan,
            "mean_corrected_SI": float(np.nanmean(corrected_si)) if corrected_si.size else np.nan,
            "patch_status": "dry_run_" + patch_status if dry_run else patch_status,
            "payload_offset_within_datagram": payload_offset,
            "depthMode_raw_from_kmall": depth_mode_raw,
            "depthMode_calib_used_by_script": depth_mode_calib,
            "rxFanIndex_from_kmall": fan,
            "unique_txSectorNumb_raw_from_kmall": unique_sector_raw,
            "unique_txSectorNumb_calib_used_by_script": unique_sector_calib,
            "n_matching_correction_keys": len(matches_calib),
        })

    if n_mrz == 0:
        raise RuntimeError(f"No #MRZ datagrams found in {input_file}")


def all_mode_to_calib(em_model: int, mode_byte: int) -> int:
    """Map an .all runtime ``mode`` byte to the calibration-file mode number."""
    mode_id, _label = all_runtime_mode_info(em_model, mode_byte)
    return mode_id_to_calib(mode_id)


def process_one_all(
    input_file: Path,
    output_file: Path,
    correction_lookup: CorrectionLookup,
    qa_rows: List[dict],
    patch_dtype: str,
    correction_units: str,
    dry_run: bool,
) -> None:
    """Patch one .all file's ``Y`` seabed-image samples by sector correction.

    The .all analog of :func:`process_one_kmall`. The ``Y`` datagram carries the
    per-beam seabed-image samples but no transmit sector, so each ``Y`` is paired
    with its ``N`` (raw range and angle) by ping counter to recover per-beam
    sectors; the depth/ping mode comes from the most recent ``R`` runtime
    datagram. Sample values are int16 deci-dB, like .kmall ``SIsample_desidB``.
    """
    if input_file.resolve() == output_file.resolve():
        raise ValueError(f"Output path must differ from input path: {input_file}")

    if not dry_run:
        shutil.copy2(input_file, output_file)

    target_file = input_file if dry_run else output_file
    correction_keys = set(correction_lookup.keys())
    last_mode_byte: Optional[int] = None
    last_em_model: Optional[int] = None
    pending_n: Dict[int, object] = {}
    n_y = 0

    for rec in iter_datagrams(input_file, types={"R", "N", "Y"}):
        t = rec.header.type_of_datagram
        if t == "R":
            last_mode_byte = parse_runtime(rec).mode
            last_em_model = rec.header.em_model
            continue
        if t == "N":
            nn = parse_raw_range_angle(rec)
            pending_n[nn.ping_counter] = nn
            continue

        # t == "Y"
        y = parse_seabed_image(rec)
        n_y += 1
        nn = pending_n.get(y.counter)
        byte_offset = int(rec.offset)
        dg_size = int(rec.header.number_of_bytes) + 4
        raw = read_raw_datagram_bytes(input_file, byte_offset, dg_size)

        num_samples_per_beam = np.array(
            [b.number_of_samples_per_beam for b in y.beams], dtype=int
        )
        si_samples = np.array(
            [s for b in y.beams for s in b.samples], dtype=float
        )

        skip_reason: Optional[str] = None
        if last_mode_byte is None:
            skip_reason = "no_runtime_mode"
        elif nn is None:
            skip_reason = "no_rawrange_match"

        if skip_reason is not None:
            qa_rows.append({
                "file": str(input_file),
                "output_file": str(output_file),
                "byte_offset": byte_offset,
                "datagram_size": dg_size,
                "total_SI_samples": int(np.sum(num_samples_per_beam)),
                "n_beams": int(len(num_samples_per_beam)),
                "n_beams_with_nonzero_correction": 0,
                "patch_status": ("dry_run_" if dry_run else "") + skip_reason,
                "ping_counter": int(y.counter),
            })
            continue

        depth_mode_calib = all_mode_to_calib(int(last_em_model), int(last_mode_byte))
        n_beams = min(len(y.beams), len(nn.beams))
        sectors_raw = np.array([nn.beams[i].tx_sector_number for i in range(n_beams)], dtype=int)
        sectors_calib = sectors_raw + 1

        beam_corrections = np.zeros(len(y.beams), dtype=np.float32)
        for i in range(n_beams):
            key = (int(depth_mode_calib), 0, int(sectors_calib[i]))
            correction = float(correction_lookup.get(key, 0.0))
            beam_corrections[i] = (
                correction * 10.0 if correction_units == "dB_to_desidB" else correction
            )

        corrected_si = apply_corrections_to_si_array(
            si_samples=si_samples,
            num_samples_per_beam=num_samples_per_beam,
            beam_corrections=beam_corrections,
        )
        patched_raw, patch_status, payload_offset = patch_si_payload_in_datagram(
            raw_datagram=raw,
            original_si=si_samples,
            corrected_si=corrected_si,
            patch_dtype=patch_dtype,
        )
        if not dry_run and patch_status.startswith("patched"):
            write_raw_datagram_bytes(target_file, byte_offset, patched_raw)

        keys_calib = {(int(depth_mode_calib), 0, int(s)) for s in sectors_calib}
        qa_rows.append({
            "file": str(input_file),
            "output_file": str(output_file),
            "byte_offset": byte_offset,
            "datagram_size": dg_size,
            "total_SI_samples": int(np.sum(num_samples_per_beam)),
            "n_beams": int(len(num_samples_per_beam)),
            "n_beams_with_nonzero_correction": int(np.count_nonzero(beam_corrections)),
            "mean_original_SI": float(np.nanmean(si_samples)) if si_samples.size else np.nan,
            "mean_corrected_SI": float(np.nanmean(corrected_si)) if corrected_si.size else np.nan,
            "patch_status": ("dry_run_" if dry_run else "") + patch_status,
            "payload_offset_within_datagram": payload_offset,
            "depthMode_calib_used_by_script": int(depth_mode_calib),
            "rxFanIndex_from_all": 0,
            "unique_txSectorNumb_raw_from_all": ";".join(str(int(x)) for x in sorted(np.unique(sectors_raw))),
            "unique_txSectorNumb_calib_used_by_script": ";".join(str(int(x)) for x in sorted(np.unique(sectors_calib))),
            "n_matching_correction_keys": len(keys_calib & correction_keys),
            "ping_counter": int(y.counter),
        })

    if n_y == 0:
        raise RuntimeError(f"No #Y (seabed image) datagrams found in {input_file}")


def detect_apply_format_and_files(input_path: Path, fmt: str) -> Tuple[str, List[Path]]:
    """Resolve the apply input format (``auto``|``kmall``|``all``) and files."""
    if fmt == "kmall":
        return "kmall", iter_kmall_files(input_path)
    if fmt == "all":
        return "all", iter_all_files(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in (".all", ".wcd"):
            return "all", iter_all_files(input_path)
        return "kmall", iter_kmall_files(input_path)
    kmall_files = iter_kmall_files(input_path)
    all_files = iter_all_files(input_path)
    if kmall_files and all_files:
        raise RuntimeError(
            f"Input directory contains both .kmall and .all files: {input_path}. "
            "Pass --format kmall or --format all to choose."
        )
    return ("all", all_files) if all_files else ("kmall", kmall_files)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Apply backscatter sector corrections to .kmall or .all seabed image samples."
    )
    parser.add_argument("--input", required=True, help="Input .kmall/.all file or directory.")
    parser.add_argument("--corrections", required=True, help="Correction CSV (depthMode,rxFanIndex,txSectorNumb,correction_dB).")
    parser.add_argument("--output-dir", required=True, help="Directory for corrected output files.")
    parser.add_argument(
        "--format",
        choices=["auto", "kmall", "all"],
        default="auto",
        help="Input format. 'auto' (default) detects .kmall vs .all from the input.",
    )
    parser.add_argument("--suffix", default="_BSCORR", help="Suffix for corrected files. Default: _BSCORR")
    parser.add_argument("--qa-csv", default="mbes_bs_correction_QA.csv", help="QA CSV output path.")
    parser.add_argument(
        "--patch-dtype",
        choices=["auto", "int16", "float32"],
        default="auto",
        help="Binary representation to patch. Default auto tries int16 then float32.",
    )
    parser.add_argument(
        "--correction-units",
        choices=["raw", "dB_to_desidB"],
        default="raw",
        help=(
            "Use raw if correction_dB is already in the same units as SIsample_desidB. "
            "Use dB_to_desidB to multiply correction values by 10 before patching."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write patched KMALL data. Only generate QA status.",
    )

    args = parser.parse_args()

    correction_lookup = load_corrections(args.corrections)

    input_path = Path(args.input).expanduser().resolve()
    fmt, input_files = detect_apply_format_and_files(input_path, args.format)
    if not input_files:
        raise RuntimeError(f"No .kmall or .all files found in input: {args.input}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_rows: List[dict] = []
    process = process_one_kmall if fmt == "kmall" else process_one_all

    for input_file in input_files:
        out_file = output_path_for(input_file, output_dir, args.suffix)
        print(f"Processing: {input_file.name} -> {out_file.name}")
        process(
            input_file=input_file,
            output_file=out_file,
            correction_lookup=correction_lookup,
            qa_rows=qa_rows,
            patch_dtype=args.patch_dtype,
            correction_units=args.correction_units,
            dry_run=args.dry_run,
        )

    pd.DataFrame(qa_rows).to_csv(args.qa_csv, index=False)
    print(f"QA written to: {args.qa_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
