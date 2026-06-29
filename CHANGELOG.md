# Changelog

All notable changes to `mbes-tools` are documented here. The project follows
[semantic versioning](https://semver.org/) (currently 0.x — minor versions may
add features; the public API is kept backward compatible where practical).

## [0.3.0] - 2026-06-29

### Added (Capability A — .all backscatter parity)
- `mbes_tools.all`: parsers for **datagram 78 `N`** (raw range and angle —
  per-beam pointing angle + transmit sector + reflectivity) and **datagram 82
  `R`** (runtime — depth/ping mode). `XBeam.is_valid` flags real detections.
  `iter_datagrams(types=...)` can skip bodies of unwanted datagram types.
  Byte layouts verified against the pyall reference on real EM2040 data.
- `mbes_tools.depth_modes`: one documented table mapping each (model, format)
  depth/ping-mode encoding onto a canonical id + label, including the
  EM2040/EM2045 case where the .all runtime mode byte selects **frequency**
  (200/300/400 kHz) rather than a depth band.
- `mbes_tools.backscatter.all_table`: `process_all_ping` / `accumulate_all_file`
  join per-ping `X`+`N`+`R`+`P` into the existing `SoundingRecord`, reusing
  `aggregate_records` / ping QC / `build_rows` / `normalize` unchanged, so
  EM2040 `.all` and EM124 `.kmall` share one backscatter pipeline.
- `.all` write-back path in `mbes_tools.backscatter.apply` (`process_one_all`)
  patches `Y` seabed-image samples by sector correction (pairs `Y`↔`N` by ping
  counter for sectors, depth mode from `R`).
- `mbes-bs-table` and `mbes-bs-apply` now **dispatch by file extension**
  (`--format auto|kmall|all`); `--reflectivity-source xyz88|rawrange78` for `.all`.
- Committed fixtures: `sample_equinox_em2040.all` (EM2040, Samoa AUV-matched) and
  `sample_tn447_em124.kmall` (EM124, Samoa ship-matched). `docs/DEPTH_MODES.md`.

### Changed
- Ping-level QC extracted to `backscatter.table.apply_ping_qc`, shared by the
  .kmall and .all front-ends (behavior for .kmall unchanged).

## [0.2.0] - 2026-06-28

### Added
- `mbes_tools.catalog`: a path-agnostic tool to inventory Kongsberg
  `.all`/`.wcd`/`.kmall`/`.kmwcd` files into a verification manifest (CSV/JSON).
  Records path, format, MB-System format id, EM model, vessel (heuristic),
  latitude/longitude, UTM zone, depth regime, datagram-type census,
  seabed-image presence, and per-file errors. Envelope-only scanning keeps it
  cheap on large files; `--per-dir-limit` samples representative files; gzip/zip
  containers are detected and reported instead of crashing. Console script
  `mbes-catalog`.
- `docs/VERIFICATION_DATA.md` and `docs/verification_manifest.csv`/`.json`:
  the real verification corpus catalog for the upgrade plan (§2), covering
  EM124, EM2040, EM304, EM302, EM122, EM710 across both hemispheres and all four
  Kongsberg file formats.

## [0.1.0]

### Added
- Initial scaffolding lifted from prior project work: `kmall`, `all`, `wcd`,
  `kmwcd`, `imb` readers; `mbsystem` MB-System wrappers; `backscatter`
  sector/angle normalization (`table`, `normalize`, `apply`) with console
  scripts `mbes-bs-table`, `mbes-bs-apply`, `mbes-bs-gui`.
