# Changelog

All notable changes to `mbes-tools` are documented here. The project follows
[semantic versioning](https://semver.org/) (currently 0.x — minor versions may
add features; the public API is kept backward compatible where practical).

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
