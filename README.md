# mbes-tools

Python tools for reading and processing Kongsberg multibeam echosounder data, with MB-System integration helpers.

## Status

v0, pre-release. Modules are being scaffolded and lifted from prior project work.

## Modules

- `mbes_tools.kmall` — Kongsberg .kmall reader: native #MRZ binary parser (skips unsupported #FCF/#SPE; optionally parses the seabed-image `SIsample_desidB` array), plus `#SKM` attitude and `#IIP`/`#IOP` installation/runtime parsers
- `mbes_tools.all` — Kongsberg .all reader (datagrams `P` position, `X` XYZ depth, `Y` seabed image, `N` raw range/angle, `R` runtime, `A` attitude, `I` installation); cross-referenced against Mike's pyall / pyAllConditioner
- `mbes_tools.install_params` — structures Kongsberg install/runtime text (transducer lever arms, mount angles, waterline, EM model, serial) across the `.all` flat (`S{n}`) and `.kmall #IIP` nested (`TRAI_TX1`/`TRAI_RX1`) schemes
- `mbes_tools.depth_modes` — documented depth/ping-mode maps across EM models and formats (incl. EM2040 `.all` frequency modes vs `.kmall` depth modes)
- `mbes_tools.beam_stat` — pluggable numpy reducers (mean/median/std/mode/trimmed-mean/percentile/min/max/range/count) that turn a beam's seabed-image samples into one backscatter value over a configurable sample window (backscatter **Source B**)
- `mbes_tools.projection` — geography-agnostic target-CRS resolution: configurable EPSG or auto UTM-zone-from-position (hemisphere/antimeridian/polar-safe); no baked-in zone
- `mbes_tools.diagnostics` — visual review suite (console script `mbes-diagnostics`): renders ARC/normalization, Source A-vs-B, multi-stat texture, `--si-window`, depth-mode split, port/starboard symmetry, swath-fan geometry, auto-UTM map, MBES_ARC parity, and corruption-recovery plots from real files (matplotlib lazy-imported)
- `mbes_tools.wc_diagnostics` — water-column visual review suite (console script `mbes-wc-diagnostics`): renders amplitude echograms with the detected-bottom overlay, geo-referenced swath wedges, nadir amplitude profiles, bottom-detect/amplitude-peak alignment, `phase_flag` 1/2 phase echograms + histograms, and a sector-frequency sanity panel from real `.kmwcd`/`.wcd` files (matplotlib lazy-imported)

The readers stream datagram-by-datagram and accept `on_error="skip"` to resynchronize past corrupt/truncated datagrams; `mbes-bs-table`/`-apply` default to `--on-error skip` so one bad ping or file never aborts a survey.
- `mbes_tools.wcd` — water column data (`k` datagram in .wcd, paired with .all); validated against real EM122 data
- `mbes_tools.kmwcd` — water column data (`#MWC` datagram in .kmwcd, paired with .kmall), including `phase_flag` 1 (int8) and 2 (int16) per-beam phase samples; validated against real EM124 and EM2040 data
- `mbes_tools.mbsystem` — Python wrappers around MB-System CLI tools (mbinfo, mbgrid, datalist generation, format codes)
- `mbes_tools.catalog` — inventory Kongsberg `.all`/`.kmall` files into a verification manifest (path, format, EM model, vessel, geography, depth regime, datagram types, seabed-image presence); console script `mbes-catalog`. See [docs/VERIFICATION_DATA.md](docs/VERIFICATION_DATA.md).
- `mbes_tools.backscatter` — sector/angle backscatter normalization for **both .kmall and .all** (one pipeline):
  - `table` — generate a sector/angle correction table; dispatches by file extension (`--format auto|kmall|all`). .kmall reads #MRZ; `all_table` joins per-ping `X`+`N`+`R`+`P` (and `Y` for Source B). Backscatter source is selectable: `--bs-source reflectivity` (Source A, per-beam reflectivity) or `--bs-source seabed_image` (Source B) with `--beam-stat` (one or more reducers) and `--si-window`. Geometry mask, slope/SD grids, ping QC, and flat-seafloor filter live in `qc`.
  - `normalize` — Lambertian sector balancing (headless compute)
  - `apply` — patch corrections into the .kmall #MRZ seabed-image samples or .all `Y` seabed-image samples (dispatches by extension)

### Backscatter console scripts

Installed with the package (see the `backscatter` / `gui` extras for dependencies):

- `mbes-bs-table` — generate the sector/angle correction table
- `mbes-bs-gui` — interactive Lambertian balancer GUI
- `mbes-bs-apply` — apply corrections to .kmall files

```bash
pip install -e '.[backscatter,gui]'
```

## Install

From a conda environment:

```bash
git clone https://github.com/Jeff-Beeson/mbes-tools.git
cd mbes-tools
conda env create -f environment.yml
conda activate mbes-tools
pip install -e .
```

## Development

Editable install in a project's environment:

```bash
conda activate <project-env>
pip install -e ~/code/mbes-tools
```

Run tests:

```bash
pytest
```

## License

MIT
