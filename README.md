# mbes-tools

Python tools for reading and processing Kongsberg multibeam echosounder data, with MB-System integration helpers.

## Status

v0, pre-release. Modules are being scaffolded and lifted from prior project work.

## Modules

- `mbes_tools.kmall` — Kongsberg .kmall reader, native #MRZ binary parser (skips unsupported #FCF and #SPE datagrams; optionally parses the seabed-image `SIsample_desidB` array)
- `mbes_tools.all` — Kongsberg .all reader (datagrams `P` position, `X` XYZ depth, `Y` seabed image, `N` raw range/angle, `R` runtime); cross-referenced against Mike's pyall / pyAllConditioner
- `mbes_tools.depth_modes` — documented depth/ping-mode maps across EM models and formats (incl. EM2040 `.all` frequency modes vs `.kmall` depth modes)
- `mbes_tools.beam_stat` — pluggable numpy reducers (mean/median/std/mode/trimmed-mean/percentile/min/max/range/count) that turn a beam's seabed-image samples into one backscatter value over a configurable sample window (backscatter **Source B**)
- `mbes_tools.wcd` — water column data (.wcd, paired with .all)
- `mbes_tools.kmwcd` — water column data (.kmwcd, paired with .kmall)
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
