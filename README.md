# mbes-tools

Python tools for reading and processing Kongsberg multibeam echosounder data, with MB-System integration helpers.

## Status

v0, pre-release. Modules are being scaffolded and lifted from prior project work.

## Modules

- `mbes_tools.kmall` — Kongsberg .kmall reader, native #MRZ binary parser (skips unsupported #FCF and #SPE datagrams; optionally parses the seabed-image `SIsample_desidB` array)
- `mbes_tools.all` — Kongsberg .all reader, builds on Mike's pyall / pyAllConditioner
- `mbes_tools.wcd` — water column data (.wcd, paired with .all)
- `mbes_tools.kmwcd` — water column data (.kmwcd, paired with .kmall)
- `mbes_tools.mbsystem` — Python wrappers around MB-System CLI tools (mbinfo, mbgrid, datalist generation, format codes)
- `mbes_tools.catalog` — inventory Kongsberg `.all`/`.kmall` files into a verification manifest (path, format, EM model, vessel, geography, depth regime, datagram types, seabed-image presence); console script `mbes-catalog`. See [docs/VERIFICATION_DATA.md](docs/VERIFICATION_DATA.md).
- `mbes_tools.backscatter` — sector/angle backscatter normalization built on the kmall reader:
  - `table` — generate a sector/angle correction table from .kmall files (geometry mask, slope/SD grids, ping QC, flat-seafloor filter live in `qc`)
  - `normalize` — Lambertian sector balancing (headless compute)
  - `apply` — patch corrections into the KMALL seabed-image samples

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
