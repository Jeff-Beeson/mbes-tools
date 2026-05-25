# mbes-tools

Python tools for reading and processing Kongsberg multibeam echosounder data, with MB-System integration helpers.

## Status

v0, pre-release. Modules are being scaffolded and lifted from prior project work.

## Modules

- `mbes_tools.kmall` — Kongsberg .kmall reader, native #MRZ binary parser (skips unsupported #FCF and #SPE datagrams)
- `mbes_tools.all` — Kongsberg .all reader, builds on Mike's pyall / pyAllConditioner
- `mbes_tools.wcd` — water column data (.wcd, paired with .all)
- `mbes_tools.kmwcd` — water column data (.kmwcd, paired with .kmall)
- `mbes_tools.mbsystem` — Python wrappers around MB-System CLI tools (mbinfo, mbgrid, datalist generation, format codes)

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
pip install -e ~/projects/mbes-tools
```

Run tests:

```bash
pytest
```

## License

MIT
