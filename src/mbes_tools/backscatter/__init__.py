"""Backscatter normalization tools for Kongsberg multibeam data.

A three-stage, survey-agnostic pipeline built on :mod:`mbes_tools.kmall`:

1. :mod:`~mbes_tools.backscatter.table` — generate a sector/angle backscatter
   correction table from .kmall files (with optional geometry/slope/QC filters
   from :mod:`~mbes_tools.backscatter.qc`).
2. :mod:`~mbes_tools.backscatter.normalize` — Lambertian sector balancing
   (headless compute, also driven interactively by the balancer GUI).
3. :mod:`~mbes_tools.backscatter.apply` — patch the corrections into the
   ``SIsample_desidB`` seabed-image array of .kmall files.

Console entry points are defined in :mod:`mbes_tools.backscatter.cli`.
"""

from __future__ import annotations

__all__ = ["table", "qc", "normalize", "apply", "cli"]
