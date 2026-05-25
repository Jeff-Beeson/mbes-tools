"""Python wrappers around MB-System CLI tools.

MB-System is a C-based external dependency that Jeff uses regularly.
This module wraps the common command-line calls via subprocess and parses
their output.

Planned coverage (v0):
- mbinfo wrapper (returns a parsed info dict)
- mbgrid wrapper (run + capture status)
- datalist (.mb-1) generation and parsing
- format code constants (mb56, mb57, mb58, mb88, mb121, mb261, ...)
- helpers for the geometry-grid + mask workflow from Monterey Canyon
  (slope, SD, roughness, threshold-masks)

Status: stub.
"""

import subprocess
from pathlib import Path
from typing import Optional


# MB-System format codes Jeff actually touches. Extend as needed.
MB_FORMAT_KMALL = 261  # Kongsberg .kmall


def _run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Internal: run an MB-System command, capture stdout/stderr, raise on non-zero exit."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


# TODO: implement mbinfo, mbgrid, datalist helpers, geometry-grid workflow.
