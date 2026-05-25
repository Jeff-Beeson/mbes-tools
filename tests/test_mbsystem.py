"""Tests for mbes_tools.mbsystem (MB-System CLI wrappers).

Pure unit tests for ``make_datalist`` and the stdout parser run unconditionally.
Integration tests for ``mbinfo`` and ``mbgrid`` skip cleanly when the binaries
are not on PATH.
"""
import shutil
from pathlib import Path

import pytest

from mbes_tools import mbsystem
from mbes_tools.mbsystem import (
    MB_FORMAT_ALL,
    MB_FORMAT_GSF,
    MB_FORMAT_KMALL,
    Bounds,
    MBGridResult,
    MBInfoResult,
)


FIXTURES = Path(__file__).parent / "fixtures"

HAS_MBINFO = shutil.which("mbinfo") is not None
HAS_MBGRID = shutil.which("mbgrid") is not None


# ---------------------------------------------------------------------------
# Pure unit tests (run everywhere).
# ---------------------------------------------------------------------------


def test_mbsystem_module_exposes_api():
    for name in [
        "mbinfo", "mbgrid", "make_datalist",
        "MBInfoResult", "MBGridResult", "Bounds",
        "MB_FORMAT_KMALL", "MB_FORMAT_ALL", "MB_FORMAT_GSF",
    ]:
        assert hasattr(mbsystem, name), f"missing {name}"


def test_format_constants():
    assert mbsystem.MB_FORMAT_KMALL == 261
    assert mbsystem.MB_FORMAT_ALL == 58
    assert mbsystem.MB_FORMAT_GSF == 121


def test_make_datalist_writes_expected_lines(tmp_path):
    files = [tmp_path / "a.kmall", tmp_path / "b.kmall"]
    for f in files:
        f.touch()

    out = tmp_path / "datalist.mb-1"
    result = mbsystem.make_datalist(files, out, format=MB_FORMAT_KMALL)

    assert result == out
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    for line, src in zip(lines, files):
        assert line.endswith(f" {MB_FORMAT_KMALL}")
        assert str(src.resolve()) in line


def test_make_datalist_refuses_overwrite_by_default(tmp_path):
    out = tmp_path / "list.mb-1"
    out.write_text("placeholder\n")

    with pytest.raises(FileExistsError):
        mbsystem.make_datalist([], out)

    # With overwrite=True, it succeeds.
    mbsystem.make_datalist([], out, overwrite=True)
    assert out.exists()


def test_mbgrid_rejects_unknown_grid_type(tmp_path):
    """Catch invalid grid_type before we even attempt to launch mbgrid."""
    fake = tmp_path / "doesnt_matter.kmall"
    fake.touch()
    with pytest.raises(ValueError):
        mbsystem.mbgrid(
            fake, output_root=tmp_path / "out",
            cell_size_m=100.0, grid_type="bogus",
        )


# ---------------------------------------------------------------------------
# Integration tests gated on MB-System binaries being installed.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_kmall():
    p = FIXTURES / "sample_dpdk027.kmall"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


@pytest.fixture
def sample_all():
    p = FIXTURES / "sample_nautilus.all"
    if not p.exists():
        pytest.skip(f"Fixture not present: {p}")
    return p


@pytest.mark.skipif(not HAS_MBINFO, reason="mbinfo not on PATH")
def test_mbinfo_parses_kmall(sample_kmall):
    info = mbsystem.mbinfo(sample_kmall)
    assert isinstance(info, MBInfoResult)

    # Format code should land on KMALL.
    assert info.format_code == MB_FORMAT_KMALL

    # DPDK027 is in Monterey Canyon — bounds should be in that ballpark.
    assert 35 < info.latitude_min < 38, info.raw_stdout
    assert -124 < info.longitude_min < -120, info.raw_stdout

    # Should have at least one record.
    assert info.number_of_records > 0

    # Depths should be positive (Kongsberg convention) and sensible (Monterey
    # Canyon has ~3000 m water depth in the lower canyon).
    assert info.depth_min > 0
    assert info.depth_max > info.depth_min


@pytest.mark.skipif(not HAS_MBINFO, reason="mbinfo not on PATH")
def test_mbinfo_parses_all(sample_all):
    info = mbsystem.mbinfo(sample_all)
    assert isinstance(info, MBInfoResult)
    assert info.format_code > 0
    assert info.number_of_records > 0


@pytest.mark.skipif(not HAS_MBINFO, reason="mbinfo not on PATH")
def test_mbinfo_raises_on_nonexistent_file(tmp_path):
    with pytest.raises(RuntimeError, match="mbinfo failed"):
        mbsystem.mbinfo(tmp_path / "does_not_exist.kmall")


@pytest.mark.skipif(not HAS_MBGRID, reason="mbgrid not on PATH")
def test_mbgrid_wrapper_invokes_command(sample_kmall, tmp_path):
    """Verify the wrapper assembles a valid mbgrid invocation and captures output.

    We deliberately do NOT require mbgrid to actually succeed — the .kmall
    fixture is just 2 #MRZ datagrams, which may be too sparse for mbgrid to
    fill any cells at 100 m resolution. Functional gridding gets tested
    once a denser fixture is in hand. For now we just verify wrapper plumbing.
    """
    out_root = tmp_path / "test_grid"
    result = mbsystem.mbgrid(
        sample_kmall,
        output_root=out_root,
        cell_size_m=100.0,
        format=MB_FORMAT_KMALL,
        grid_type="median",
    )

    # Wrapper plumbing: returns the right shape, ran the binary.
    assert isinstance(result, MBGridResult)
    assert result.output_path == out_root.with_suffix(".grd")
    # Either stdout or stderr should have content; mbgrid is verbose.
    assert (result.stdout or result.stderr), (
        "mbgrid produced no captured output at all — likely a wrapper bug"
    )

    # If mbgrid did succeed against the tiny fixture, the grid file should
    # exist. Either outcome is acceptable here.
    if result.returncode == 0:
        assert result.output_path.exists()
