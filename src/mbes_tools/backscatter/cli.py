"""Console entry points for the backscatter pipeline.

These thin wrappers are referenced by ``[project.scripts]`` in pyproject.toml:

    mbes-bs-table -> table_main   (generate sector/angle correction table)
    mbes-bs-apply -> apply_main   (patch corrections into KMALL seabed image)
    mbes-bs-gui   -> gui_main     (interactive Lambertian balancer GUI)
"""

from __future__ import annotations


def table_main() -> None:
    from mbes_tools.backscatter import table

    table.main()


def apply_main() -> None:
    from mbes_tools.backscatter import apply

    apply.main()


def gui_main() -> None:
    from mbes_tools.scripts.sector_balancer_gui import main

    main()
