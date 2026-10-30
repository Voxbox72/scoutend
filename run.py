"""Entry point that works without installing the package.

    python run.py build     # rebuild data/processed from Project Data/
    python run.py eval      # baselines vs reference model
    python run.py all

Equivalent to ``pip install -e .`` followed by ``scoutend build``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from scoutend.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
