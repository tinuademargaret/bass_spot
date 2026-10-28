"""Community Cloud entrypoint with CPU-only dependencies beside it."""

import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
runpy.run_path(str(root / "app.py"), run_name="__main__")
