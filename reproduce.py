"""Repository-level entry point for artifact validation and reproduction."""

from pathlib import Path
import runpy


SCRIPT = Path(__file__).resolve().parent / "tools" / "reproduce.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
