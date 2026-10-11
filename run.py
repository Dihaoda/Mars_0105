"""Run the source checkout without installing dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from work2.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
