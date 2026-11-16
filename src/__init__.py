"""Day 7 Agentic RAG package. Run modules from the repo root: python -m src.<module>."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_STARTER = str(ROOT / 'starter')
if _STARTER not in sys.path:  # starter/agentic.py imports `baseline` as a top-level module
    sys.path.insert(0, _STARTER)
