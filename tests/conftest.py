"""Pytest conftest for tests — ensures the top-level solarseg package is importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Remove baseline/ from sys.path to prevent its solarseg/ from shadowing ours
sys.path = [p for p in sys.path if str(ROOT / "baseline") not in p]
sys.path.insert(0, str(ROOT))