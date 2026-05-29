"""
Root pytest configuration — adds each module's src/ directory to sys.path
so that test files can import directly (no sys.path hacks in test files).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent

for src_dir in sorted(ROOT.glob("0*/src")):
    sys.path.insert(0, str(src_dir))
