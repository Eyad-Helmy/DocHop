"""
CodeArchGuard — Interactive Web Demo Launcher
Run this script to launch the full interactive web demo:
    python app.py
Or via CLI:
    cag ui
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from codearchguard.ui import main

if __name__ == "__main__":
    main()
