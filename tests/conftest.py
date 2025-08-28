import sys
import pathlib

# Ensure the core project package is importable despite spaces in path
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "TopStepB - MAIN CLEAN - BEFORE SECOND STRATEGY"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
