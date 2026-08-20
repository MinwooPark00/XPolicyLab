import sys
from pathlib import Path

# Match XPolicyLab's installed `XPolicyLab.py` package facade during local tests.
XPL_ROOT = Path(__file__).resolve().parents[3]
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))
