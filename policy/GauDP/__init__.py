"""Self-contained GauDP adapter for XPolicyLab."""

try:
    from .deploy import *
except ImportError:
    pass

try:
    from .model import *
except ImportError:
    pass
