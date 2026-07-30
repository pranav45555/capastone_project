"""
pages package init — ensures the dashboard root is on sys.path
so that all page files can import from config, components, utils.
"""
import sys
import os

_dashboard_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _dashboard_root not in sys.path:
    sys.path.insert(0, _dashboard_root)
