"""Puts the repository root on sys.path for `pytest tests/`; `requires` skips tests whose data is absent."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def requires(*paths: str):
    """A skipif mark for tests that need these repository-relative files (data/, dds_results/, runs/, ...)."""
    missing = [p for p in paths if not os.path.exists(os.path.join(REPO, p))]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)}")
