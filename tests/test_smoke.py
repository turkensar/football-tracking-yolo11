"""Basic smoke tests verifying package import and metadata."""

import futbolcv


def test_package_importable() -> None:
    """Ensure the futbolcv package imports and exposes a version string."""
    assert isinstance(futbolcv.__version__, str)
    assert len(futbolcv.__version__) > 0
