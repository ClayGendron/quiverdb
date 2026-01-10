"""Basic tests for QuiverDB."""

from quiverdb import __version__


def test_version() -> None:
    """Verify package version is set."""
    assert __version__ == "0.0.1"
