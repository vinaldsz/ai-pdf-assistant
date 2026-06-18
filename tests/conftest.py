"""Pytest configuration and shared fixtures."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that load ML models (~90 MB download on first run); "
        "deselect with -m 'not slow'",
    )
