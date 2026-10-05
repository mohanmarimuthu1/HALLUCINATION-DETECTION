import pytest

from halludetect.api import resolve
from halludetect.observe import observer


@pytest.fixture(autouse=True)
def _fresh_catalogs():
    # resolve caches one catalog per key for the process; tests that
    # monkeypatch fetch_free_models need it refetched.
    resolve._catalogs.clear()
    yield
    resolve._catalogs.clear()


@pytest.fixture(autouse=True)
def _fresh_observer():
    observer.reset()
    yield
