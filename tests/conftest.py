import pytest

from leakage_probe.features import build_cases
from leakage_probe.generate import generate_fleet
from leakage_probe.probe import probe_fleet


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(5, 40, 7)


@pytest.fixture(scope="module")
def cases(fleet):
    return build_cases(fleet)


@pytest.fixture(scope="module")
def probe_results(fleet, cases):
    return probe_fleet(fleet, cases, n_perm=10, seed=5)
