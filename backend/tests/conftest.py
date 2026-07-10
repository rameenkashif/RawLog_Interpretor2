"""
conftest.py
-----------
Shared pytest fixtures for the petrophysics test suite.

Generates a synthetic-but-physically-plausible well log DataFrame so unit
tests can run without needing the real Z-02..Z-08 LAS files. The synthetic
curves are built from known "ground truth" lithology zones, so each
petrophysics function can be checked against expected qualitative behaviour
(e.g. shale zones should get high VSH, clean+wet zones should get high SWE
near 1, etc.) as well as exact-formula unit checks.

When real LAS files are dropped into backend/data/raw/, the same
petrophysics functions are exercised end-to-end via las_loader.py --
these fixtures are purely for fast, deterministic unit testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.config_loader import get_well_config


@pytest.fixture
def config():
    """Default field-wide config (no per-well override)."""
    return get_well_config(None)


@pytest.fixture
def synthetic_well_df():
    """A 200-sample synthetic well with three distinct lithology zones:

      0-60   : clean, wet sand   (low GR, low RHOB->high PHIT, low resistivity)
      60-120 : shale             (high GR, higher RHOB, low resistivity)
      120-200: clean, hydrocarbon-bearing sand (low GR, low RHOB, HIGH resistivity)

    Depth step = 0.1524 m (0.5 ft), typical wireline sample rate.
    """
    rng = np.random.default_rng(42)
    n = 200
    step = 0.1524
    dept = np.arange(n) * step + 1000.0

    gr = np.empty(n)
    rhob = np.empty(n)
    nphi = np.empty(n)
    resistivity = np.empty(n)
    dt = np.empty(n)

    # Zone 1: clean wet sand
    z1 = slice(0, 60)
    gr[z1] = rng.normal(30, 3, 60)
    rhob[z1] = rng.normal(2.25, 0.02, 60)
    nphi[z1] = rng.normal(0.22, 0.01, 60)
    resistivity[z1] = rng.normal(5, 0.5, 60)
    dt[z1] = rng.normal(85, 2, 60)

    # Zone 2: shale
    z2 = slice(60, 120)
    gr[z2] = rng.normal(120, 5, 60)
    rhob[z2] = rng.normal(2.45, 0.02, 60)
    nphi[z2] = rng.normal(0.30, 0.02, 60)
    resistivity[z2] = rng.normal(2, 0.3, 60)
    dt[z2] = rng.normal(100, 2, 60)

    # Zone 3: clean hydrocarbon-bearing sand
    z3 = slice(120, 200)
    gr[z3] = rng.normal(25, 3, 80)
    rhob[z3] = rng.normal(2.15, 0.02, 80)
    nphi[z3] = rng.normal(0.20, 0.01, 80)
    resistivity[z3] = rng.normal(80, 8, 80)
    dt[z3] = rng.normal(80, 2, 80)

    return pd.DataFrame(
        {
            "DEPT": dept,
            "GR": gr,
            "RESISTIVITY": resistivity,
            "RHOB": rhob,
            "NPHI": nphi,
            "DT": dt,
        }
    )


@pytest.fixture
def step_depth():
    return 0.1524


@pytest.fixture(autouse=True)
def _isolate_coordinate_repos(tmp_path, monkeypatch):
    """coordinate_calibration_service.py's default repo accessors
    (get_coordinate_calibration_repository / get_coordinate_tie_override_
    repository) are module-level singletons backed by real files under
    backend/data/coordinate_overrides/. Any test that exercises a well-tie
    path without explicitly injecting its own repo (e.g. via
    SegyVolume.get_well_tie or synthetic_seismogram_service, which don't
    expose that as a parameter) would otherwise read/write that REAL
    directory -- leaking calibration state across unrelated tests, and
    even across separate pytest invocations since it's a real file, not
    an in-memory fixture. Force every test's default singleton to a fresh
    per-test tmp_path instance instead; tests that pass their own
    calibration_repo/override_repo explicitly are unaffected."""
    import app.coordinate_calibration_repository as ccr
    import app.coordinate_tie_override_repository as ctor

    monkeypatch.setattr(
        ccr, "_repository", ccr.FileCoordinateCalibrationRepository(path=tmp_path / "calibration.json")
    )
    monkeypatch.setattr(
        ctor, "_repository", ctor.FileCoordinateTieOverrideRepository(base_dir=tmp_path / "overrides")
    )
